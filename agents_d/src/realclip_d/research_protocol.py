"""Four-step research provenance checks; never a biological accuracy score."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .contracts import ResearchOutput, ResearchResultV2, ResearchResultV3, ResearchResultV4, SampleInput, Source
from .storage import read_json


def _identity_receipts(sample: SampleInput, receipts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Related-compound queries must never establish the sample's identity."""
    trusted_keys = {sample.compound.inchikey} if sample.compound.inchikey else set()
    qualifying = []
    for receipt in receipts:
        if receipt.get("tool") != "resolve_compound":
            continue
        body = receipt.get("result", {})
        compound = body.get("data", {}).get("compound", {})
        if body.get("status") == "ok" and compound:
            if sample.compound.inchikey and compound.get("InChIKey") != sample.compound.inchikey:
                continue
            if sample.compound.pubchem_cid and compound.get("CID") != sample.compound.pubchem_cid:
                continue
            if compound.get("InChIKey"):
                trusted_keys.add(compound["InChIKey"])
        qualifying.append(body)
    for receipt in receipts:
        if receipt.get("tool") != "get_chembl_evidence":
            continue
        body = receipt.get("result", {})
        data = body.get("data", {})
        molecule = data.get("molecule", {})
        key = (molecule.get("molecule_structures") or {}).get("standard_inchi_key")
        identifier = data.get("exact_molecule_id") or molecule.get("molecule_chembl_id")
        if sample.compound.inchikey and key != sample.compound.inchikey:
            continue
        if sample.compound.chembl_id and identifier != sample.compound.chembl_id:
            continue
        if (key and key in trusted_keys) or (sample.compound.chembl_id and identifier == sample.compound.chembl_id):
            qualifying.append(body)
    return qualifying


def research_receipts(directory: Path) -> list[dict[str, Any]]:
    import json

    path = directory / "tools.jsonl"
    receipts = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    source_path = directory / "sources.json"
    sources = read_json(source_path) if source_path.is_file() else []
    # Hosted web search runs server-side, so it is represented by SDK responses,
    # not by a Python function-tool receipt.
    for path in sorted((directory / "responses").glob("*.json")):
        for item in read_json(path).get("output", []):
            if item.get("type") == "web_search_call":
                receipts.append({"tool": "web_search", "result": {
                    "status": item.get("status"), "data": item,
                    "sources": [source for source in sources if source.get("kind") == "web_search"
                                and source.get("cache_ref") == f"responses/{path.name}"],
                }})
    return receipts


def validate_research_protocol(result: ResearchOutput, sample: SampleInput,
                               tool_receipts: list[dict[str, Any]]) -> None:
    if isinstance(result, ResearchResultV4):
        if result.sample_id != sample.sample_id:
            raise ValueError("Research result belongs to another sample")
        return  # No source-ID, URL, title, or reading-completion gates.
    if isinstance(result, ResearchResultV3):
        if result.sample_id != sample.sample_id:
            raise ValueError("Research result belongs to another sample")
        identity = _identity_receipts(sample, tool_receipts)
        if not identity:
            raise ValueError("Stage 1 requires an actual identity-tool attempt")
        identity_sources = {source["source_id"] for receipt in identity for source in receipt.get("sources", [])}
        if not set(result.compound_identification.source_ids) <= identity_sources:
            raise ValueError("Compound identification sources must come from identity-tool receipts")
        usable = set()
        web_urls = set()
        for receipt in tool_receipts:
            body = receipt.get("result", {})
            if receipt.get("tool") == "list_sources":
                continue  # A registry is not an independent successful retrieval.
            if body.get("status") == "ok":
                usable.update(source["source_id"] for source in body.get("sources", []))
            elif receipt.get("tool") == "web_search" and body.get("status") == "completed":
                usable.update(source["source_id"] for source in body.get("sources", []))
                action = body.get("data", {}).get("action", {})
                web_urls.update(source["url"] for source in action.get("sources", []) if source.get("url"))
                if action.get("url"):
                    web_urls.add(action["url"])
        usable.update(source.id for source in result.sources if source.url in web_urls)
        cited = {sid for finding in result.compound_research_and_analysis.findings for sid in finding.source_ids}
        cited.update(sid for hypothesis in result.morphology_hypotheses for sid in hypothesis.source_ids)
        if not cited <= usable:
            raise ValueError("Findings and hypotheses must cite successful retrieval receipts, not failed-only sources")
        # The simplified contract has no access/status/evidence enums. Do not
        # infer such labels from prose or silently reinstate V2 requirements.
        # Authentic-source checks and guidance links do not validate biology.
        return
    if not isinstance(result, ResearchResultV2):
        return  # Historical V1 artifacts remain V1; no invented V2 evidence.
    review = result.identity_context
    if review.context != sample.context:
        raise ValueError("Identity/context review must preserve the supplied experimental conditions")
    expected_unknown = {key for key in ("cell_line", "dose", "treatment_duration")
                        if not getattr(sample.context, key) or not getattr(sample.context, key).strip()}
    if set(review.unknown_conditions) != expected_unknown or len(review.unknown_conditions) != len(expected_unknown):
        raise ValueError("Unknown experimental conditions must be explicitly and accurately recorded")

    identity = _identity_receipts(sample, tool_receipts)
    if not identity:
        raise ValueError("Step 1 requires an actual identity-tool attempt")
    identity_sources = {source["source_id"] for receipt in identity for source in receipt.get("sources", [])}
    if not set(review.source_ids) <= identity_sources:
        raise ValueError("Identity review sources must come from identity-tool receipts")
    if review.status in {"verified", "resolved"}:
        successes = [receipt for receipt in identity if receipt.get("status") == "ok"]
        successful_sources = {source["source_id"] for receipt in successes for source in receipt.get("sources", [])}
        if not successes or not review.source_ids or not set(review.source_ids) <= successful_sources:
            raise ValueError("Resolved identity requires successful retrieval and identity sources")
        verified_sources = {source["source_id"] for receipt in successes
                            if receipt.get("data", {}).get("identity_verified") is True
                            for source in receipt.get("sources", [])}
        if review.status == "verified" and not set(review.source_ids) & verified_sources:
            raise ValueError("Verified identity requires a cited exact verified identity receipt")
    if review.status in {"ambiguous", "unresolved"} and any(
        fact.origin == "literature" and fact.compound_scope == "exact_compound" for fact in result.facts
    ):
        raise ValueError("Unresolved identity cannot establish exact-compound literature evidence")

    actual_moa = {label.strip() for label in sample.known_moa}
    declared_moa = {fact.statement for fact in result.facts if fact.origin == "provided_moa"}
    if actual_moa != declared_moa:
        raise ValueError("Provided-MOA facts must retain exactly the supplied MOA labels")

    article_accesses: list[tuple[set[str], dict[str, set[str]]]] = []
    for receipt in tool_receipts:
        body = receipt.get("result", {})
        if receipt.get("tool") != "read_article" or body.get("status") != "ok":
            continue
        data = body.get("data", {})
        accesses: dict[str, set[str]] = {}
        for source in body.get("sources", []):
            source_id, url = source["source_id"], source.get("url", "")
            available = accesses.setdefault(source_id, set())
            if data.get("abstract") and not url.rstrip("/").endswith("fullTextXML"):
                available.add("abstract")
            if data.get("fulltext_status") == "accessible" and data.get("body_text") and url.rstrip("/").endswith("fullTextXML"):
                available.add("full_text")
        article_accesses.append((set(accesses), accesses))
    for fact in result.facts:
        if fact.origin != "literature":
            continue
        cited = set(fact.source_ids)
        covered: set[str] = set()
        for companions, accesses in article_accesses:
            access_bearing = {source_id for source_id, levels in accesses.items() if fact.access in levels}
            # A paper's metadata and XML can be co-cited. At least one cited
            # source must actually bear the declared access; metadata alone
            # still cannot establish full-text reading. Unrelated receipts
            # never inherit another paper's access.
            if cited & access_bearing:
                covered.update(companions)
        if not cited <= covered:
            raise ValueError("Literature evidence requires matching read_article access receipts, not a search hit or inaccessible full text")

    attempted = any(receipt.get("tool") in {"search_literature", "read_article", "web_search"} for receipt in tool_receipts)
    if result.literature_review.status == "not_searched":
        if attempted:
            raise ValueError("Literature review says not_searched despite recorded research attempts")
    elif not attempted:
        raise ValueError("Step 2 literature status requires an actual search or article-reading attempt")


def mock_research_v2(sample: SampleInput, sources: list[Source]) -> ResearchResultV2:
    """An explicitly empty scientific result used by the offline SDK model."""
    facts = [{"fact_id": f"provided_moa_{index + 1}", "statement": label.strip(),
              "source_ids": [], "evidence_level": "provided_moa", "origin": "provided_moa",
              "compound_scope": "unknown", "access": "input",
              "conditions": "Supplied experiment context; no independent verification.",
              "limitations": "Input prior, not a measured morphology or a verified mechanism."}
             for index, label in enumerate(dict.fromkeys(sample.known_moa))]
    return ResearchResultV2(
        sample_id=sample.sample_id,
        identity_summary="Synthetic mock fixture; no scientific identity resolution.",
        sources=sources, facts=facts, hypotheses=[],
        identity_context={"status": "unresolved", "source_ids": [source.source_id for source in sources],
                          "context": sample.context.model_dump(),
                          "unknown_conditions": [key for key in ("cell_line", "dose", "treatment_duration")
                                                 if not getattr(sample.context, key) or not getattr(sample.context, key).strip()],
                          "conflicts": ["Synthetic identity lookup is not a real chemical identification."]},
        literature_review={"status": "no_morphology_found", "search_summary": "Synthetic empty literature fixture; no real search was performed.",
                           "fact_ids": [], "limitations": ["No scientific literature was retrieved."]},
        mechanism_review={"status": "no_supported_inference" if facts else "not_applicable",
                          "fact_ids": [fact["fact_id"] for fact in facts],
                          "summary": "No biological inference is made by this scripted model.",
                          "limitations": ["Input MOA labels, if present, remain unverified priors."]},
        limitations=["SYNTHETIC MOCK OUTPUT: plumbing verification only; no scientific research was performed."],
    )


def mock_research_v3(sample: SampleInput, sources: list[Source]) -> ResearchResultV3:
    """Empty scientific output for offline plumbing checks, never real research."""
    return ResearchResultV3(
        sample_id=sample.sample_id,
        compound_identification={
            "summary": "Synthetic mock fixture; no actual compound identity has been established.",
            "source_ids": [source.source_id for source in sources],
        },
        compound_research_and_analysis={
            "findings": [],
            "research_gaps": "SYNTHETIC MOCK OUTPUT: no scientific research was performed. "
                             "Supplied MOA is an unverified input; no morphology is inferred.",
        },
        morphology_hypotheses=[], observation_guidance=[],
        sources=[{"id": source.source_id, "title": source.title, "url": source.url} for source in sources],
    )
