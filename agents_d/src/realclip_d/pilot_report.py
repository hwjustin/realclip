"""Offline, read-only rendering of a first-agent pilot's saved artifacts.

Only index.html, REPORT.md and summary.json are written. No model or network
client is imported; raw English statements are displayed without translation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import quote, unquote, urlsplit


BASIS = {"direct_literature": "Direct literature observation", "indirect_literature": "Indirect literature evidence",
         "moa_inference": "MoA inference"}
RATIONALE_NOTICE = (
    "Rationale fields are visible, evidence-grounded justifications supplied in the structured output. "
    "They are not a transcript of private internal reasoning, which is unavailable to this report."
)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _objects(value: Any) -> list[dict]:
    """Structured display only; the complete untouched candidate is also shown."""
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def _read(path: Path, errors: list[str], *, required: bool = False) -> Any:
    if path.is_symlink():
        errors.append(f"Refusing to read symlink: {path.name}")
        return None
    if not path.is_file():
        if required:
            errors.append(f"Missing file: {path.name}")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        errors.append(f"Cannot read {path.name}: {type(exc).__name__}: {exc}")
        return None


def _lines(path: Path, errors: list[str]) -> list[dict]:
    if path.is_symlink():
        errors.append(f"Refusing to read symlink: {path.name}")
        return []
    if not path.is_file():
        return []
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("expected JSON object")
            rows.append(row)
        except ValueError as exc:
            errors.append(f"Cannot read {path.name}:{number}: {exc}")
    return rows


def _relative(root: Path, path: Path) -> str | None:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return None


def _href(value: Any) -> str | None:
    """Allow HTTP(S) or confined relative links; reject scripts and credentials."""
    if not isinstance(value, str) or not value or any(ord(c) < 32 for c in value):
        return None
    try:
        parsed = urlsplit(value)
        host = parsed.hostname
    except ValueError:
        return None
    if parsed.scheme:
        if parsed.scheme.lower() not in {"http", "https"} or not host or parsed.username or parsed.password:
            return None
        return value
    decoded = unquote(parsed.path)
    if parsed.netloc or decoded.startswith(("/", "\\")) or "\\" in decoded or ".." in Path(decoded).parts:
        return None
    return quote(value, safe="/-_.~")


def _link(label: str, value: Any) -> str:
    href = _href(value)
    return f'<a href="{escape(href, quote=True)}">{escape(label)}</a>' if href else escape(label)


def _pre(value: Any) -> str:
    return "<pre>" + escape(value if isinstance(value, str) else _json(value)) + "</pre>"


def _field(label: str, value: Any) -> str:
    return f"<dt>{escape(label)}</dt><dd>{_pre(value)}</dd>"


def _model_output(response: dict) -> dict | None:
    """Recover an unvalidated candidate when the runtime could not parse output."""
    for item in reversed(response.get("output", [])):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                try:
                    result = json.loads(content["text"])
                    if isinstance(result, dict) and ("hypotheses" in result or "morphology_hypotheses" in result):
                        return result
                except (ValueError, KeyError):
                    pass
    return None


def _has_image_or_cp(value: Any) -> bool:
    if isinstance(value, dict):
        if value.get("type") == "input_image" or value.get("tool") == "query_cp":
            return True
        if value.get("images") or value.get("cp_file"):
            return True
        return any(_has_image_or_cp(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_image_or_cp(item) for item in value)
    return isinstance(value, str) and value.startswith("data:image/")


def _receipt_summary(row: dict) -> dict:
    result = row.get("result") or {}
    data = result.get("data") or {}
    article = data.get("article") or {}
    model_data = (row.get("model_result") or {}).get("data") or {}
    model_view = model_data.get("_model_view")
    # Full-text retrieval and model exposure are different events. Keep the
    # exact selected windows and their metadata, never imply the entire paper
    # was shown merely because the underlying client downloaded it.
    exposure = {"model_result_recorded": "model_result" in row,
                "metadata": model_view,
                "note": ("Excerpts and omission metadata shown to the model; the complete raw receipt is saved as result in tools.jsonl."
                         if model_view else "Model excerpt metadata was not recorded; this does not establish that the model read the entire paper.")}
    if row.get("tool") == "read_article" and "model_result" in row:
        exposure.update(abstract_shown=model_data.get("abstract"),
                        body_windows_shown=model_data.get("body_windows"),
                        fulltext_status_reported_to_model=model_data.get("fulltext_status"))
        if "reading" in model_data:
            exposure.update(reading=model_data["reading"], sections=model_data.get("sections", []),
                            linked_materials=model_data.get("linked_materials", []))
    return {"tool": row.get("tool"), "arguments": row.get("arguments"),
            "keyword_arguments": row.get("keyword_arguments"), "status": result.get("status"),
            "message": result.get("message"), "query": data.get("query"),
            "hit_count": data.get("hit_count"), "article_title": article.get("title"),
            "article_id": article.get("id"), "article_source": article.get("source"),
            "pmcid": article.get("pmcid"), "doi": article.get("doi"), "publication_year": article.get("pubYear"),
            "abstract_available": bool(data.get("abstract")),
            "fulltext_status": data.get("fulltext_status"),
            "body_text_available": bool(data.get("body_text")),
            "model_exposure": exposure,
            "sources": result.get("sources", [])}


def _collect_sample(root: Path, entry: dict, ledger: dict) -> dict:
    sample_id = str(entry.get("sample_id", ""))
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", sample_id):
        raise ValueError("Manifest sample_id must be a single safe path component")
    sample_dir = root / "samples" / sample_id
    stage = sample_dir / "research"
    if _relative(root, stage) is None:
        raise ValueError("Sample stage must remain within run directory")
    errors: list[str] = []
    inputs = _read(stage / "input.json", errors, required=True)
    status = _read(sample_dir / "status.json", errors) or _read(stage / "status.json", errors) or {}
    initial_status = _read(sample_dir / "initial_status.json", errors)
    result = _read(stage / "result.json", errors)
    candidate = _read(stage / "candidate_result.json", errors)
    complete = _read(stage / "complete.json", errors)
    validation = _read(stage / "validation.json", errors)
    sources = _read(stage / "sources.json", errors) or []
    tools = _lines(stage / "tools.jsonl", errors)
    calls = _lines(stage / "calls.jsonl", errors)
    model_directories = [stage]
    continuations = []
    for directory in sorted(stage.glob("continuation_*")):
        if not directory.is_dir() or _relative(root, directory) is None:
            errors.append("Ignoring an invalid continuation or one outside the run directory")
            continue
        model_directories.append(directory)
        continuation_calls = _lines(directory / "calls.jsonl", errors)
        calls.extend(continuation_calls)
        continuations.append({"directory": _relative(root, directory),
                              "input": _read(directory / "input.json", errors),
                              "binding": _read(directory / "binding.json", errors),
                              "usage": _read(directory / "usage.json", errors),
                              "calls": continuation_calls,
                              "note": "Generation continued from saved evidence after the initial stage; additional model requests are recorded separately."})
    response_paths = [path for directory in model_directories for path in sorted((directory / "responses").glob("*.json"))]
    request_paths = [path for directory in model_directories for path in sorted((directory / "requests").glob("*.json"))]

    def turn_label(path: Path) -> str:
        prefix = path.parent.parent.relative_to(stage).as_posix()
        return (prefix + "/" if prefix != "." else "") + path.stem

    response_by_stem = {turn_label(path): path for path in response_paths}
    request_by_stem = {turn_label(path): path for path in request_paths}
    timeline, used_receipts, boundary_violations = [], set(), []
    if (sample_dir / "observation").exists():
        boundary_violations.append("An observation directory exists; the run cannot be described as first-agent-only")
    for stem in sorted(set(response_by_stem) | set(request_by_stem)):
        request_path, response_path = request_by_stem.get(stem), response_by_stem.get(stem)
        request = _read(request_path, errors) if request_path else None
        response = _read(response_path, errors) if response_path else None
        response = response or {}
        if request and _has_image_or_cp(request.get("input")):
            boundary_violations.append(f"Model request {stem} contains image or CP input")
        events = []
        for output in response.get("output", []):
            kind = output.get("type")
            if kind == "function_call":
                arguments = output.get("arguments")
                try:
                    arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
                except ValueError:
                    pass
                event = {"type": kind, "tool": output.get("name"), "arguments": arguments,
                         "call_id": output.get("call_id"), "receipt": None}
                for index, receipt in enumerate(tools):
                    if index not in used_receipts and receipt.get("tool") == output.get("name"):
                        event["receipt"] = _receipt_summary(receipt)
                        event["receipt_line"] = index + 1
                        used_receipts.add(index)
                        break
                if event["receipt"] is None:
                    errors.append(f"Turn {stem}, tool {output.get('name')}: saved receipt is missing")
                events.append(event)
            elif kind == "web_search_call":
                events.append({"type": kind, "status": output.get("status"), "action": output.get("action")})
        recovered = _model_output(response)
        if recovered is not None and candidate is None:
            candidate = recovered
        timeline.append({"turn": stem, "request": _relative(root, request_path) if request_path else None,
                         "response": _relative(root, response_path) if response_path else None,
                         "response_missing": response_path is None, "events": events,
                         "usage": response.get("raw_usage") or response.get("usage"),
                         "output_types": [row.get("type") for row in response.get("output", [])]})
    if _has_image_or_cp(inputs) or any(row.get("tool") == "query_cp" for row in tools):
        boundary_violations.append("Research input or tool records contain images or CP")
    if not result and not candidate:
        errors.append("No displayable structured output was saved; inspect raw responses and failure status")
    completed = bool(complete and result and validation and
                     (validation.get("structure_valid") or validation.get("references_valid")))
    selected = result or candidate
    modern = isinstance(selected, dict) and "compound_identification" in selected
    hypotheses = _objects(selected.get("morphology_hypotheses" if modern else "hypotheses")) if isinstance(selected, dict) else []
    request_ids = {call.get("request_id") for call in calls if call.get("request_id")}
    budget_rows = [row for row in ledger.get("requests", []) if row.get("request_id") in request_ids]
    unmatched = [_receipt_summary(row) | {"receipt_line": index + 1}
                 for index, row in enumerate(tools) if index not in used_receipts]
    return {"sample_id": sample_id, "name": entry.get("name", sample_id),
            "selection_reason": entry.get("selection_reason"), "known_moa": entry.get("known_moa", []),
            "input": inputs, "status": status, "completed": completed,
            "initial_status": initial_status, "continuations": continuations,
            "output_state": "completed" if completed else "unvalidated_candidate" if selected else "no_output",
            "result": selected, "validation": validation,
            "biological_accuracy_validated": False,
            "hypothesis_count": len(hypotheses),
            "basis_counts": {} if modern else {key: Counter(h.get("basis", "unspecified") for h in hypotheses)[key] for key in BASIS},
            "sources": sources, "timeline": timeline, "unmatched_tool_receipts": unmatched,
            "tool_calls": len(tools), "model_requests": len(request_paths),
            "calls": calls, "budget_requests": budget_rows,
            "cost_estimate_usd": sum(row.get("cost_estimate_usd", 0) for row in budget_rows),
            "boundary_violations": boundary_violations, "errors": errors,
            "artifacts": {path.name: _relative(root, path) for path in
                          (stage / "input.json", stage / "request_input.json", stage / "result.json",
                           stage / "candidate_result.json", stage / "complete.json", stage / "sources.json",
                           stage / "validation.json", stage / "tools.jsonl", stage / "calls.jsonl",
                           stage / "output_contract.json", stage / "output_schema.json", stage / "output_format.md",
                           sample_dir / "status.json", sample_dir / "initial_status.json") if path.is_file()}}


def _sources_html(root: Path, sample: dict) -> str:
    body = "<h3>Sources and raw caches</h3><ul>"
    for source in sample["sources"]:
        title = str(source.get("title") or source.get("source_id") or "source")
        body += "<li>" + _link(title, source.get("url"))
        body += " <code>" + escape(str(source.get("source_id", ""))) + "</code>"
        cache = source.get("cache_ref")
        if cache:
            path = root / "samples" / sample["sample_id"] / "research" / cache
            relative = _relative(root, path)
            body += " · " + (_link("Raw cache", relative) if relative and path.is_file() else "Cache missing or outside this run directory")
        body += " · " + escape(str(source.get("retrieved_at", "Retrieval time not recorded"))) + "</li>"
    return body + "</ul>"


def _sample_html(root: Path, sample: dict) -> str:
    text = f'<article id="{escape(sample["sample_id"], quote=True)}"><h2>{escape(str(sample["name"]))}</h2>'
    if sample["completed"] and (sample.get("validation") or {}).get("reference_validation") == "not_required":
        label = "Structure checks passed. Source provenance is saved separately; biological accuracy has not been independently validated."
    else:
        label = "Structure and reference checks passed; biological accuracy has not been independently validated" if sample["completed"] else "Validation incomplete: the original candidate output is shown without revision or filtering"
    text += f"<p class='notice'>{label}</p>"
    text += "<dl>" + _field("Selection reason", sample["selection_reason"]) + _field("Actual research input", sample["input"])
    text += _field("Stage status", sample["status"]) + _field("Evidence category counts", sample["basis_counts"])
    text += _field("Estimated cost for this compound (USD)", sample["cost_estimate_usd"]) + "</dl>"
    if sample["initial_status"] or sample["continuations"]:
        text += "<h3>Initial failure and continuation from saved evidence</h3>"
        text += "<p>The initial failure record is preserved. Continuation uses saved tool evidence, with model requests and costs recorded separately; the full timeline includes both the initial stage and continuation.</p>"
        text += _pre({"initial_status": sample["initial_status"], "continuations": sample["continuations"]})
    if sample["errors"] or sample["boundary_violations"]:
        text += "<h3>Missing records, errors, or boundary violations</h3>" + _pre(sample["errors"] + sample["boundary_violations"])
    text += "<h3>Model turns and actual retrieval process</h3>"
    for turn in sample["timeline"]:
        text += "<details open><summary>Turn " + escape(turn["turn"]) + " · "
        text += _link("Model request", turn["request"]) + " · " + _link("Raw response", turn["response"]) + "</summary>"
        text += _pre({key: turn[key] for key in ("output_types", "response_missing", "events", "usage")}) + "</details>"
    if sample["unmatched_tool_receipts"]:
        text += "<h4>Receipts not yet matched to model turns (in saved order)</h4>" + _pre(sample["unmatched_tool_receipts"])
    result = sample["result"]
    if isinstance(result, dict) and "compound_identification" in result:
        text += "<h3>Four-stage research result (original English)</h3><dl>"
        for key, label in (("compound_identification", "Stage 1: Compound Identification"),
                           ("compound_research_and_analysis", "Stage 2: Compound Research and Analysis"),
                           ("morphology_hypotheses", "Stage 3: Morphology Hypotheses"),
                           ("observation_guidance", "Stage 4: Observation Guidance")):
            text += _field(label, result.get(key))
        text += "</dl>"
        if not result.get("morphology_hypotheses"):
            text += "<p>No morphological hypotheses.</p>"
        text += "<details><summary>Complete structured output (no revisions or removed fields)</summary>" + _pre(result) + "</details>"
    elif isinstance(result, dict):
        text += "<h3>Four-step research summary (original English)</h3><dl>"
        for key, label in (("identity_summary", "Identity summary"), ("identity_context", "Step 1: identity and conditions"),
                           ("literature_review", "Step 2: direct literature observations"),
                           ("mechanism_review", "Step 3: MoA inference")):
            text += _field(label, result.get(key))
        text += "</dl><h3>Evidence facts (original English; model synthesis)</h3>"
        for fact in _objects(result.get("facts")):
            text += "<details open><summary>" + escape(str(fact.get("fact_id", "fact"))) + " · "
            text += escape(str(fact.get("evidence_level", "unknown"))) + "</summary>" + _pre(fact) + "</details>"
        text += "<h3>Step 4: morphological hypotheses and verification methods (original English)</h3>"
        for hypothesis in _objects(result.get("hypotheses")):
            text += "<section class='hypothesis'><h4>" + escape(str(hypothesis.get("hypothesis_id", "hypothesis")))
            text += " · " + escape(BASIS.get(hypothesis.get("basis"), str(hypothesis.get("basis", "Unclassified")))) + "</h4><dl>"
            for key, label in (("change", "Morphological change"), ("fact_ids", "Supporting fact IDs"), ("rationale", "Visible evidence-grounded rationale"),
                               ("applicable_conditions", "Applicable conditions"), ("uncertainty", "Uncertainty"),
                               ("where_to_check", "Where to inspect / channels"), ("verification", "Comparison, support, and conditions preventing assessment"),
                               ("disconfirming_observation", "Disconfirming observation")):
                text += _field(label, hypothesis.get(key))
            text += "</dl></section>"
        if not result.get("hypotheses"):
            text += "<p>No morphological hypotheses.</p>"
        text += "<h3>Limitations (original English)</h3>" + _pre(result.get("limitations", []))
        text += "<details><summary>Complete structured output (no revisions or removed fields)</summary>" + _pre(result) + "</details>"
    text += _sources_html(root, sample)
    text += "<h3>Saved raw artifacts</h3><ul>"
    text += "".join("<li>" + _link(name, path) + "</li>" for name, path in sample["artifacts"].items())
    return text + "</ul></article>"


def _markdown(summary: dict) -> str:
    parts = ["# D: first research agent · three-compound pilot", "",
             "Only saved research records are displayed; report generation does not start the second agent or call APIs. English facts and hypotheses are preserved verbatim.",
             "Structure and reference checks do not validate biological accuracy; failed outputs are explicitly marked as unvalidated candidates.", "",
             RATIONALE_NOTICE, "",
             "Detailed report: [index.html](index.html); structured summary: [summary.json](summary.json).", "",
             "## Budget and run boundaries", "", _fence(summary["budget"]), "", _fence(summary["boundaries"])]
    if summary.get("audit") is not None:
        parts += ["", "## Independent offline evidence review", "",
                  "The following review is saved separately from the original model output; its recorded scope and exclusions apply.", "",
                  _fence(summary["audit"]), "",
                  "Selected review record: [" + summary["audit_filename"] + "](" + summary["audit_filename"] + ")."]
    for sample in summary["samples"]:
        parts += ["", "## " + str(sample["name"]), "",
                  "Status: " + sample["output_state"], "", "Selection reason:", _fence(sample["selection_reason"]),
                  "Actual input:", _fence(sample["input"]), "Evidence category counts:", _fence(sample["basis_counts"]),
                  "Initial status and continuation from saved evidence:", _fence({"initial_status": sample["initial_status"], "continuations": sample["continuations"]}),
                  "### Retrieval process (model turns and actual tool receipts)", "", _fence(sample["timeline"])]
        if sample["errors"] or sample["boundary_violations"]:
            parts += ["", "Errors or missing records:", _fence(sample["errors"] + sample["boundary_violations"])]
        parts += ["", "### Evidence and morphological hypotheses (complete original English)", "", _fence(sample["result"]),
                  "", "### Sources", "", _fence(sample["sources"]), "", "Raw files:", ""]
        parts.extend(f"- [{name}]({_href(path)})" for name, path in sample["artifacts"].items())
    return "\n".join(parts) + "\n"


def _fence(value: Any) -> str:
    text = value if isinstance(value, str) else _json(value)
    fence = "`" * max(3, max((len(part) for part in text.split("\n") if part and set(part) == {"`"}), default=0) + 1)
    # A longer delimiter also handles embedded Markdown or HTML as plain text.
    while fence in text:
        fence += "`"
    return fence + "\n" + text + "\n" + fence


def generate_report(run_dir: Path) -> dict:
    """Read saved pilot material and write only the three report artifacts."""
    root = run_dir.resolve()
    errors: list[str] = []
    manifest = _read(root / "manifest.json", errors, required=True)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("samples"), list):
        raise ValueError("Expected manifest.json with a samples list")
    ledger = _read(root / "budget.json", errors) or {}
    run_status = _read(root / "status.json", errors)
    audit_filename = "audit.en.json" if (root / "audit.en.json").exists() else "audit.json"
    audit = _read(root / audit_filename, errors)
    presentation = _read(root / "presentation.en.json", errors)
    selection_overrides = {}
    if presentation is not None:
        if not isinstance(presentation, dict) or presentation.get("language") != "en" or not isinstance(presentation.get("selection_reasons"), dict):
            raise ValueError("Expected presentation.en.json with language 'en' and a selection_reasons object")
        known_sample_ids = {entry.get("sample_id") for entry in manifest["samples"]}
        selection_overrides = {sample_id: reason for sample_id, reason in presentation["selection_reasons"].items()
                               if sample_id in known_sample_ids and isinstance(reason, str) and reason.strip()}
    samples = [_collect_sample(root, entry | ({"selection_reason": selection_overrides[entry["sample_id"]]}
                                             if entry.get("sample_id") in selection_overrides else {}), ledger)
               for entry in manifest["samples"]]
    budget = {"cap_usd": manifest.get("budget_cap_usd", 5),
              "spent_estimate_usd": ledger.get("spent_estimate_usd"),
              "blocked": ledger.get("blocked"), "ledger": ledger,
              "note": "Usage estimate calculated from locally configured prices, not an OpenAI invoice; inspect the ledger for unknown usage and reservations."}
    boundaries = {"research_only": not any(row["boundary_violations"] for row in samples),
                  "observation_directories": [str(path.relative_to(root)) for path in root.glob("samples/*/observation")],
                  "manifest_second_agent_enabled": manifest.get("second_agent_enabled"),
                  "manifest_images_or_cp_supplied": manifest.get("images_or_cp_supplied"),
                  "recorded_second_agent_calls": run_status.get("second_agent_calls") if isinstance(run_status, dict) else None,
                  "note": "Checked only against saved requests and directories; report generation makes no network requests or model calls."}
    summary = {"schema_version": "1", "generated_at": datetime.now(timezone.utc).isoformat(),
               "run_directory": str(root), "settings": manifest.get("settings"), "budget": budget,
               "boundaries": boundaries, "run_status": run_status, "audit": audit,
               "audit_filename": audit_filename if audit is not None else None,
               "presentation_provenance": {"file": "presentation.en.json" if presentation is not None else None,
                                           "selection_reason_overrides": sorted(selection_overrides),
                                           "original_manifest_unchanged": True},
               "rationale_visibility": {"visible_evidence_grounded_rationale": True,
                                        "private_internal_reasoning_available": False,
                                        "note": RATIONALE_NOTICE},
               "errors": errors, "samples": samples}
    html = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'">
<title>D · Research agent three-compound pilot</title><style>
body{font:16px/1.65 system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1.2rem;color:#18222e;background:#f8fafc}
h1,h2,h3,h4{line-height:1.3}article{background:white;border:1px solid #d9e2ed;padding:1.5rem;margin:2rem 0;border-radius:12px}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f5f9;padding:.8rem;border-radius:6px;font:13px/1.6 ui-monospace,monospace}
dt{font-weight:650;margin-top:.8rem}dd{margin:.2rem 0 1rem}.notice{padding:1rem;background:#fff5da;border-left:4px solid #d99819}
details{margin:.8rem 0}.hypothesis{border-top:2px solid #a9c4de;margin-top:1.3rem}a{color:#145cb0}li{overflow-wrap:anywhere}
</style><h1>D: first research agent · three-compound pilot</h1>
<p>This report reads local records only. Facts and hypotheses preserve the model's original English output; biological accuracy has not been independently validated. Candidates with incomplete validation are preserved in full.</p>
<p>Each model turn lists saved tool calls and article access status. fulltext_status describes retrieval; model_exposure shows the actual reading pages and outline, or historical abstract/body windows, passed to the model. Successful full-text retrieval does not mean the model read the entire paper. Text exposure does not establish model comprehension.</p>"""
    html += "<p>" + escape(RATIONALE_NOTICE) + "</p>"
    html += "<p>" + _link("Markdown report", "REPORT.md") + " · " + _link("Structured summary", "summary.json") + "</p>"
    html += "<h2>Budget and boundaries</h2>" + _pre({"budget": budget, "boundaries": boundaries, "errors": errors})
    if audit is not None:
        html += "<h2>Independent offline evidence review</h2><p>The following review is saved separately from the original model output; its recorded scope and exclusions apply. "
        html += _link("Selected review record", audit_filename) + "</p>" + _pre(audit)
    html += "<nav><ul>" + "".join('<li><a href="#' + escape(row["sample_id"], quote=True) + '">' + escape(str(row["name"])) + "</a></li>" for row in samples) + "</ul></nav>"
    html += "".join(_sample_html(root, sample) for sample in samples) + "</html>\n"
    for name, content in (("summary.json", _json(summary) + "\n"), ("REPORT.md", _markdown(summary)), ("index.html", html)):
        if (root / name).is_symlink():
            raise ValueError(f"Refusing to overwrite report symlink: {name}")
        (root / name).write_text(content, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = generate_report(args.run_dir)
    print(_json({"report": str(args.run_dir.resolve() / "index.html"),
                 "samples": len(summary["samples"]), "errors": summary["errors"]}))


if __name__ == "__main__":
    main()
