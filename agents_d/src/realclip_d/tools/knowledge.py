"""Small official API clients; no model logic or prior experiment dependencies."""
from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET
from functools import wraps
from typing import Any
from urllib.parse import quote, urljoin, urlparse

from ..contracts import CompoundIdentity, Source, ToolResult
from .http import CachedHTTP, HTTPRecord, TransportError
from .article_reading import ArticleMode, DEFAULT_MAX_CHARS, parse_article, reading_selection, validate_reading


class APIError(RuntimeError):
    def __init__(self, result: ToolResult):
        super().__init__(result.message)
        self.result = result


def api_boundary(function):
    """Keep malformed upstream payloads auditable instead of losing tool receipts."""
    @wraps(function)
    def wrapped(self, *args, **kwargs):
        try:
            return function(self, *args, **kwargs)
        except APIError as exc:
            return exc.result
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            return ToolResult(status="error", sources=list(self.sources),
                              message=f"Unexpected official API payload: {type(exc).__name__}: {exc}")
        except OSError as exc:
            return ToolResult(status="unavailable", sources=list(self.sources),
                              message=f"Response cache unavailable: {exc}")
    return wrapped


class OfficialClient:
    def __init__(self, http: CachedHTTP):
        self.http = http
        self.sources: list[Source] = []

    def _get(self, url: str, params: dict[str, Any] | None = None) -> tuple[HTTPRecord, Source]:
        try:
            record = self.http.get(url, params=params)
        except TransportError as exc:
            raise APIError(ToolResult(status="unavailable", message=str(exc))) from exc
        source_id = "api_" + hashlib.sha256(record.url.encode() + b"\0" + record.retrieved_at.encode() + b"\0" + record.body).hexdigest()[:24]
        source = Source(source_id=source_id, url=record.url,
                        retrieved_at=record.retrieved_at, cache_ref=record.cache_ref)
        if source_id not in {row.source_id for row in self.sources}:
            self.sources.append(source)
        if 300 <= record.status_code < 400:
            raise APIError(ToolResult(status="unavailable", sources=[source], message="API redirect was not followed; only fixed official endpoints are allowed"))
        if record.status_code >= 400:
            status = "not_found" if record.status_code == 404 else "rate_limited" if record.status_code == 429 else "unavailable"
            raise APIError(ToolResult(status=status, sources=[source], message=f"HTTP {record.status_code}"))
        return record, source

    def _json(self, url: str, params: dict[str, Any] | None = None) -> tuple[dict[str, Any], Source]:
        record, source = self._get(url, params)
        try:
            payload = record.json()
        except (ValueError, UnicodeDecodeError) as exc:
            raise APIError(ToolResult(status="error", sources=[source], message="API did not return valid JSON")) from exc
        if not isinstance(payload, dict):
            raise APIError(ToolResult(status="error", sources=[source], message="Expected an API JSON object"))
        return payload, source


class PubChemClient(OfficialClient):
    PUG_REST = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"
    PUG_VIEW = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound"

    @api_boundary
    def resolve_compound(self, identity: CompoundIdentity) -> ToolResult:
        """Resolve identity without silently selecting among multiple structures."""
        sources: list[Source] = []
        if identity.inchikey:
            namespace, identifier = "inchikey", identity.inchikey
        elif identity.pubchem_cid:
            namespace, identifier = "cid", str(identity.pubchem_cid)
        elif identity.smiles:
            namespace, identifier = "smiles", identity.smiles
        elif identity.name:
            namespace, identifier = "name", identity.name
        else:
            return ToolResult(status="unavailable", message="PubChem resolution requires an InChIKey, CID, SMILES or name")
        path = f"{self.PUG_REST}/{namespace}/{quote(identifier, safe='')}/property/InChIKey,InChI,IUPACName,Title,MolecularFormula/JSON"
        try:
            data, source = self._json(path)
            sources.append(source)
            rows = data.get("PropertyTable", {}).get("Properties", [])
            if not rows:
                return ToolResult(status="not_found", sources=sources, data={"candidates": []})
            if identity.inchikey:
                exact = [row for row in rows if row.get("InChIKey") == identity.inchikey]
                if not exact:
                    return ToolResult(status="ambiguous", sources=sources, data={"candidates": rows, "identity_verified": False}, message="Returned structures do not match the complete InChIKey")
                rows = exact
            if identity.pubchem_cid:
                rows = [row for row in rows if row.get("CID") == identity.pubchem_cid]
                if not rows:
                    return ToolResult(status="ambiguous", sources=sources, data={"identity_verified": False}, message="Provided CID and returned structure disagree")
            if len(rows) != 1:
                return ToolResult(status="ambiguous", sources=sources, data={"candidates": rows, "identity_verified": False}, message="Multiple candidate structures; supply a complete InChIKey or CID")
            selected = rows[0]
            synonyms: list[str] = []
            warnings: list[str] = []
            try:
                synonym_data, source = self._json(f"{self.PUG_REST}/cid/{selected['CID']}/synonyms/JSON")
                sources.append(source)
                for item in synonym_data.get("InformationList", {}).get("Information", []):
                    synonyms.extend(item.get("Synonym", []))
            except APIError as exc:
                sources.extend(exc.result.sources)
                warnings.append("Synonyms unavailable: " + str(exc))
            return ToolResult(status="ok", sources=sources, data={
                "compound": selected, "synonyms": synonyms[:100],
                "identity_verified": bool(identity.inchikey and selected.get("InChIKey") == identity.inchikey),
                "verification_basis": "complete_inchikey" if identity.inchikey else "provided_cid" if identity.pubchem_cid else "unique_candidate_only",
                "warnings": warnings,
            })
        except APIError as exc:
            return exc.result.model_copy(update={"sources": sources + exc.result.sources})

    @api_boundary
    def get_annotations(self, cid: int, heading: str = "Pharmacology and Biochemistry") -> ToolResult:
        if cid <= 0:
            return ToolResult(status="error", message="CID must be positive")
        try:
            data, source = self._json(f"{self.PUG_VIEW}/{cid}/JSON", {"heading": heading})
            return ToolResult(status="ok", sources=[source], data={"cid": cid, "heading": heading, "record": data.get("Record", data)})
        except APIError as exc:
            return exc.result


class ChEMBLClient(OfficialClient):
    BASE = "https://www.ebi.ac.uk/chembl/api/data/"

    def _pages(self, resource: str, params: dict[str, Any], limit: int) -> tuple[list[dict[str, Any]], list[Source], bool]:
        url = f"{self.BASE}{resource}.json"
        params = {**params, "limit": min(limit, 100), "offset": 0}
        rows: list[dict[str, Any]] = []
        sources: list[Source] = []
        plural = {"activity": "activities", "mechanism": "mechanisms", "molecule": "molecules"}[resource]
        for _ in range(20):
            payload, source = self._json(url, params)
            sources.append(source)
            batch = payload.get(plural, [])
            rows.extend(batch[:max(0, limit-len(rows))])
            next_url = payload.get("page_meta", {}).get("next")
            if not next_url:
                return rows, sources, False
            if len(rows) >= limit:
                return rows, sources, True
            resolved = urljoin(url, next_url)
            if urlparse(resolved).netloc != urlparse(self.BASE).netloc or not urlparse(resolved).path.startswith("/chembl/api/data/"):
                raise APIError(ToolResult(status="error", sources=sources, message="Unexpected ChEMBL pagination target"))
            url, params = resolved, None
        return rows, sources, True

    @api_boundary
    def get_evidence(self, inchikey: str | None = None, chembl_id: str | None = None, max_records: int = 100) -> ToolResult:
        if not 1 <= max_records <= 500:
            return ToolResult(status="error", message="max_records must be 1..500")
        if chembl_id and not re.fullmatch(r"CHEMBL\d+", chembl_id):
            return ToolResult(status="error", message="Invalid ChEMBL molecule ID")
        sources: list[Source] = []
        try:
            if chembl_id:
                molecule, source = self._json(f"{self.BASE}molecule/{chembl_id}.json")
                sources.append(source)
                if molecule.get("molecule_chembl_id") != chembl_id:
                    return ToolResult(status="ambiguous", sources=sources, data={"candidates": [molecule], "identity_verified": False}, message="Returned molecule does not match the provided ChEMBL ID")
                candidates = [molecule]
            elif inchikey:
                candidates, pages, _ = self._pages("molecule", {"molecule_structures__standard_inchi_key": inchikey}, max_records)
                sources.extend(pages)
            else:
                return ToolResult(status="error", message="Provide a complete InChIKey or ChEMBL ID")
            if inchikey:
                exact = [row for row in candidates if (row.get("molecule_structures") or {}).get("standard_inchi_key") == inchikey]
                if candidates and not exact:
                    return ToolResult(status="ambiguous", sources=sources, data={"candidates": candidates, "identity_verified": False}, message="Returned molecule does not match the complete InChIKey")
                candidates = exact
            if not candidates:
                return ToolResult(status="not_found", sources=sources, data={"identity_verified": False})
            if len(candidates) != 1:
                return ToolResult(status="ambiguous", sources=sources, data={"candidates": candidates})
            molecule = candidates[0]
            exact_id = molecule["molecule_chembl_id"]
            parent_id = (molecule.get("molecule_hierarchy") or {}).get("parent_chembl_id") or exact_id
            mechanisms, pages, mechanism_truncated = self._pages("mechanism", {"molecule_chembl_id": parent_id}, max_records)
            sources.extend(pages)
            activities, pages, activity_truncated = self._pages("activity", {"molecule_chembl_id": exact_id}, max_records)
            sources.extend(pages)
            linked: dict[str, list[dict[str, Any]]] = {"assays": [], "targets": [], "documents": []}
            missing: list[str] = []
            for resource, field, plural in [("assay", "assay_chembl_id", "assays"), ("target", "target_chembl_id", "targets"), ("document", "document_chembl_id", "documents")]:
                ids = sorted({row[field] for row in activities + mechanisms if row.get(field)})
                for linked_id in ids[:20]:
                    if not re.fullmatch(r"CHEMBL\d+", linked_id):
                        missing.append(f"{resource}:{linked_id}:invalid_id")
                        continue
                    try:
                        record, source = self._json(f"{self.BASE}{resource}/{linked_id}.json")
                        sources.append(source)
                        linked[plural].append(record)
                    except APIError as exc:
                        sources.extend(exc.result.sources)
                        missing.append(f"{resource}:{linked_id}:{exc.result.status}")
                if len(ids) > 20:
                    missing.append(f"{resource}:linked_record_limit_20")
            return ToolResult(status="ok", sources=sources, data={
                "molecule": molecule, "exact_molecule_id": exact_id,
                "parent_molecule_id": parent_id, "identity_verified": bool(inchikey),
                "mechanism_scope": "parent_aggregated", "activity_scope": "exact_molecule",
                "mechanisms": mechanisms, "activities": activities, **linked,
                "truncated": {"mechanisms": mechanism_truncated, "activities": activity_truncated},
                "missing_linked_records": missing,
            })
        except APIError as exc:
            return exc.result.model_copy(update={"sources": sources + exc.result.sources})


class EuropePMCClient(OfficialClient):
    BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest/"

    @api_boundary
    def search(self, query: str, limit: int = 10, cursor: str | None = None,
               result_offset: int = 0) -> ToolResult:
        if not query.strip() or not 1 <= limit <= 100:
            return ToolResult(status="error", message="Provide a query and limit 1..100")
        if type(result_offset) is not int or result_offset < 0:
            return ToolResult(status="error", message="result_offset must be a nonnegative integer")
        if cursor is not None and (not isinstance(cursor, str) or not cursor):
            return ToolResult(status="error", message="cursor must be a nonempty Europe PMC cursor or null")
        try:
            params = {"query": query, "format": "json", "resultType": "core", "pageSize": limit}
            if cursor is not None:
                params["cursorMark"] = cursor
            data, source = self._json(f"{self.BASE}search", params)
            results = data.get("resultList", {}).get("result", [])
            if result_offset > len(results):
                return ToolResult(status="error", sources=[source], message="result_offset exceeds the retrieved result count")
            next_cursor = data.get("nextCursorMark")
            if not results or next_cursor == cursor:
                next_cursor = None
            return ToolResult(status="ok" if results else "not_found", sources=[source], data={
                "query": query, "limit": limit, "cursor": cursor, "result_offset": result_offset,
                "hit_count": data.get("hitCount", 0), "results": results, "next_cursor": next_cursor})
        except APIError as exc:
            return exc.result

    @api_boundary
    def read_article(self, article_id: str, source: str = "MED", mode: ArticleMode = "overview",
                     section_ids: list[str] | None = None, offset: int = 0,
                     max_chars: int = DEFAULT_MAX_CHARS, auto_read: bool = False) -> ToolResult:
        error = validate_reading(mode, section_ids, offset, max_chars)
        if error:
            return ToolResult(status="error", message=error)
        source = source.upper()
        if not re.fullmatch(r"[A-Za-z0-9._-]+", article_id) or not re.fullmatch(r"[A-Z]+", source):
            return ToolResult(status="error", message="Invalid article identifier")
        query = f"EXT_ID:{article_id} AND SRC:{source}" if not article_id.upper().startswith("PMC") else f"PMCID:{article_id.upper()}"
        result = self.search(query, limit=2)
        if result.status != "ok":
            return result
        rows = result.data["results"]
        if len(rows) != 1:
            return ToolResult(status="ambiguous", sources=result.sources, data={"candidates": rows}, message="Article identifier resolved to multiple records")
        article = rows[0]
        data: dict[str, Any] = {
            "article": article, "abstract": article.get("abstractText"),
            "fulltext_status": "not_in_pmc", "fulltext_xml": None,
            "body_text": None, "mode": mode,
            "document": {"text": "", "sections": []},
        }
        sources = list(result.sources)
        pmcid = article.get("pmcid")
        if pmcid and re.fullmatch(r"PMC\d+", pmcid):
            try:
                record, receipt = self._get(f"{self.BASE}{pmcid}/fullTextXML")
                sources.append(receipt)
                data["fulltext_xml"] = record.text
                document = parse_article(record.text, data["abstract"])
                if not data["abstract"]:
                    data["abstract"] = document["abstract_text"] or None
                data.update(fulltext_status="accessible" if document["body_text"] else "no_readable_body",
                            document=document, body_text=document["body_text"] or None)
            except APIError as exc:
                sources.extend(exc.result.sources)
                data.update(fulltext_status="unavailable", fulltext_error=exc.result.message)
            except ET.ParseError:
                data.update(fulltext_status="invalid_xml", fulltext_error="Full-text response is not valid XML")
        if mode != "overview" and data["fulltext_status"] != "accessible":
            return ToolResult(status="unavailable", sources=sources, data=data,
                              message="Requested article text is unavailable; use overview for the available abstract and access status.")
        try:
            from ..tool_language import article_presentation
            presentation = article_presentation(data)
            data["reading"] = reading_selection(presentation, article_id, source, mode, section_ids, offset,
                                                  max_chars, auto_read=auto_read)
        except ValueError as exc:
            return ToolResult(status="error", sources=sources, data=data, message=str(exc))
        return ToolResult(status="ok", sources=sources, data=data)
