from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from .synonyms import expand_query_synonyms

MAX_QUERY_LANES = 6

_IDENTIFIER_PATTERNS = (
    re.compile(r"[A-Za-z0-9_./\\-]+\.[A-Za-z0-9]{1,6}\b"),
    re.compile(r"\bv?\d+\.\d+(?:\.\d+)?\b"),
    re.compile(r"\b[A-Z]{2,}[A-Z0-9_-]*\d+[A-Z0-9_-]*\b"),
    re.compile(r"\b[a-z]+(?:_[a-z0-9]+)+\b"),
    re.compile(r"\b[a-z]+(?:[A-Z][a-z0-9]+)+\b"),
)
_CONJUNCTION_SPLIT = re.compile(r"[?.!;]+|\b(?:and|also|then|plus|while|versus|vs)\b", re.IGNORECASE)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


@lru_cache(maxsize=1)
def _get_nlp() -> Any | None:
    try:
        import spacy
    except ImportError:
        return None
    try:
        return spacy.load("en_core_web_sm")
    except Exception:
        return None


def mask_identifiers(text: str) -> tuple[str, list[str]]:
    masked = text
    identifiers: list[str] = []
    for pattern in _IDENTIFIER_PATTERNS:
        for match in pattern.findall(masked):
            if match not in identifiers:
                identifiers.append(match)
        masked = pattern.sub(" IDENTIFIER ", masked)
    return _normalize(masked), identifiers


def _spacy_clauses(text: str) -> list[str]:
    nlp = _get_nlp()
    if nlp is None:
        return []
    clauses: list[str] = []
    doc = nlp(text)
    for sent in doc.sents:
        predicates = [token for token in sent if token.pos_ in {"VERB", "AUX"} and token.dep_ in {"ROOT", "conj"}]
        if len(predicates) <= 1:
            clauses.append(sent.text)
            continue
        subject = ""
        for chunk in sent.noun_chunks:
            if chunk.root.dep_ in {"nsubj", "nsubjpass"}:
                subject = chunk.text
                break
        for predicate in predicates:
            subtree = _normalize(" ".join(token.text for token in predicate.subtree))
            if subject and subject.split()[0] not in subtree:
                subtree = f"{subject} {subtree}"
            clauses.append(subtree)
        clauses.append(sent.text)
    return clauses


def _regex_clauses(text: str) -> list[str]:
    return [part for part in _CONJUNCTION_SPLIT.split(text) if part.strip()]


def build_query_plan(query: str, max_lanes: int = MAX_QUERY_LANES) -> list[str]:
    """Decompose a query into independent retrieval lanes (spaCy, regex fallback)."""
    normalized = _normalize(query)
    if not normalized:
        return []
    lanes: list[str] = [normalized]
    masked, identifiers = mask_identifiers(normalized)
    clauses = _spacy_clauses(masked) or _regex_clauses(normalized)
    for clause in clauses:
        candidate = _normalize(clause).strip(" ,.;:-")
        if candidate and candidate not in lanes and len(candidate) >= 4:
            lanes.append(candidate)
        if len(lanes) >= max_lanes:
            break
    for identifier in identifiers:
        if len(lanes) >= max_lanes:
            break
        if identifier not in lanes:
            lanes.append(identifier)
    aliases = expand_query_synonyms(normalized)
    if aliases and len(lanes) < max_lanes:
        synonym_lane = _normalize(" ".join(aliases))
        if synonym_lane and synonym_lane not in lanes:
            lanes.append(synonym_lane)
    return lanes[:max_lanes]
