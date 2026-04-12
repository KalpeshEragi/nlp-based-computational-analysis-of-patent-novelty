"""
Claim parsing module.
Uses spaCy to extract components (noun phrases), functions (verbs),
and relations (component → function) from each patent claim.
"""

import re
import spacy

# Load the English spaCy model once at module level
nlp = spacy.load("en_core_web_sm")


def is_dependent(claim_text: str) -> bool:
    """
    Detect whether a claim is dependent.
    A dependent claim references another claim number or 'preceding claim'.
    """
    pattern = r"(claim\s+\d+|preceding claim)"
    return bool(re.search(pattern, claim_text, re.IGNORECASE))


def extract_components_functions(claim_text: str) -> dict:
    """
    Use spaCy NLP to extract:
      - components: meaningful noun phrases (filtered for length and stop words)
      - functions:  root verbs and action verbs in the claim
      - relations:  simple (component, function) pairings based on subject-verb proximity
    """
    doc = nlp(claim_text)

    # --- Components: noun chunks with at least 2 tokens, skip generic/short phrases ---
    SKIP_LEMMAS = {"system", "method", "device", "apparatus", "claim", "use"}
    components = []
    for chunk in doc.noun_chunks:
        # Keep noun phrases that have 2+ tokens and aren't trivially generic
        if len(chunk) >= 2 and chunk.root.lemma_.lower() not in SKIP_LEMMAS:
            components.append(chunk.text.lower().strip())
    # Deduplicate while preserving order
    components = list(dict.fromkeys(components))

    # --- Functions: verbs that carry semantic weight (not auxiliary/copula) ---
    SKIP_POS = {"AUX"}
    SKIP_VERB_LEMMAS = {"be", "have", "do", "include", "comprise", "consist"}
    functions = []
    for token in doc:
        if (
            token.pos_ == "VERB"
            and token.dep_ not in {"aux", "auxpass"}
            and token.lemma_.lower() not in SKIP_VERB_LEMMAS
        ):
            functions.append(token.lemma_.lower())
    functions = list(dict.fromkeys(functions))

    # --- Relations: pair each component with the nearest following verb ---
    relations = []
    for chunk in doc.noun_chunks:
        comp_text = chunk.text.lower().strip()
        if comp_text not in components:
            continue
        # Walk right from the chunk head to find the first relevant verb
        for token in doc[chunk.end :]:
            if (
                token.pos_ == "VERB"
                and token.lemma_.lower() not in SKIP_VERB_LEMMAS
            ):
                relations.append((comp_text, token.lemma_.lower()))
                break  # one relation per component is enough for this prototype

    return {
        "components": components,
        "functions": functions,
        "relations": relations,
    }


def parse_claim(claim_text: str) -> dict:
    """
    Full claim parse result (without embedding — added later by similarity module).

    Returns:
        {
            "text": str,
            "dependent": bool,
            "components": List[str],
            "functions": List[str],
            "relations": List[tuple],
            "embedding": None   # placeholder; filled in similarity.py
        }
    """
    parsed = extract_components_functions(claim_text)
    return {
        "text": claim_text,
        "dependent": is_dependent(claim_text),
        "components": parsed["components"],
        "functions": parsed["functions"],
        "relations": parsed["relations"],
        "embedding": None,  # filled later
    }


def parse_all_claims(claims: list[str]) -> list[dict]:
    """Parse a list of raw claim strings into structured dicts."""
    return [parse_claim(c) for c in claims]