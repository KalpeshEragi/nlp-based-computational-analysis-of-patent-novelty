"""
Structured comparison module.
Compares components, functions, and relations between an invention claim
and a single prior art candidate claim.
"""


def _overlap(list_a: list, list_b: list) -> list:
    """Return elements common to both lists (case-insensitive)."""
    set_b = {x.lower() for x in list_b}
    return [x for x in list_a if x.lower() in set_b]


def _new_elements(list_a: list, list_b: list) -> list:
    """Return elements in list_a that are NOT in list_b."""
    set_b = {x.lower() for x in list_b}
    return [x for x in list_a if x.lower() not in set_b]


def _relation_similarity(rel_a: list[tuple], rel_b: list[tuple]) -> float:
    """
    Simple Jaccard-style similarity between two sets of (component, function) pairs.
    Returns a float in [0.0, 1.0].
    """
    set_a = {(c.lower(), f.lower()) for c, f in rel_a}
    set_b = {(c.lower(), f.lower()) for c, f in rel_b}
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


def compare_claims(invention_claim: dict, prior_claim: dict) -> dict:
    """
    Perform structured comparison between one invention claim and one prior art claim.

    Returns:
        {
            "components_overlap": List[str],
            "new_components": List[str],
            "functions_overlap": List[str],
            "new_functions": List[str],
            "relation_similarity": float,
        }
    """
    inv_components = invention_claim["components"]
    pri_components = prior_claim["components"]

    inv_functions = invention_claim["functions"]
    pri_functions = prior_claim["functions"]

    inv_relations = invention_claim["relations"]
    pri_relations = prior_claim["relations"]

    return {
        "components_overlap": _overlap(inv_components, pri_components),
        "new_components": _new_elements(inv_components, pri_components),
        "functions_overlap": _overlap(inv_functions, pri_functions),
        "new_functions": _new_elements(inv_functions, pri_functions),
        "relation_similarity": _relation_similarity(inv_relations, pri_relations),
    }