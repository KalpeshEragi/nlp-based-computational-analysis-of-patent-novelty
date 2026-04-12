"""
Inventive step decision engine.
Applies rule-based logic to structured comparison results
to determine whether each invention claim is novel.
"""


# Thresholds (tunable for the prototype)
COMPONENT_OVERLAP_RATIO_THRESHOLD = 0.6   # >60% components in common → "same"
FUNCTION_OVERLAP_RATIO_THRESHOLD = 0.6    # >60% functions in common → "same"
HIGH_SIMILARITY_THRESHOLD = 0.85          # cosine similarity for strong match


def _overlap_ratio(overlap: list, reference: list) -> float:
    """Fraction of reference elements that appear in the overlap."""
    if not reference:
        return 0.0
    return len(overlap) / len(reference)


def decide_inventiveness(
    invention_claim: dict,
    top_matches: list[dict],
    comparison_results: list[dict],
) -> dict:
    """
    Evaluate inventive step for a single invention claim against its top-k matches.

    Logic (applied to the BEST matching prior art claim):
        - IF components same AND functions same → NOT INVENTIVE
        - ELIF components same BUT functions differ → WEAK INVENTIVE
        - ELIF new components present → INVENTIVE

    Args:
        invention_claim:    parsed invention claim dict
        top_matches:        list of top-k similar prior art claim dicts (from similarity.py)
        comparison_results: list of comparison dicts (from comparator.py), same order

    Returns:
        {
            "verdict": str,       # "NOT INVENTIVE" | "WEAK INVENTIVE" | "INVENTIVE"
            "reason": str,
            "best_match_patent": str,
            "best_match_text": str,
            "best_similarity": float,
        }
    """
    if not top_matches:
        return {
            "verdict": "INVENTIVE",
            "reason": "No similar prior art found.",
            "best_match_patent": None,
            "best_match_text": None,
            "best_similarity": 0.0,
        }

    # Use the closest (highest cosine similarity) prior art claim as the benchmark
    best_idx = 0  # top_matches is already sorted descending by similarity
    best_match = top_matches[best_idx]
    comparison = comparison_results[best_idx]

    inv_components = invention_claim["components"]
    inv_functions = invention_claim["functions"]

    comp_overlap_ratio = _overlap_ratio(
        comparison["components_overlap"], inv_components
    )
    func_overlap_ratio = _overlap_ratio(
        comparison["functions_overlap"], inv_functions
    )
    has_new_components = len(comparison["new_components"]) > 0
    high_sim = best_match["similarity"] >= HIGH_SIMILARITY_THRESHOLD

    # --- Decision tree ---
    if (
        comp_overlap_ratio >= COMPONENT_OVERLAP_RATIO_THRESHOLD
        and func_overlap_ratio >= FUNCTION_OVERLAP_RATIO_THRESHOLD
        and high_sim
    ):
        verdict = "NOT INVENTIVE"
        reason = (
            f"High semantic similarity ({best_match['similarity']:.2f}) with {best_match['patent_id']}. "
            f"{int(comp_overlap_ratio*100)}% component overlap and "
            f"{int(func_overlap_ratio*100)}% function overlap detected."
        )

    elif (
        comp_overlap_ratio >= COMPONENT_OVERLAP_RATIO_THRESHOLD
        and func_overlap_ratio < FUNCTION_OVERLAP_RATIO_THRESHOLD
    ):
        verdict = "WEAK INVENTIVE"
        reason = (
            f"Similar components to {best_match['patent_id']} "
            f"(overlap: {int(comp_overlap_ratio*100)}%) "
            f"but introduces new functions: {comparison['new_functions']}."
        )

    elif has_new_components:
        verdict = "INVENTIVE"
        reason = (
            f"Introduces novel components not found in prior art: "
            f"{comparison['new_components']}."
        )

    else:
        # Fallback: moderate similarity but no strong overlap pattern
        verdict = "WEAK INVENTIVE"
        reason = (
            f"Moderate similarity ({best_match['similarity']:.2f}) with prior art. "
            f"Some new functions detected: {comparison['new_functions']}."
        )

    return {
        "verdict": verdict,
        "reason": reason,
        "best_match_patent": best_match["patent_id"],
        "best_match_text": best_match["claim_text"],
        "best_similarity": best_match["similarity"],
    }


def aggregate_results(claim_results: list[dict]) -> dict:
    """
    Aggregate claim-level verdicts into an overall patent-level decision.

    Rules:
        - Any "NOT INVENTIVE" claim → flag overall as at risk
        - Mix of verdicts → PARTIAL NOVELTY
        - All "INVENTIVE" → INVENTIVE

    Returns:
        {
            "total_claims": int,
            "not_inventive_count": int,
            "weak_inventive_count": int,
            "inventive_count": int,
            "overall_verdict": str,
            "summary_line": str,
        }
    """
    counts = {"NOT INVENTIVE": 0, "WEAK INVENTIVE": 0, "INVENTIVE": 0}
    for result in claim_results:
        v = result["verdict"]
        if v in counts:
            counts[v] += 1

    total = len(claim_results)
    not_inv = counts["NOT INVENTIVE"]
    weak = counts["WEAK INVENTIVE"]
    inventive = counts["INVENTIVE"]

    if not_inv > 0 and inventive == 0:
        overall = "NOT INVENTIVE"
    elif inventive == total:
        overall = "INVENTIVE"
    else:
        overall = "PARTIAL NOVELTY"

    summary_line = (
        f"Out of {total} claims: "
        f"{inventive} inventive, {weak} weakly inventive, {not_inv} not inventive. "
        f"Overall verdict: {overall}."
    )

    return {
        "total_claims": total,
        "not_inventive_count": not_inv,
        "weak_inventive_count": weak,
        "inventive_count": inventive,
        "overall_verdict": overall,
        "summary_line": summary_line,
    }