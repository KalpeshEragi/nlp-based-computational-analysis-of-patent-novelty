"""
Summary generation module.
Produces a Gemini-style natural language summary of the patent analysis.
The generate_summary() function accepts an optional Gemini API key.
If no key is provided, it produces a high-quality rule-based simulation.
"""


def _simulated_summary(claim_results: list[dict], aggregation: dict) -> str:
    """
    Rule-based simulation of a Gemini-style summary.
    Groups claims by verdict and constructs a structured analysis.
    """
    not_inventive = [r for r in claim_results if r["verdict"] == "NOT INVENTIVE"]
    weak = [r for r in claim_results if r["verdict"] == "WEAK INVENTIVE"]
    inventive = [r for r in claim_results if r["verdict"] == "INVENTIVE"]

    def claim_ids(results):
        return ", ".join(str(r["claim_id"]) for r in results) if results else "None"

    novel_components = set()
    for r in inventive + weak:
        for comp in r.get("new_components", []):
            novel_components.add(comp)

    lines = [
        "=" * 70,
        "PATENT INVENTIVE STEP ANALYSIS — SUMMARY REPORT",
        "=" * 70,
        "",
        f"Overall Verdict: {aggregation['overall_verdict']}",
        "",
        "--- Claim-Level Findings ---",
        "",
        f"NOT INVENTIVE claims  : {claim_ids(not_inventive)}",
        f"WEAK INVENTIVE claims : {claim_ids(weak)}",
        f"INVENTIVE claims      : {claim_ids(inventive)}",
        "",
        "--- Key Observations ---",
        "",
    ]

    if not_inventive:
        patents_cited = {r["best_match_patent"] for r in not_inventive}
        lines.append(
            f"Claims {claim_ids(not_inventive)} show substantial overlap with prior art "
            f"patents {', '.join(patents_cited)}. These claims may require amendment "
            f"to introduce distinguishing features."
        )

    if weak:
        lines.append(
            f"Claims {claim_ids(weak)} share structural components with prior art but "
            f"introduce modified or extended functional behaviour. These claims carry "
            f"moderate novelty risk and should be reviewed for claim differentiation."
        )

    if inventive:
        lines.append(
            f"Claims {claim_ids(inventive)} introduce genuinely novel elements not "
            f"anticipated by the reviewed prior art."
        )

    if novel_components:
        lines.append(
            f"\nKey novel technical elements identified: "
            f"{', '.join(list(novel_components)[:8])}."
        )

    lines += [
        "",
        "--- Recommendations ---",
        "",
        "1. Amend non-inventive independent claims to emphasise novel combinations.",
        "2. Add dependent claims that specifically claim the novel sensor fusion approach.",
        "3. Strengthen claims around hyperspectral analysis and adaptive weighting logic.",
        "4. Consider adding method claims that articulate the unique operational sequence.",
        "5. Document technical effect and unexpected advantage in the description.",
        "",
        "=" * 70,
    ]

    return "\n".join(lines)


def generate_summary(
    claim_results: list[dict],
    aggregation: dict,
    gemini_api_key: str = None,
) -> str:
    """
    Generate a natural language analysis summary.

    If a Gemini API key is provided, sends a structured prompt to the
    Gemini API and returns its response. Otherwise falls back to the
    high-quality rule-based simulation.

    Args:
        claim_results:   list of per-claim analysis dicts
        aggregation:     overall aggregation dict from inventive_engine
        gemini_api_key:  optional string; if None, simulation is used

    Returns:
        str: multi-line summary text
    """
    if gemini_api_key:
        try:
            import google.generativeai as genai

            genai.configure(api_key=gemini_api_key)
            model = genai.GenerativeModel("gemini-1.5-pro")

            # Build a structured prompt from the analysis results
            prompt_lines = [
                "You are a senior patent attorney and AI analyst.",
                "Analyse the following patent claim evaluation results and produce a professional summary.",
                "",
                f"Overall verdict: {aggregation['overall_verdict']}",
                f"Statistics: {aggregation['summary_line']}",
                "",
                "Claim-level results:",
            ]
            for r in claim_results:
                prompt_lines.append(
                    f"  Claim {r['claim_id']}: {r['verdict']} — {r['reason']}"
                )

            prompt_lines += [
                "",
                "Please provide:",
                "1. A clear statement of which claims are novel and which are not.",
                "2. The key areas of innovation.",
                "3. Practical recommendations to strengthen the patent.",
            ]

            response = model.generate_content("\n".join(prompt_lines))
            return response.text

        except Exception as exc:
            print(f"[WARNING] Gemini API call failed ({exc}). Falling back to simulation.")

    # Default: high-quality rule-based simulation
    return _simulated_summary(claim_results, aggregation)