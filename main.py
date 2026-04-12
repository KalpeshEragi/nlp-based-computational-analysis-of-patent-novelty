"""
main.py — Patent Inventive Step Analyzer
Pipeline runner. Execute this file to run the full analysis.

Usage:
    python main.py
    python main.py --gemini-key YOUR_API_KEY   (optional, for real Gemini output)
"""

import argparse
import pprint

from data.mock_patents import prior_art_data, invention_data
from claim_parser import parse_all_claims
from similarity import compute_embeddings, get_top_k_similar
from comparator import compare_claims
from inventive_engine import decide_inventiveness, aggregate_results
from summary import generate_summary
from graph import build_knowledge_graph

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
TOP_K = 5          # number of prior art claims to compare per invention claim
GRAPH_OUTPUT = "knowledge_graph.png"


def main(gemini_api_key: str = None):

    print("\n" + "=" * 60)
    print("  Patent Inventive Step Analyzer — Starting Pipeline")
    print("=" * 60 + "\n")

    # -----------------------------------------------------------------------
    # STEP 1: Parse all prior art claims
    # -----------------------------------------------------------------------
    print("[1/7] Parsing prior art claims...")
    all_prior_parsed = []
    for patent in prior_art_data:
        parsed_claims = parse_all_claims(patent["claims"])
        for claim in parsed_claims:
            claim["patent_id"] = patent["patent_id"]   # attach source patent id
        all_prior_parsed.extend(parsed_claims)
    print(f"      → {len(all_prior_parsed)} prior art claims parsed.\n")

    # -----------------------------------------------------------------------
    # STEP 2: Parse all invention claims
    # -----------------------------------------------------------------------
    print("[2/7] Parsing invention claims...")
    invention_parsed = parse_all_claims(invention_data["claims"])
    print(f"      → {len(invention_parsed)} invention claims parsed.\n")

    # -----------------------------------------------------------------------
    # STEP 3: Compute embeddings for ALL claims
    # -----------------------------------------------------------------------
    print("[3/7] Computing sentence embeddings (this may take a moment)...")
    compute_embeddings(all_prior_parsed)
    compute_embeddings(invention_parsed)
    print("      → Embeddings computed.\n")

    # -----------------------------------------------------------------------
    # STEP 4–6: Per-claim similarity filter, structured comparison, decision
    # -----------------------------------------------------------------------
    print("[4/7] Running per-claim analysis...\n")
    claim_results = []

    for idx, inv_claim in enumerate(invention_parsed):
        claim_id = f"C{idx + 1}"

        # 4a. Semantic similarity filter → top-k prior art candidates
        top_matches = get_top_k_similar(inv_claim, all_prior_parsed, k=TOP_K)

        # 4b. Structured comparison against each top-k candidate
        comparisons = [
            compare_claims(inv_claim, match["parsed"])
            for match in top_matches
        ]

        # 4c. Inventive step decision
        decision = decide_inventiveness(inv_claim, top_matches, comparisons)

        # Build the full result record for this claim
        result = {
            "claim_id": claim_id,
            "claim_text": inv_claim["text"],
            "components": inv_claim["components"],
            "functions": inv_claim["functions"],
            "top_matches": [
                {
                    "patent_id": m["patent_id"],
                    "similarity": round(m["similarity"], 4),
                    "text": m["claim_text"][:120] + "...",
                }
                for m in top_matches
            ],
            "components_overlap": comparisons[0]["components_overlap"] if comparisons else [],
            "new_components": comparisons[0]["new_components"] if comparisons else [],
            "functions_overlap": comparisons[0]["functions_overlap"] if comparisons else [],
            "new_functions": comparisons[0]["new_functions"] if comparisons else [],
            "relation_similarity": comparisons[0]["relation_similarity"] if comparisons else 0.0,
            "verdict": decision["verdict"],
            "reason": decision["reason"],
            "best_match_patent": decision["best_match_patent"],
            "best_similarity": decision["best_similarity"],
        }

        claim_results.append(result)

        # Pretty-print per-claim result to terminal
        print(f"  {claim_id}: [{result['verdict']}]")
        print(f"       Best match: {result['best_match_patent']} "
              f"(sim={result['best_similarity']:.3f})")
        print(f"       Reason: {result['reason']}")
        print()

    # -----------------------------------------------------------------------
    # STEP 7: Aggregate results
    # -----------------------------------------------------------------------
    print("[5/7] Aggregating results...")
    aggregation = aggregate_results(claim_results)
    print(f"      → {aggregation['summary_line']}\n")

    # -----------------------------------------------------------------------
    # STEP 8: Generate summary
    # -----------------------------------------------------------------------
    print("[6/7] Generating summary...")
    summary_text = generate_summary(claim_results, aggregation, gemini_api_key)
    print("\n" + summary_text + "\n")

    # -----------------------------------------------------------------------
    # STEP 9: Build knowledge graph
    # -----------------------------------------------------------------------
    print("[7/7] Building knowledge graph...")
    build_knowledge_graph(
        invention_parsed,
        title=f"Invention Patent — {invention_data['title']}",
        output_path=GRAPH_OUTPUT,
    )

    print("\nPipeline complete.")
    print(f"Knowledge graph saved to: {GRAPH_OUTPUT}\n")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patent Inventive Step Analyzer")
    parser.add_argument(
        "--gemini-key",
        type=str,
        default=None,
        help="Optional Gemini API key. If omitted, simulated summary is used.",
    )
    args = parser.parse_args()
    main(gemini_api_key=args.gemini_key)