"""
app.py — Patent Inventive Step Analyzer  (Streamlit UI)
Run:  streamlit run app.py
"""

import os
import importlib
import streamlit as st

from claim_parser import parse_all_claims
from similarity import compute_embeddings, get_top_k_similar
from comparator import compare_claims
from inventive_engine import decide_inventiveness, aggregate_results
from summary import generate_summary
from graph import build_knowledge_graph, build_per_claim_graphs, build_grid_view

# ── page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="Patent Inventive Step Analyzer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

TOP_K = 5

# ── custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    background-color: #0A0C12;
    color: #E2E8F0;
}

/* ── Headings ── */
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0F1117;
    border-right: 1px solid #1E2130;
}
section[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #4A90D9 0%, #357ABD 100%);
    color: white;
    border: none;
    padding: 0.7rem 1rem;
    border-radius: 8px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.95rem;
    letter-spacing: 0.03em;
    transition: all 0.2s;
    cursor: pointer;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(74, 144, 217, 0.4);
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #13161F;
    border: 1px solid #1E2130;
    border-radius: 12px;
    padding: 1.2rem 1.5rem !important;
    transition: border-color 0.2s;
}
[data-testid="metric-container"]:hover {
    border-color: #4A90D9;
}
[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 2rem !important;
    color: #4A90D9 !important;
}
[data-testid="stMetricLabel"] {
    color: #8892A4 !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* ── Expanders ── */
details {
    background: #13161F !important;
    border: 1px solid #1E2130 !important;
    border-radius: 10px !important;
    margin-bottom: 0.6rem !important;
    transition: border-color 0.2s;
}
details:hover { border-color: #2A3050 !important; }
details summary {
    padding: 0.8rem 1.2rem !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.85rem !important;
    color: #CBD5E1 !important;
    cursor: pointer;
}
details[open] { border-color: #4A90D9 !important; }

/* ── Code / mono spans ── */
code {
    font-family: 'DM Mono', monospace !important;
    background: #1E2130 !important;
    color: #E07B39 !important;
    padding: 0.15em 0.45em !important;
    border-radius: 4px !important;
    font-size: 0.82rem !important;
}

/* ── Progress bar ── */
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #4A90D9, #E07B39) !important;
    border-radius: 99px;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #13161F;
    border-radius: 10px;
    gap: 4px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #8892A4;
    border-radius: 8px;
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 0.85rem;
    padding: 0.4rem 1rem;
    transition: all 0.15s;
}
.stTabs [aria-selected="true"] {
    background: #1E2130 !important;
    color: #4A90D9 !important;
}

/* ── Verdict badges (via custom html) ── */
.badge {
    display: inline-block;
    padding: 0.25em 0.8em;
    border-radius: 99px;
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.04em;
}
.badge-inventive   { background: #0D3B2B; color: #34D399; border: 1px solid #34D399; }
.badge-weak        { background: #3B2D06; color: #FBBF24; border: 1px solid #FBBF24; }
.badge-not         { background: #3B0D0D; color: #F87171; border: 1px solid #F87171; }

/* ── Info boxes ── */
.info-box {
    background: #13161F;
    border-left: 3px solid #4A90D9;
    border-radius: 0 8px 8px 0;
    padding: 0.9rem 1.2rem;
    margin: 0.5rem 0;
    font-size: 0.88rem;
    color: #CBD5E1;
    line-height: 1.6;
}
.warn-box {
    background: #1F1A0D;
    border-left: 3px solid #FBBF24;
    border-radius: 0 8px 8px 0;
    padding: 0.9rem 1.2rem;
    margin: 0.5rem 0;
    font-size: 0.88rem;
    color: #FDE68A;
}
.error-box {
    background: #1F0D0D;
    border-left: 3px solid #F87171;
    border-radius: 0 8px 8px 0;
    padding: 0.9rem 1.2rem;
    margin: 0.5rem 0;
    font-size: 0.88rem;
    color: #FCA5A5;
}

/* ── Divider ── */
hr { border-color: #1E2130 !important; }

/* ── Tag pills ── */
.tag {
    display: inline-block;
    background: #1E2130;
    color: #8892A4;
    border-radius: 99px;
    padding: 0.15em 0.6em;
    font-size: 0.75rem;
    margin: 2px;
    font-family: 'DM Mono', monospace;
}
.tag-new {
    background: #0D2B1F;
    color: #6EE7B7;
    border: 1px solid #34D399;
}
.tag-overlap {
    background: #2B1F0D;
    color: #FCD34D;
    border: 1px solid #FBBF24;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ─────────────────────────────────────────────────────────────────

def verdict_badge(verdict: str) -> str:
    if verdict == "INVENTIVE":
        return '<span class="badge badge-inventive">● INVENTIVE</span>'
    elif verdict == "WEAK INVENTIVE":
        return '<span class="badge badge-weak">◑ WEAK INVENTIVE</span>'
    else:
        return '<span class="badge badge-not">✕ NOT INVENTIVE</span>'


def verdict_icon(verdict: str) -> str:
    return {"INVENTIVE": "🟢", "WEAK INVENTIVE": "🟡", "NOT INVENTIVE": "🔴"}.get(verdict, "⚪")


def tag_pills(items: list[str], style: str = "") -> str:
    if not items:
        return '<span class="tag">—</span>'
    return "".join(f'<span class="tag {style}">{t}</span>' for t in items[:8])


def sim_color(sim: float) -> str:
    if sim >= 0.85:
        return "#F87171"
    elif sim >= 0.65:
        return "#FBBF24"
    return "#34D399"


def sim_bar_html(sim: float) -> str:
    color = sim_color(sim)
    pct = int(sim * 100)
    return f"""
    <div style="display:flex;align-items:center;gap:8px;margin:4px 0;">
        <div style="flex:1;height:6px;background:#1E2130;border-radius:99px;overflow:hidden;">
            <div style="width:{pct}%;height:100%;background:{color};border-radius:99px;"></div>
        </div>
        <span style="font-family:'DM Mono',monospace;font-size:0.78rem;color:{color};min-width:38px;">{sim:.3f}</span>
    </div>"""


# ── Available datasets ───────────────────────────────────────────────────────
AVAILABLE_DATASETS = ["mock_patents", "mock_patent1", "mock_patent2", "mock_patent3"]


# ── Pipeline ─────────────────────────────────────────────────────────────────

def run_pipeline(dataset_name: str, api_key: str | None):
    # ── Load dataset ──────────────────────────────────────────────────────
    try:
        data_module = importlib.import_module(f"data.{dataset_name}")
        prior_art_data = data_module.prior_art_data
        invention_data = data_module.invention_data
    except ModuleNotFoundError:
        st.markdown(
            f'<div class="error-box">❌ Dataset module <code>data/{dataset_name}.py</code> not found. '
            f'Make sure the file exists inside the <code>data/</code> folder.</div>',
            unsafe_allow_html=True,
        )
        return
    except AttributeError as exc:
        st.markdown(
            f'<div class="error-box">❌ Dataset loaded but missing required variable: {exc}</div>',
            unsafe_allow_html=True,
        )
        return

    # ── Progress UI ───────────────────────────────────────────────────────
    progress_bar = st.progress(0)
    status = st.empty()

    # Step 1 — Parse
    status.markdown("⚙️ **[1/6]** Parsing prior art and invention claims…")
    all_prior_parsed: list[dict] = []
    for patent in prior_art_data:
        parsed = parse_all_claims(patent["claims"])
        for c in parsed:
            c["patent_id"] = patent["patent_id"]
        all_prior_parsed.extend(parsed)
    invention_parsed = parse_all_claims(invention_data["claims"])
    progress_bar.progress(15)

    # Step 2 — Embeddings
    status.markdown(
        f"🧠 **[2/6]** Computing embeddings for "
        f"{len(all_prior_parsed)} prior art + {len(invention_parsed)} invention claims…"
    )
    compute_embeddings(all_prior_parsed)
    compute_embeddings(invention_parsed)
    progress_bar.progress(35)

    # Step 3 — Graphs
    status.markdown("🕸️ **[3/6]** Building knowledge graphs…")
    build_knowledge_graph(
        invention_parsed,
        title=f"Invention — {invention_data['title']}",
        output_path="knowledge_graph.png",
    )
    per_claim_paths = build_per_claim_graphs(invention_parsed)
    grid_path = build_grid_view(invention_parsed, cols=4)
    progress_bar.progress(50)

    # Step 4 — Analysis
    status.markdown("🔍 **[4/6]** Running per-claim inventive step analysis…")
    claim_results: list[dict] = []

    for idx, inv_claim in enumerate(invention_parsed):
        claim_id = f"C{idx + 1}"
        top_matches = get_top_k_similar(inv_claim, all_prior_parsed, k=TOP_K)
        comparisons = [compare_claims(inv_claim, m["parsed"]) for m in top_matches]
        decision = decide_inventiveness(inv_claim, top_matches, comparisons)

        best_comp = comparisons[0] if comparisons else {}

        claim_results.append({
            "claim_id":           claim_id,
            "claim_idx":          idx + 1,
            "claim_text":         inv_claim["text"],
            "dependent":          inv_claim["dependent"],
            "components":         inv_claim["components"],
            "functions":          inv_claim["functions"],
            "top_matches":        top_matches,
            "components_overlap": best_comp.get("components_overlap", []),
            "new_components":     best_comp.get("new_components", []),
            "functions_overlap":  best_comp.get("functions_overlap", []),
            "new_functions":      best_comp.get("new_functions", []),
            "relation_similarity":best_comp.get("relation_similarity", 0.0),
            "verdict":            decision["verdict"],
            "reason":             decision["reason"],
            "best_match_patent":  decision["best_match_patent"],
            "best_match_text":    decision["best_match_text"],
            "best_similarity":    decision["best_similarity"],
            "graph_path":         per_claim_paths[idx] if idx < len(per_claim_paths) else None,
        })

    progress_bar.progress(75)

    # Step 5 — Aggregate
    status.markdown("📊 **[5/6]** Aggregating verdicts…")
    aggregation = aggregate_results(claim_results)
    progress_bar.progress(87)

    # Step 6 — Summary
    status.markdown("✍️ **[6/6]** Generating analysis summary…")
    summary_text = generate_summary(claim_results, aggregation, api_key)
    progress_bar.progress(100)
    status.empty()
    progress_bar.empty()

    # ── Render results ────────────────────────────────────────────────────
    _render_results(invention_data, prior_art_data, claim_results,
                    aggregation, summary_text, grid_path)


# ── Results rendering ────────────────────────────────────────────────────────

def _render_results(
    invention_data, prior_art_data,
    claim_results, aggregation, summary_text, grid_path,
):
    ov = aggregation["overall_verdict"]
    ov_color = {"INVENTIVE": "#34D399", "PARTIAL NOVELTY": "#FBBF24",
                "NOT INVENTIVE": "#F87171"}.get(ov, "#8892A4")

    # ── Hero banner ───────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="
        background:linear-gradient(135deg,#13161F 0%,#1E2130 100%);
        border:1px solid #1E2130;
        border-radius:16px;
        padding:2rem 2.5rem;
        margin-bottom:1.5rem;
    ">
        <p style="font-family:'DM Mono',monospace;font-size:0.75rem;
                  color:#8892A4;letter-spacing:0.12em;margin-bottom:0.3rem;">
            ANALYSIS COMPLETE
        </p>
        <h2 style="font-family:'Syne',sans-serif;font-size:1.6rem;
                   color:#E2E8F0;margin:0 0 0.5rem 0;">
            {invention_data['title']}
        </h2>
        <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
            <span style="font-family:'DM Mono',monospace;font-size:1rem;
                         color:{ov_color};font-weight:700;
                         background:rgba(255,255,255,0.04);
                         border:1px solid {ov_color}44;
                         border-radius:8px;padding:0.3em 0.9em;">
                {ov}
            </span>
            <span style="color:#8892A4;font-size:0.85rem;">
                {len(prior_art_data)} prior art patents · {aggregation['total_claims']} invention claims analysed
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Metrics row ───────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Claims",    aggregation["total_claims"])
    m2.metric("✅ Inventive",    aggregation["inventive_count"])
    m3.metric("⚠️ Weak",        aggregation["weak_inventive_count"])
    m4.metric("❌ Not Inventive",aggregation["not_inventive_count"])
    m5.metric("Prior Art Patents", len(prior_art_data))

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────
    tab_overview, tab_claims, tab_graph, tab_summary = st.tabs([
        "📋  Overview", "🔍  Claim Analysis", "🕸️  Knowledge Graphs", "📝  Summary"
    ])

    # ── TAB 1: Overview ───────────────────────────────────────────────────
    with tab_overview:
        st.markdown("#### Verdict Distribution")

        # Simple ASCII-style bar chart using columns
        col_chart, col_legend = st.columns([3, 2])
        with col_chart:
            total = aggregation["total_claims"] or 1
            for label, count, color in [
                ("INVENTIVE",     aggregation["inventive_count"],      "#34D399"),
                ("WEAK INVENTIVE",aggregation["weak_inventive_count"], "#FBBF24"),
                ("NOT INVENTIVE", aggregation["not_inventive_count"],  "#F87171"),
            ]:
                pct = int((count / total) * 100)
                st.markdown(f"""
                <div style="margin:10px 0;">
                    <div style="display:flex;justify-content:space-between;
                                font-family:'DM Mono',monospace;font-size:0.78rem;
                                color:#8892A4;margin-bottom:4px;">
                        <span>{label}</span>
                        <span>{count} claims ({pct}%)</span>
                    </div>
                    <div style="height:10px;background:#1E2130;border-radius:99px;overflow:hidden;">
                        <div style="width:{pct}%;height:100%;background:{color};
                                    border-radius:99px;transition:width 1s;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with col_legend:
            st.markdown("#### Prior Art Patents")
            for patent in prior_art_data:  # type: ignore[name-defined]
                st.markdown(f"""
                <div style="background:#13161F;border:1px solid #1E2130;
                             border-radius:8px;padding:0.6rem 0.9rem;margin:4px 0;
                             font-size:0.83rem;">
                    <code>{patent['patent_id']}</code>
                    <span style="color:#8892A4;margin-left:8px;">
                        {patent.get('title','—')}
                    </span>
                </div>
                """, unsafe_allow_html=True)

        # Claim verdict quick-reference table
        st.markdown("#### Claim Verdict Table")
        table_html = """<div style="overflow-x:auto;">
<table style="width:100%;border-collapse:collapse;font-size:0.82rem;font-family:'DM Mono',monospace;">
    <thead>
        <tr style="background:#1E2130;color:#8892A4;text-align:left;">
            <th style="padding:8px 12px;">Claim</th>
            <th style="padding:8px 12px;">Type</th>
            <th style="padding:8px 12px;">Verdict</th>
            <th style="padding:8px 12px;">Best Match</th>
            <th style="padding:8px 12px;">Similarity</th>
        </tr>
    </thead>
    <tbody>"""
        
        for res in claim_results:
            dep_tag = "Dependent" if res["dependent"] else "Independent"
            dep_color = "#4A90D9" if not res["dependent"] else "#8892A4"
            v = res["verdict"]
            v_color = {"INVENTIVE":"#34D399","WEAK INVENTIVE":"#FBBF24","NOT INVENTIVE":"#F87171"}.get(v,"#8892A4")
            sim = res["best_similarity"]
            sc = sim_color(sim)
            table_html += f"""
        <tr style="border-bottom:1px solid #1E2130;">
            <td style="padding:7px 12px;color:#CBD5E1;font-weight:600;">{res['claim_id']}</td>
            <td style="padding:7px 12px;color:{dep_color};">{dep_tag}</td>
            <td style="padding:7px 12px;color:{v_color};font-weight:600;">{v}</td>
            <td style="padding:7px 12px;color:#8892A4;">{res['best_match_patent'] or '—'}</td>
            <td style="padding:7px 12px;color:{sc};">{sim:.3f}</td>
        </tr>"""
            
        table_html += "</tbody></table></div>"
        st.markdown(table_html, unsafe_allow_html=True)

    # ── TAB 2: Claim Analysis ─────────────────────────────────────────────
    with tab_claims:
        # Filter controls
        col_f1, col_f2, _ = st.columns([1, 1, 2])
        with col_f1:
            filter_verdict = st.selectbox(
                "Filter by verdict",
                ["All", "INVENTIVE", "WEAK INVENTIVE", "NOT INVENTIVE"],
            )
        with col_f2:
            filter_type = st.selectbox(
                "Filter by claim type",
                ["All", "Independent", "Dependent"],
            )

        filtered = claim_results
        if filter_verdict != "All":
            filtered = [r for r in filtered if r["verdict"] == filter_verdict]
        if filter_type != "All":
            want_dep = filter_type == "Dependent"
            filtered = [r for r in filtered if r["dependent"] == want_dep]

        st.markdown(f"<p style='color:#8892A4;font-size:0.82rem;'>"
                    f"Showing {len(filtered)} of {len(claim_results)} claims</p>",
                    unsafe_allow_html=True)

        for res in filtered:
            v = res["verdict"]
            icon = verdict_icon(v)
            badge = verdict_badge(v)
            dep_label = "Dependent" if res["dependent"] else "Independent"

            with st.expander(
                f"{icon}  {res['claim_id']}  ·  {dep_label}  ·  "
                f"Best match: {res['best_match_patent'] or '—'}  ·  "
                f"sim={res['best_similarity']:.3f}",
                expanded=False,
            ):
                # Claim header
                st.markdown(
                    f"{badge} &nbsp; "
                    f"<span style='font-family:DM Mono,monospace;font-size:0.75rem;"
                    f"color:#8892A4;'>{dep_label} Claim</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="info-box">{res["claim_text"]}</div>',
                    unsafe_allow_html=True,
                )

                left, right = st.columns([1, 1])

                with left:
                    # Knowledge graph
                    st.markdown("**Knowledge Graph**")
                    gpath = res.get("graph_path")
                    if gpath and os.path.exists(gpath):
                        st.image(gpath, use_column_width=True)
                    else:
                        st.markdown(
                            '<div class="warn-box">Graph not available for this claim.</div>',
                            unsafe_allow_html=True,
                        )

                    # Extracted elements
                    st.markdown("**Extracted Components**")
                    st.markdown(tag_pills(res["components"]), unsafe_allow_html=True)
                    st.markdown("**Extracted Functions**")
                    st.markdown(tag_pills(res["functions"]), unsafe_allow_html=True)

                with right:
                    # Inventiveness reasoning
                    st.markdown("**Inventive Step Reasoning**")
                    reason_color = {
                        "INVENTIVE": "info-box",
                        "WEAK INVENTIVE": "warn-box",
                        "NOT INVENTIVE": "error-box",
                    }.get(v, "info-box")
                    st.markdown(
                        f'<div class="{reason_color}">{res["reason"]}</div>',
                        unsafe_allow_html=True,
                    )

                    # Overlap analysis
                    st.markdown("**Component Overlap** *(shared with best prior art)*")
                    st.markdown(
                        tag_pills(res["components_overlap"], "tag-overlap"),
                        unsafe_allow_html=True,
                    )
                    st.markdown("**New Components** *(not in prior art)*")
                    st.markdown(
                        tag_pills(res["new_components"], "tag-new"),
                        unsafe_allow_html=True,
                    )
                    st.markdown("**New Functions** *(not in prior art)*")
                    st.markdown(
                        tag_pills(res["new_functions"], "tag-new"),
                        unsafe_allow_html=True,
                    )

                    # Relation similarity
                    rel_sim = res.get("relation_similarity", 0.0)
                    st.markdown("**Relation Similarity to Best Match**")
                    st.markdown(sim_bar_html(rel_sim), unsafe_allow_html=True)

                    # Top-K matches
                    st.markdown("**Top-K Prior Art Matches**")
                    for match in res["top_matches"]:
                        sc = sim_color(match["similarity"])
                        short_text = match["claim_text"][:110] + (
                            "…" if len(match["claim_text"]) > 110 else ""
                        )
                        st.markdown(f"""
                        <div style="background:#0F1117;border:1px solid #1E2130;
                                    border-radius:8px;padding:0.7rem 1rem;margin:5px 0;">
                            <div style="display:flex;justify-content:space-between;
                                        align-items:center;margin-bottom:4px;">
                                <code style="color:#4A90D9;">{match['patent_id']}</code>
                                <span style="font-family:'DM Mono',monospace;
                                             font-size:0.78rem;color:{sc};">
                                    {match['similarity']:.4f}
                                </span>
                            </div>
                            {sim_bar_html(match['similarity'])}
                            <p style="color:#8892A4;font-size:0.78rem;
                                      margin:6px 0 0 0;font-style:italic;">
                                {short_text}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                    # Best match claim text
                    if res.get("best_match_text"):
                        st.markdown("**Closest Prior Art Claim**")
                        st.markdown(
                            f'<div class="warn-box" style="font-style:italic;">'
                            f'{res["best_match_text"]}</div>',
                            unsafe_allow_html=True,
                        )

    # ── TAB 3: Knowledge Graphs ───────────────────────────────────────────
    with tab_graph:
        st.markdown("#### Invention Knowledge Graph (Full)")
        full_graph = os.path.join("outputs", "graphs", "invention_full.png")
        if os.path.exists(full_graph):
            st.image(full_graph, use_column_width=True)
        else:
            st.markdown(
                '<div class="warn-box">Full graph image not found. Re-run analysis.</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("#### Per-Claim Graph Grid View")
        if os.path.exists(grid_path):
            with st.expander("Show / hide grid (all claims)", expanded=True):
                st.image(grid_path, use_column_width=True)
        else:
            st.markdown(
                '<div class="warn-box">Grid view not found.</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("#### Individual Claim Graphs")
        gcols = st.columns(2)
        for i, res in enumerate(claim_results):
            gpath = res.get("graph_path")
            with gcols[i % 2]:
                if gpath and os.path.exists(gpath):
                    st.image(gpath, caption=f"{res['claim_id']} — {res['verdict']}",
                             use_column_width=True)

    # ── TAB 4: Summary ────────────────────────────────────────────────────
    with tab_summary:
        st.markdown("#### Overall Verdict")
        st.markdown(
            f'<div class="info-box">{aggregation["summary_line"]}</div>',
            unsafe_allow_html=True,
        )

        st.markdown("#### Patent Analysis Report")
        api_label = "Gemini API" if GEMINI_API_KEY else "Simulated (rule-based)"
        st.markdown(
            f'<p style="font-family:DM Mono,monospace;font-size:0.75rem;'
            f'color:#8892A4;margin-bottom:0.8rem;">Generated by: {api_label}</p>',
            unsafe_allow_html=True,
        )
        # Render summary in a styled block
        st.markdown(
            f'<div style="background:#13161F;border:1px solid #1E2130;'
            f'border-radius:12px;padding:1.5rem 2rem;'
            f'font-family:Plus Jakarta Sans,sans-serif;font-size:0.88rem;'
            f'line-height:1.8;color:#CBD5E1;white-space:pre-wrap;">'
            f'{summary_text}'
            f'</div>',
            unsafe_allow_html=True,
        )


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar() -> tuple[str, str | None]:
    with st.sidebar:
        st.markdown("""
        <div style="padding:1rem 0 0.5rem;">
            <p style="font-family:'Syne',sans-serif;font-size:1.3rem;
                      font-weight:800;color:#E2E8F0;margin:0;">⚖️ Patent</p>
            <p style="font-family:'Syne',sans-serif;font-size:1.3rem;
                      font-weight:800;color:#4A90D9;margin:0;">Analyzer</p>
            <p style="font-family:'DM Mono',monospace;font-size:0.7rem;
                      color:#8892A4;margin-top:4px;letter-spacing:0.1em;">
                INVENTIVE STEP PIPELINE v2
            </p>
        </div>
        <hr style="border-color:#1E2130;margin:0.5rem 0 1rem;">
        """, unsafe_allow_html=True)

        st.markdown("**Dataset**")
        dataset_choice = st.selectbox(
            "Select patent dataset",
            AVAILABLE_DATASETS,
            label_visibility="collapsed",
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Pipeline Settings**")
        top_k = st.slider("Top-K similar prior art claims", 1, 10, TOP_K)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**API**")
        api_status = "✅ Loaded" if GEMINI_API_KEY else "❌ Not set (using simulation)"
        st.markdown(
            f'<div class="info-box" style="font-size:0.78rem;">'
            f'Gemini API Key: {api_status}</div>',
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        run = st.button("▶ Run Analysis", type="primary")

        st.markdown("<hr style='border-color:#1E2130;'>", unsafe_allow_html=True)
        st.markdown("""
        <p style="font-family:'DM Mono',monospace;font-size:0.7rem;color:#4A5568;">
        Pipeline:<br>
        1 · Parse claims (spaCy)<br>
        2 · Embed (MiniLM-L6-v2)<br>
        3 · Semantic filter (cosine)<br>
        4 · Structured comparison<br>
        5 · Inventive step decision<br>
        6 · Aggregate + summarise
        </p>
        """, unsafe_allow_html=True)

        return dataset_choice, run, top_k


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    dataset_choice, run, top_k = render_sidebar()

    # Landing state
    if not run:
        st.markdown("""
        <div style="
            min-height: 60vh;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
            padding: 4rem 2rem;
        ">
            <p style="font-family:'DM Mono',monospace;font-size:0.78rem;
                      color:#8892A4;letter-spacing:0.15em;margin-bottom:1rem;">
                RESEARCH PROTOTYPE
            </p>
            <h1 style="font-family:'Syne',sans-serif;font-size:3rem;
                       color:#E2E8F0;margin:0 0 1rem 0;line-height:1.1;">
                Patent Inventive<br>Step Analyzer
            </h1>
            <p style="color:#8892A4;max-width:500px;line-height:1.7;font-size:0.92rem;">
                Semantic similarity filtering · Structured NLP reasoning ·
                Rule-based inventive step decisions · Knowledge graph visualisation
            </p>
            <div style="margin-top:2rem;display:flex;gap:12px;flex-wrap:wrap;justify-content:center;">
                <span style="background:#13161F;border:1px solid #1E2130;border-radius:99px;
                             padding:0.4em 1em;font-family:'DM Mono',monospace;font-size:0.78rem;color:#4A90D9;">
                    spaCy NLP
                </span>
                <span style="background:#13161F;border:1px solid #1E2130;border-radius:99px;
                             padding:0.4em 1em;font-family:'DM Mono',monospace;font-size:0.78rem;color:#4A90D9;">
                    sentence-transformers
                </span>
                <span style="background:#13161F;border:1px solid #1E2130;border-radius:99px;
                             padding:0.4em 1em;font-family:'DM Mono',monospace;font-size:0.78rem;color:#4A90D9;">
                    cosine similarity
                </span>
                <span style="background:#13161F;border:1px solid #1E2130;border-radius:99px;
                             padding:0.4em 1em;font-family:'DM Mono',monospace;font-size:0.78rem;color:#4A90D9;">
                    networkx graphs
                </span>
            </div>
            <p style="color:#4A5568;font-size:0.8rem;margin-top:3rem;font-family:'DM Mono',monospace;">
                ← Select a dataset and click <strong style="color:#4A90D9;">Run Analysis</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
        return

    # Override global TOP_K with slider value
    global TOP_K
    TOP_K = top_k

    st.markdown(
        f'<div class="info-box">🚀 Running pipeline on dataset: '
        f'<code>{dataset_choice}</code></div>',
        unsafe_allow_html=True,
    )

    run_pipeline(dataset_choice, GEMINI_API_KEY)


if __name__ == "__main__":
    main()