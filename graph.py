"""
graph.py — Knowledge Graph Visualisation
Supports:
  - Per-claim graphs  → outputs/graphs/claim_X.png
  - Full invention graph → outputs/graphs/invention_full.png
  - Grid view of all claims → outputs/graphs/grid_view.png
"""

import os
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx

# ── colour palette ──────────────────────────────────────────────────────────
COMPONENT_COLOR = "#4A90D9"
FUNCTION_COLOR  = "#E07B39"
EDGE_COLOR      = "#AAAAAA"
BG_COLOR        = "#0F1117"
TEXT_COLOR      = "#FFFFFF"

GRAPH_DIR = os.path.join("outputs", "graphs")


def _ensure_dir():
    os.makedirs(GRAPH_DIR, exist_ok=True)


def _build_graph(parsed_claims: list[dict]) -> tuple[nx.DiGraph, set, set]:
    """
    Build a directed graph from a list of parsed claims.
    Returns (graph, component_nodes, function_nodes).
    """
    G = nx.DiGraph()
    component_nodes: set[str] = set()
    function_nodes:  set[str] = set()

    for claim in parsed_claims:
        for comp, func in claim.get("relations", []):
            comp_label = comp[:38]
            func_label = func[:28]
            G.add_node(comp_label, node_type="component")
            G.add_node(func_label, node_type="function")
            G.add_edge(comp_label, func_label)
            component_nodes.add(comp_label)
            function_nodes.add(func_label)

    return G, component_nodes, function_nodes


def _draw_graph(
    ax: plt.Axes,
    G: nx.DiGraph,
    component_nodes: set,
    function_nodes: set,
    title: str,
    font_size: int = 7,
):
    """Render a single graph onto the given Axes."""
    ax.set_facecolor(BG_COLOR)
    ax.set_title(title, color=TEXT_COLOR, fontsize=font_size + 1,
                 fontweight="bold", pad=8)
    ax.axis("off")

    if G.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "No relations extracted",
                ha="center", va="center", color="#888888",
                transform=ax.transAxes, fontsize=font_size)
        return

    pos = nx.spring_layout(G, seed=42, k=2.2)

    nx.draw_networkx_nodes(
        G, pos, nodelist=list(component_nodes),
        node_color=COMPONENT_COLOR, node_size=900, alpha=0.95, ax=ax,
    )
    nx.draw_networkx_nodes(
        G, pos, nodelist=list(function_nodes),
        node_color=FUNCTION_COLOR, node_size=600,
        node_shape="s", alpha=0.95, ax=ax,
    )
    nx.draw_networkx_edges(
        G, pos, edge_color=EDGE_COLOR,
        arrows=True, arrowsize=12, width=1.0, alpha=0.6, ax=ax,
    )
    nx.draw_networkx_labels(
        G, pos, font_size=font_size - 1,
        font_color=TEXT_COLOR, ax=ax,
    )


# ── Public API ───────────────────────────────────────────────────────────────

def build_knowledge_graph(
    parsed_claims: list[dict],
    title: str = "Invention Patent — Knowledge Graph",
    output_path: str = "knowledge_graph.png",
) -> str:
    """
    Build and save the FULL invention knowledge graph.
    Also writes to outputs/graphs/invention_full.png.
    Returns the saved file path.
    """
    _ensure_dir()
    G, comp_nodes, func_nodes = _build_graph(parsed_claims)

    fig, ax = plt.subplots(figsize=(18, 11), facecolor=BG_COLOR)
    _draw_graph(ax, G, comp_nodes, func_nodes, title, font_size=8)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COMPONENT_COLOR, label="Component"),
        Patch(facecolor=FUNCTION_COLOR,  label="Function"),
    ]
    ax.legend(handles=legend_elements, loc="upper left",
              fontsize=9, facecolor="#1E2130", labelcolor=TEXT_COLOR,
              framealpha=0.8)

    full_path = os.path.join(GRAPH_DIR, "invention_full.png")
    plt.tight_layout()
    plt.savefig(full_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    # Also write to the user-specified path for backward compatibility
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"[graph] Full knowledge graph saved → {full_path}")
    return full_path


def build_per_claim_graphs(parsed_claims: list[dict]) -> list[str]:
    """
    Build and save one knowledge graph per invention claim.
    Writes to outputs/graphs/claim_1.png, claim_2.png, …
    Returns a list of file paths (one per claim, in order).
    """
    _ensure_dir()
    paths: list[str] = []

    for idx, claim in enumerate(parsed_claims, start=1):
        G, comp_nodes, func_nodes = _build_graph([claim])

        fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG_COLOR)
        short_text = claim["text"][:60] + ("…" if len(claim["text"]) > 60 else "")
        _draw_graph(ax, G, comp_nodes, func_nodes,
                    title=f"Claim {idx}: {short_text}", font_size=7)

        path = os.path.join(GRAPH_DIR, f"claim_{idx}.png")
        plt.tight_layout()
        plt.savefig(path, dpi=130, bbox_inches="tight", facecolor=BG_COLOR)
        plt.close()
        paths.append(path)

    print(f"[graph] Per-claim graphs saved → {GRAPH_DIR}/claim_*.png")
    return paths


def build_grid_view(parsed_claims: list[dict], cols: int = 4) -> str:
    """
    Assemble all per-claim graphs into one grid PNG.
    Writes to outputs/graphs/grid_view.png.
    Returns the file path.
    """
    _ensure_dir()
    n = len(parsed_claims)
    rows = math.ceil(n / cols)

    fig = plt.figure(figsize=(cols * 5, rows * 4), facecolor=BG_COLOR)
    gs  = gridspec.GridSpec(rows, cols, figure=fig,
                            hspace=0.45, wspace=0.35)

    for idx, claim in enumerate(parsed_claims):
        ax = fig.add_subplot(gs[idx // cols, idx % cols])
        G, comp_nodes, func_nodes = _build_graph([claim])
        _draw_graph(ax, G, comp_nodes, func_nodes,
                    title=f"Claim {idx + 1}", font_size=6)

    # Blank out any remaining grid cells
    for idx in range(n, rows * cols):
        ax = fig.add_subplot(gs[idx // cols, idx % cols])
        ax.set_facecolor(BG_COLOR)
        ax.axis("off")

    grid_path = os.path.join(GRAPH_DIR, "grid_view.png")
    plt.savefig(grid_path, dpi=120, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"[graph] Grid view saved → {grid_path}")
    return grid_path