import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def print_graph_explanation_report(graph_idx, y_true, y_pred, confidences, 
                                   top_edges_all_graphs, top_nodes_all_graphs,
                                   label_map={0: "Control", 1: "ASD"}, top_k=10):
    """
    Prints a report for one graph including prediction info and most influential edges/nodes.
 
    """
    true_label = y_true[graph_idx]
    predicted_label = y_pred[graph_idx]
    confidence = confidences[graph_idx]

    print("="*60)
    print(f"Graph {graph_idx}")
    print(f"Predicted: {label_map[predicted_label]} (Confidence: {confidence:.2f})")
    print(f"Ground Truth: {label_map[true_label]}")
    print("-"*60)

    print(f"Top {top_k} Influential Edges (based on ΔE):")
    for edge, delta in top_edges_all_graphs[graph_idx][:top_k]:
        print(f"  • Edge {edge} -> ΔE = {delta:.4f}")

    print(f"\nTop {top_k} Influential Nodes (sum of ΔE over incident edges):")
    for node, score in top_nodes_all_graphs[graph_idx][:top_k]:
        print(f"  • Node {node} -> Score = {score:.4f}")

    print("="*60 + "\n")


def visualize_graph_explanation(edge_list, node_scores, top_k_nodes=20, graph_idx=0):
    G = nx.Graph()

    # Add edges with weights (abs(ΔE))
    for (u, v), delta in edge_list:
        G.add_edge(u, v, weight=abs(delta))

    pos = nx.spring_layout(G, seed=42)

    # Normalize node importance scores
    nodes = list(G.nodes())
    scores = np.array([node_scores.get(n, 0.0) for n in nodes])
    max_score = scores.max() if scores.size > 0 else 1.0
    norm_scores = scores / max_score

    # Normalize edge weights
    raw_weights = np.array([G[u][v]['weight'] for u, v in G.edges()])
    if len(raw_weights) > 0:
        norm_edge_weights = 1 + 4 * (raw_weights - raw_weights.min()) / (raw_weights.ptp() + 1e-6)
    else:
        norm_edge_weights = []

    # Draw edges (after normalization) with alpha
    nx.draw_networkx_edges(
        G, pos,
        width=norm_edge_weights,
        edge_color='gray',
        alpha=0.5
    )

    # Draw nodes
    node_draw = nx.draw_networkx_nodes(
        G, pos, node_size=300,
        node_color=norm_scores, cmap=plt.cm.cool
    )

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=8)

    # Add title and colorbar
    plt.title(f"Explanation for Graph {graph_idx}")
    cbar = plt.colorbar(node_draw)
    cbar.set_label("Node Importance")

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"figures/graph_explanation_{graph_idx}.png")
    plt.close()


def plot_influential_rois_for_graph(graph_idx,
                                    top_nodes_all_graphs,
                                    roi_labels,
                                    top_k=10,
                                    title_prefix="Influential ROIs for Graph"):
    """
    Visualizes the top influential ROIs for a specific graph (subject) with horizontal bars.

    Args:
        graph_idx (int): Index of the graph (subject) to visualize.
        top_nodes_all_graphs (list): List of lists. Each element is [(node_idx, score), ...] for that graph.
        roi_labels (list): List of ROI names (length = 111).
        top_k (int): Number of top influential ROIs to display.
    """

    top_nodes = sorted(top_nodes_all_graphs[graph_idx], key=lambda x: x[1], reverse=True)[:top_k]
    node_indices = [n for n, _ in top_nodes]
    node_scores = [s for _, s in top_nodes]
    roi_names = [roi_labels[n] for n in node_indices]

    colors = plt.cm.viridis(np.linspace(0.3, 1.0, len(node_indices)))

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(roi_names, node_scores, color=colors, alpha=0.9)

    ax.set_xlabel("Influence Score", fontsize=12)
    ax.set_ylabel("ROI", fontsize=12)
    ax.set_title(f"{title_prefix} {graph_idx}", fontsize=14, pad=15)
    ax.invert_yaxis()  # So the highest score is at the top
    ax.grid(axis='x', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(f"figures/influential_rois_{graph_idx}.png")