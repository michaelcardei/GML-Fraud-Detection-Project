import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import networkx as nx
from torch_geometric.utils import to_networkx
from sklearn.metrics import classification_report

PLOTS_DIR = "./plots"

def __check_dir():
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)

def plot_loss_acc(loss_arr, acc_arr, plot_fn="train_loss_plot"):
    plt.figure(figsize=(7, 5))
    plt.tight_layout()
    plt.plot(loss_arr, color="tab:blue", label="Train Loss", linewidth=2)
    plt.plot(acc_arr, color="tab:orange", label="Test Accuracy", linewidth=2)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Value", fontsize=14)
    plt.title("Loss and Accuracy vs. Epochs", fontsize=16, weight="bold")
    plt.grid(visible=True, linestyle="--", alpha=0.7)
    plt.legend(loc="upper right", fontsize=12)
    __check_dir()
    plt.savefig(f"{PLOTS_DIR}/{plot_fn}.png", dpi=300)
    plt.show()


def plot_classification_report(data_pg, pred, plot_fn="classification_report"):
    y_true = data_pg.y[data_pg.test_mask].cpu().numpy()
    y_pred = pred.cpu().numpy()
    report = classification_report(y_true, y_pred, digits=4, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    plt.tight_layout()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("tight")
    ax.axis("off")
    table = ax.table(
        cellText=report_df.values,
        colLabels=report_df.columns,
        rowLabels=report_df.index,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)
    __check_dir()
    plt.savefig(f"{PLOTS_DIR}/{plot_fn}.png", dpi=300)
    plt.show()
    
def plot_graph(data_pg, start=1500, end=1900):
    G = to_networkx(data_pg, to_undirected=True, node_attrs=["x", "y"])
    sub_G = G.subgraph(range(start, end))
    
    connected_components = list(nx.connected_components(sub_G))
    largest_cc_nodes = list(max(connected_components, key=len))

    sub_G = sub_G.subgraph(largest_cc_nodes).copy()

    plt.figure(figsize=(6, 6))
    plt.gca().set_facecolor("lightgray")
    plt.title(f"Subgraph with {nx.number_of_nodes(sub_G)} nodes and {nx.number_of_edges(sub_G)} edges", fontsize=16)
    pos = nx.spring_layout(sub_G)
    node_colors = [
        "blue" if sub_G.nodes[node]["y"] == 0 else "red" for node in sub_G.nodes
    ]
    nx.draw(
        sub_G,
        pos,
        with_labels=False,
        node_color=node_colors,
        node_size=100,
        font_size=10,
        font_color="black",
        edge_color="gray",
    )
    plt.tight_layout()

    # Add custom legend
    legend_handles = [
        plt.Line2D([0], [0], color="blue", marker="o", markersize=10, label="Legitimate", linestyle=""),
        plt.Line2D([0], [0], color="red", marker="o", markersize=10, label="Fraudulent", linestyle=""),
    ]
    plt.legend(handles=legend_handles, loc="upper right", fontsize=12)
    plt.savefig(f"{PLOTS_DIR}/{"graph_sample"}.png", dpi=300)
    plt.show()
    
