import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

DATA_DIR = "./ieee_fraud_detection"
GRAPH_DIR = "./graph"


def load_and_preprocess_data(data_name):
    # Load transaction and identity data, then merge them
    if data_name != "train" and data_name != "test":
        raise ValueError("Data name should be either 'train' or 'test'")

    transactions = pd.read_csv(f"{DATA_DIR}/{data_name}_transaction.csv")
    identity = pd.read_csv(f"{DATA_DIR}/{data_name}_identity.csv")
    data = transactions.merge(identity, on="TransactionID", how="left")

    # Fill missing values with -1
    data.fillna(-1, inplace=True)

    # Encode categorical features
    categorical_cols = [
        "ProductCD",
        "card4",
        "card6",
        "P_emaildomain",
        "R_emaildomain",
        "DeviceType",
        "DeviceInfo",
        "M1",
        "M2",
        "M3",
        "M4",
        "M5",
        "M6",
        "M7",
        "M8",
        "M9",
        "id_12",
        "id_15",
        "id_16",
        "id_23",
        "id_27",
        "id_28",
        "id_29",
        "id_30",
        "id_31",
        "id_33",
        "id_34",
        "id_35",
        "id_36",
        "id_37",
        "id_38",
    ]

    # Encode categorical features
    for col in categorical_cols:
        if col in data.columns:
            data[col] = data[col].astype(str)
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])

    # Separate fraud and non-fraud data
    fraud_data = data[data["isFraud"] == 1]
    non_fraud_data = data[data["isFraud"] == 0]

    # Balance the dataset
    fraud_sample = fraud_data
    sample_size = fraud_sample.shape[0]
    non_fraud_sample = non_fraud_data.sample(n=sample_size, random_state=42)
    data_balanced = pd.concat([fraud_sample, non_fraud_sample], axis=0).reset_index(
        drop=True
    )

    data_balanced = data_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    return data_balanced


def create_edges(data_balanced, edge_features, k=10):
    # Initialize edge list
    edge_index_list = []

    # Function to add edges based on shared feature values
    def add_edges_limited(data, feature, k):
        feature_groups = data.groupby(feature).groups
        for group_indices in feature_groups.values():
            if len(group_indices) == 0:
                continue
            for idx in group_indices:
                other_indices = [i for i in group_indices if i != idx]
                selected_indices = (
                    np.random.choice(other_indices, size=k, replace=False)
                    if len(other_indices) > k
                    else other_indices
                )
                for other_idx in selected_indices:
                    edge_index_list.append([idx, other_idx])

    # Add edges to edge list based on shared feature values
    for feature in edge_features:
        add_edges_limited(data_balanced, feature, k)

    # Convert list to tensor
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()

    return edge_index


def prepare_node_features_and_labels(data_balanced):
    # Get all feature names except "TransactionID" and "isFraud"
    feature_names = [
        col for col in data_balanced.columns if col not in ["TransactionID", "isFraud"]
    ]

    # Encode categorical features
    non_numeric_columns = (
        data_balanced[feature_names].select_dtypes(include=["object"]).columns.tolist()
    )
    for col in non_numeric_columns:
        le = LabelEncoder()
        data_balanced[col] = le.fit_transform(data_balanced[col].astype(str))

    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data_balanced[feature_names])

    node_features = torch.tensor(scaled_features, dtype=torch.float)
    node_labels = torch.tensor(data_balanced["isFraud"].values, dtype=torch.long)

    return node_features, node_labels


def split_data(data_pg, labels):
    # Create train and test masks instead of splitting
    train_mask = np.zeros(data_pg.num_nodes, dtype=bool)
    test_mask = np.zeros(data_pg.num_nodes, dtype=bool)

    # Stratified split to maintain proportion of fraudulent vs. non-fraudulent transactions
    train_indices, test_indices = train_test_split(
        np.arange(data_pg.num_nodes),
        test_size=0.2,
        stratify=labels.numpy(),
        random_state=42,
    )

    train_mask[train_indices] = True
    test_mask[test_indices] = True

    data_pg.train_mask = torch.tensor(train_mask)
    data_pg.test_mask = torch.tensor(test_mask)

    return data_pg


def create_graph(graph_fn=None, data_name="train"):
    data_balanced = load_and_preprocess_data(data_name=data_name)

    edge_features = [
        "card1",
        "card2",
        "card3",
        "card5",
        "addr1",
        "addr2",
        "P_emaildomain",
        "R_emaildomain",
        "DeviceInfo",
    ]

    edge_index = create_edges(data_balanced, edge_features, k=10)
    node_features, node_labels = prepare_node_features_and_labels(data_balanced)
    data_pg = Data(x=node_features, y=node_labels, edge_index=edge_index)
    data_pg = split_data(data_pg, node_labels)

    if graph_fn:
        print("Saving graph")
        os.makedirs(GRAPH_DIR, exist_ok=True)
        torch.save(data_pg, f"{GRAPH_DIR}/{graph_fn}")
        print("Graph saved")

    return data_pg


def load_graph(graph_fn):
    return torch.load(f"{GRAPH_DIR}/{graph_fn}")


def main():
    print("Making graph")
    create_graph(graph_fn="fraud_network.pth")


if __name__ == "__main__":
    main()
