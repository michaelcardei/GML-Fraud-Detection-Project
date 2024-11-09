import pandas as pd
import networkx as nx

# Read in identity and transaction data
train_identity = pd.read_csv("RawData/train_identity.csv")
print(f"train_identity shape: {train_identity.shape}")

train_transaction = pd.read_csv("RawData/train_transaction.csv")
print(f"train_transaction shape: {train_transaction.shape}")

# Fill in missing values, drop empty columns
train_identity_numerical = train_identity.select_dtypes(include='number')
train_identity[train_identity_numerical.columns] = train_identity_numerical.fillna(train_identity_numerical.median())
train_identity = train_identity.apply(lambda x: x.fillna(x.value_counts().index[0]))

train_transaction_numerical = train_transaction.select_dtypes(include='number')
train_transaction[train_transaction_numerical.columns] = train_transaction_numerical.fillna(train_transaction_numerical.median())
cols_with_nan = train_transaction.isna().sum()
empty_cols = cols_with_nan[cols_with_nan == len(train_transaction)].index
train_transaction.drop(columns=empty_cols, inplace=True)

# Merge identity and transaction data on TransactionID
train_data = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')
train_data_squished = train_data.sample(frac=0.025)

# Create a multigraph with TransactionIDs as nodes and columns as edges (some of the columns had too much data and the graph was too large)
G_train = nx.MultiGraph()
# columns = ['P_emaildomain', 'R_emaildomain', 'card4', 'card6']
edge_features = ['P_emaildomain', 'R_emaildomain', ]
node_features = train_data_squished.drop(columns=edge_features + ['TransactionID']).columns

for _, row in train_data_squished.iterrows():
    transaction_id = row['TransactionID']
    node_attributes = {feature: row[feature] for feature in node_features}
    G_train.add_node(transaction_id, **node_attributes)

for edge in edge_features:
    col_groups = train_data_squished.groupby(edge)['TransactionID'].apply(list)
    for transactions in col_groups:
        for i in range(len(transactions)):
            for j in range(i + 1, len(transactions)):
                G_train.add_edge(transactions[i], transactions[j], key=edge)
                
nx.write_graphml(G_train, 'transaction_multigraph.graphml')