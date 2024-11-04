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
columns = ['P_emaildomain', 'R_emaildomain', ]

for col in columns:
    col_groups = train_data_squished.groupby(col)['TransactionID'].apply(list)
    for transactions in col_groups:
        for i in range(len(transactions)):
            for j in range(i + 1, len(transactions)):
                G_train.add_edge(transactions[i], transactions[j], key=col)
                
nx.write_graphml(G_train, 'transaction_multigraph.graphml')