import torch
import torch.nn as nn
from make_graph import create_graph
from models.gcn import FraudGCN

# Dataset path
data_folder = "./ieee_fraud_detection"

def main():
    # Load the graph data
    print("Making graph")
    data_pg = create_graph(data_folder)

    # Model parameters
    input_dim = data_pg.x.shape[1]
    hidden_dim = 128
    output_dim = 2

    # Initialize model and move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Creating Model")
    model = FraudGCN(
        input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim
    ).to(device)
    data_pg = data_pg.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    loss_arr, acc_arr = [], []

    # Training loop
    epochs = 1000
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data_pg)
        loss = criterion(out[data_pg.train_mask], data_pg.y[data_pg.train_mask])
        loss.backward()
        optimizer.step()

        # Evaluation
        model.eval()
        with torch.no_grad():
            out = model(data_pg)
            _, pred = out[data_pg.test_mask].max(dim=1)
            correct = int((pred == data_pg.y[data_pg.test_mask]).sum())
            
            curr_acc = round(correct / int(data_pg.test_mask.sum()), 4)
            curr_loss = round(loss.item(), 4)
            loss_arr.append(curr_loss)
            acc_arr.append(curr_acc)

            if (epoch + 1) % 50 == 0:
                print(f"Epoch: {epoch+1}, Loss: {curr_loss}, Test Acc: {curr_acc}")
                pred_counts = torch.bincount(pred)
                print(f"Predicted label counts: {pred_counts}")

if __name__ == "__main__":
    main()
