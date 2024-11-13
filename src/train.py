import torch
import torch.nn as nn
from src.make_graph import load_graph
from src.make_plots import plot_loss_acc
from models.gcn import FraudGCN

MODELS_DIR = "./models"


def train(model, data_pg, epochs=1000, lr=0.0001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    loss_arr, acc_arr = [], []

    # Training loop
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

    return model, loss_arr, acc_arr

def main():
    # Load the graph data
    print("Loading graph")
    data_pg = load_graph(graph_fn="fraud_network.pth")

    # Model parameters
    input_dim = data_pg.x.shape[1]
    hidden_dim = 128
    output_dim = 2

    # Initialize model and move to device
    print("Creating Model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gcn2 = FraudGCN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

    # Train the model
    print("Training Model")
    gcn2, loss, acc = train(gcn2.to(device), data_pg.to(device))

    # Save the model
    print("Saving Model")
    torch.save(gcn2.state_dict(), f"{MODELS_DIR}/gcn2.pth")

    # Plot the loss and accuracy
    print("Plotting Loss and Accuracy")
    plot_loss_acc(loss, acc)
    
if __name__ == "__main__":
    main()
