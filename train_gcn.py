import torch
import torch.nn as nn
from torch_geometric.data import Data
from sklearn.metrics import classification_report
from models.gcn import FraudGNN
from make_graph import create_graph  

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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Creating Model")
    model = FraudGNN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(device)
    data_pg = data_pg.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

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
            acc = correct / int(data_pg.test_mask.sum())
            
            print(f'Epoch: {epoch+1}, Loss: {loss.item():.4f}, Test Acc: {acc:.4f}')
            
            if (epoch + 1) % 50 == 0:
                pred_counts = torch.bincount(pred)
                print(f'Predicted label counts: {pred_counts}')

    y_true = data_pg.y[data_pg.test_mask].cpu().numpy()
    y_pred = pred.cpu().numpy()
    print(classification_report(y_true, y_pred, digits=4))

if __name__ == "__main__":
    main()
