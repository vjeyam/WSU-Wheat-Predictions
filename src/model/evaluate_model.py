import os
import pandas as pd
import torch
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
import yaml

def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.drop('target', axis=1).values
    y = data['target'].values
    return X, y

class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def evaluate(X_val, y_val, model, scaler):
    model.eval()
    X_val_scaled = scaler.transform(X_val)
    val_dataset = TensorDataset(torch.tensor(X_val_scaled, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    predictions = []
    with torch.no_grad():
        for X_batch, _ in val_loader:
            outputs = model(X_batch).squeeze()
            predictions.extend(outputs.numpy())
    
    mae = mean_absolute_error(y_val, predictions)
    mse = mean_squared_error(y_val, predictions)
    
    return {'MAE': mae, 'MSE': mse}

def main():
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    val_file = os.getenv('VAL_FILE', config['data']['val_file'])
    model_dir = os.getenv('MODEL_DIR', config['model']['save_dir'])
    
    X_val, y_val = load_data(val_file)
    scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
    
    model = SimpleNN(X_val.shape[1])
    model.load_state_dict(torch.load(os.path.join(model_dir, 'model_weights.pth')))
    
    metrics = evaluate(X_val, y_val, model, scaler)
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(model_dir, 'evaluation_metrics.csv'), index=False)
    print(metrics)

if __name__ == "__main__":
    main()