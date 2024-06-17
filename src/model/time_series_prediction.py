import os
import pandas as pd
import torch
import joblib
from torch.utils.data import DataLoader, TensorDataset
import yaml

def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.values
    return X

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

def predict_time_series(model, scaler, X_future):
    model.eval()
    X_future_scaled = scaler.transform(X_future)
    future_dataset = TensorDataset(torch.tensor(X_future_scaled, dtype=torch.float32))
    future_loader = DataLoader(future_dataset, batch_size=32, shuffle=False)
    
    predictions = []
    with torch.no_grad():
        for X_batch in future_loader:
            outputs = model(X_batch[0]).squeeze()
            predictions.extend(outputs.numpy())
    
    return predictions

def main():
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    future_file = os.getenv('FUTURE_FILE', config['data']['future_file'])
    model_dir = os.getenv('MODEL_DIR', config['model']['save_dir'])
    
    X_future = load_data(future_file)
    scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
    
    model = SimpleNN(X_future.shape[1])
    model.load_state_dict(torch.load(os.path.join(model_dir, 'model_weights.pth')))
    
    predictions = predict_time_series(model, scaler, X_future)
    predictions_df = pd.DataFrame(predictions, columns=['Predictions'])
    predictions_df.to_csv(os.path.join(model_dir, 'time_series_predictions.csv'), index=False)
    print(predictions_df.head())

if __name__ == "__main__":
    main()