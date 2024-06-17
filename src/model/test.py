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

def test(X_test, y_test, model, scaler):
    model.eval()
    X_test_scaled = scaler.transform(X_test)
    test_dataset = TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    predictions = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            outputs = model(X_batch).squeeze()
            predictions.extend(outputs.numpy())
    
    return predictions

def main():
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    test_file = os.getenv('TEST_FILE', config['data']['test_file'])
    model_dir = os.getenv('MODEL_DIR', config['model']['save_dir'])
    
    X_test, y_test = load_data(test_file)
    scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
    
    model = SimpleNN(X_test.shape[1])
    model.load_state_dict(torch.load(os.path.join(model_dir, 'model_weights.pth')))
    
    predictions = test(X_test, y_test, model, scaler)
    
    results = pd.DataFrame({'True': y_test, 'Predicted': predictions})
    results.to_csv(os.path.join(model_dir, 'test_results.csv'), index=False)
    print(results.head())

if __name__ == "__main__":
    main()