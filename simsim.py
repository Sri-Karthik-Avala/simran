import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

# Function to create dataset for time series prediction
def create_dataset(dataset, lookback):
    X, y = [], []
    for i in range(len(dataset)-lookback):
        feature = dataset[i:i+lookback]
        target = dataset[i+1:i+lookback+1]
        X.append(feature)
        y.append(target)
    X = np.array(X).reshape(-1, lookback, 1)  # Reshape to 3D (samples, lookback, features)
    y = np.array(y).reshape(-1, lookback, 1)  # Reshape to 3D (samples, lookback, features)
    return torch.tensor(X).float(), torch.tensor(y).float()

# Define LSTM model
class AirModel(nn.Module):
    def __init__(self):
        super(AirModel, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 1)
    
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

# App title
st.title("Time Series Prediction with LSTM")

# File uploader for CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# Model parameters
lookback = st.sidebar.slider("Lookback", min_value=10, max_value=365, value=10)
epochs = st.sidebar.slider("Epochs", min_value=100, max_value=5000, step=100, value=1000)
batch_size = st.sidebar.slider("Batch size", min_value=4, max_value=32, value=8)

if uploaded_file is not None:
    # Load CSV
    df = pd.read_csv(uploaded_file)
    st.write("Data preview:", df.head())
    
    # Extract time series data
    timeseries = df[["Amount Net Sales"]].values.astype('float32')

    # Train-test split
    train_size = int(len(timeseries) * 0.67)
    test_size = len(timeseries) - train_size
    train, test = timeseries[:train_size], timeseries[train_size:]
    
    # Create datasets
    X_train, y_train = create_dataset(train, lookback=lookback)
    X_test, y_test = create_dataset(test, lookback=lookback)
    
    # Display the shapes of the datasets
    st.write(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # DataLoader for batching
    train_loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=batch_size)

    # Initialize model, optimizer, and loss function
    model = AirModel()
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()

    # Training loop
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Validation step every 100 epochs
        if epoch % 100 == 0:
            model.eval()
            with torch.no_grad():
                y_pred_train = model(X_train)
                train_rmse = torch.sqrt(loss_fn(y_pred_train, y_train))
                y_pred_test = model(X_test)
                test_rmse = torch.sqrt(loss_fn(y_pred_test, y_test))
            st.write(f"Epoch {epoch}: Train RMSE {train_rmse.item():.4f}, Test RMSE {test_rmse.item():.4f}")
    
    # Prediction button
    if st.button("Predict"):
        with torch.no_grad():
            train_plot = np.ones_like(timeseries) * np.nan
            train_plot[lookback:train_size] = model(X_train)[:, -1, :].numpy()

            test_plot = np.ones_like(timeseries) * np.nan
            test_plot[train_size+lookback:len(timeseries)] = model(X_test)[:, -1, :].numpy()

        # Plot results
        fig, ax = plt.subplots()
        ax.plot(timeseries, label='Original data')
        ax.plot(train_plot, label='Train prediction', color='red')
        ax.plot(test_plot, label='Test prediction', color='green')
        ax.legend()

        st.pyplot(fig)
