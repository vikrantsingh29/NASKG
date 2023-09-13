import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("C:\\Users\\vikrant.singh\\Downloads\\processed_data.txt",
                 sep=" ", header=None)
data.columns = ['timestamp', 'icao24', 'latitude', 'longitude']

full_sequences = []

for _, group in data.groupby('icao24'):
    full_seq = group[['latitude', 'longitude']].values
    full_sequences.append(full_seq)

full_sequences = np.array(full_sequences)

# Splitting into training and testing sets
train_size = int(0.8 * len(full_sequences))
train_sequences = full_sequences[:train_size]
test_sequences = full_sequences[train_size:]

# Convert to PyTorch tensors
train_sequences_tensor = torch.tensor(train_sequences, dtype=torch.float32)
test_sequences_tensor = torch.tensor(test_sequences, dtype=torch.float32)


# Model definition
class Seq2SeqTrajectoryPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(Seq2SeqTrajectoryPredictor, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sequence_length = 20

    def forward(self, x, teacher_force_ratio=0.5):
        batch_size = x.size(0)
        seq_len = x.size(1)

        outputs = torch.zeros(batch_size, self.sequence_length, 2)

        # Initial input is the last point in the given sequence
        input = x[:, -1, :].unsqueeze(1)

        for t in range(self.sequence_length):  # This loop now runs for 20 iterations
            output, (hidden, cell) = self.lstm(input)
            output = self.fc(output.squeeze(1))
            outputs[:, t, :] = output

            # Decide if we will use teacher forcing or not
            teacher_force = np.random.random() < teacher_force_ratio

            # Get the next input for the LSTM
            input = output.unsqueeze(1) if not teacher_force or t >= seq_len else x[:, t, :].unsqueeze(1)


        return outputs


model = Seq2SeqTrajectoryPredictor(input_size=2, hidden_size=50, num_layers=2, output_size=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_function = nn.MSELoss()

# Training
num_epochs = 100
batch_size = 32
for epoch in range(num_epochs):
    for i in range(0, len(train_sequences), batch_size):
        inputs = train_sequences_tensor[i:i + batch_size]
        targets = train_sequences_tensor[i:i + batch_size]

        outputs = model(inputs)
        loss = loss_function(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Testing Loop
# For testing, we'll provide the first 12 points and predict the next 8.
predictions = []
model.eval()
with torch.no_grad():
    for i in range(len(train_sequences)):
        inputs = train_sequences_tensor[i, :12].unsqueeze(0)
        sequence_predictions = []

        # Using the model to predict the next 8 points
        for _ in range(8):
            output = model(inputs)
            sequence_predictions.append(output[0, -1, :].numpy())
            inputs = torch.cat((inputs[:, 1:, :], output[:, -1, :].unsqueeze(1)), dim=1)

        predictions.append(np.concatenate([train_sequences_tensor[i, :12].numpy(), sequence_predictions], axis=0))

predictions = np.array(predictions)
for i in range(10):  # Plotting for the first 10 trajectories
    print(predictions[i])
    # print(train_sequences[i, 12:, 1], train_sequences[i, 12:, 0])

# Adjusted Plotting
# As we are now predicting the continuation of the trajectory, we'll adjust the plotting code accordingly.
# for i in range(10):  # Plotting for the first 10 trajectories
#     plt.figure(figsize=(10, 5))
#
#     plt.subplot(1, 2, 1)
#     plt.plot(train_sequences[i, :12, 1], train_sequences[i, :12, 0], 'bo-', label='Given')
#     plt.plot(predictions[i, 12:, 1], predictions[i, 12:, 0], 'ro-', label='Predicted')
#     plt.plot(train_sequences[i, 11:, 1], train_sequences[i, 11:, 0], 'go-', label='Actual')
#     plt.xlabel('Longitude')
#     plt.ylabel('Latitude')
#     plt.legend()
#     plt.title('Latitude and Longitude')
#
#     plt.tight_layout()
#     plt.show()