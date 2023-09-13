import nni
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
import os


# Define Custom Scoring Function
class CustomScoringFunction(nn.Module):
    def __init__(self, embedding_dim, hidden_units):
        super(CustomScoringFunction, self).__init__()
        self.fc = nn.Linear(embedding_dim * 3, hidden_units)
        self.out = nn.Linear(hidden_units, 1)

    def forward(self, h, r, t):
        x = torch.cat([h, r, t], dim=-1)
        x = F.relu(self.fc(x))
        return self.out(x)


# Define a custom dataset (assuming the data is in a specific format)
class CustomDataset(Dataset):
    def __init__(self, file_path):
        self.data = open(file_path).readlines()
        self.label_encoder = LabelEncoder()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Parse the line, assuming a specific format
        # Adapt this according to the format of your data
        line = self.data[idx].strip().split()
        h, r, t = line
        return h, r, t

    def fit_label_encoder(self):
        # Fit the label encoder on the entire dataset
        all_data = [line.strip().split() for line in self.data]
        all_values = [value for triple in all_data for value in triple]
        self.label_encoder.fit(all_values)


# Load data
train_dataset = CustomDataset(os.path.join('C:\\Users\\vikrant.singh\\PycharmProjects\\NASKG\\UMLS', 'train.txt'))
val_dataset = CustomDataset(os.path.join('C:\\Users\\vikrant.singh\\PycharmProjects\\NASKG\\UMLS', 'valid.txt'))

# Fit the label encoder on the entire dataset
train_dataset.fit_label_encoder()
val_dataset.label_encoder = train_dataset.label_encoder  # Share the label encoder between train and validation datasets

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Get hyper-parameters from NNI
params = nni.get_next_parameter()

# Define model, loss, and optimizer
embedding_dim = 50
model = CustomScoringFunction(embedding_dim, params['hidden_units'])
criterion = nn.MSELoss()  # Define a suitable loss function
optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

# Training loop
num_epochs = 5  # Set the number of epochs
for epoch in range(num_epochs):
    for h, r, t in train_loader:
        # Transform the values using label encoder
        h_encoded = train_dataset.label_encoder.transform(h)
        r_encoded = train_dataset.label_encoder.transform(r)
        t_encoded = train_dataset.label_encoder.transform(t)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(torch.tensor(h_encoded), torch.tensor(r_encoded), torch.tensor(t_encoded))
        loss = criterion(outputs, torch.tensor(t_encoded))  # You might need to adapt this depending on your problem

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

# Evaluation loop on the validation set
model.eval()
total_val_loss = 0
with torch.no_grad():
    for h, r, t in val_loader:
        # Transform the values using label encoder
        h_encoded = val_dataset.label_encoder.transform(h)
        r_encoded = val_dataset.label_encoder.transform(r)
        t_encoded = val_dataset.label_encoder.transform(t)

        # Compute the scores for the candidate entities
        scores = model(torch.tensor(h_encoded), torch.tensor(r_encoded), torch.tensor(t_encoded))

        # Print the scores
        # Note: Depending on the size of your validation set, this could generate a lot of output
        print("Scores for candidate entities:", scores)

        # Compute the loss
        loss = criterion(scores, torch.tensor(t_encoded))
        total_val_loss += loss.item()

# Average validation loss
average_val_loss = total_val_loss / len(val_loader)
# Report final performance to NNI
nni.report_final_result(average_val_loss)




# s(h,r,t) = W2 * ReLU(W1 * [h;r;t] + b1) + b2
