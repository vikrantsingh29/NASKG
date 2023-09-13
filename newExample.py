import torch
import torch.nn as nn
from torch.optim import SGD
# from nni.nas.pytorch import mutables
# from nni.algorithms.nas.pytorch.ppo import PPO
# from nni.nas.pytorch import apply_fixed_architecture
from torch.utils.data import DataLoader, Dataset
# import os
#
# # Define your mutable layer
# class MyNetwork(nn.Module):
#     def __init__(self):
#         super(MyNetwork, self).__init__()
#         self.mutable = mutables.LayerChoice([
#             nn.Linear(10, 10),  # Option 1
#             nn.Linear(10, 20),  # Option 2
#             nn.Linear(10, 30),  # Option 3
#         ])
#
#     def forward(self, x):
#         return self.mutable(x)
#
# # Define a linear scoring function
# class LinearScoringFunction(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.W = nn.Parameter(torch.randn(10, 10))
#         self.b = nn.Parameter(torch.randn(10))
#
#     def forward(self, h, r, t):
#         return self.W * torch.cat([h, r, t], dim=-1) + self.b
#
# # Define a polynomial scoring function
# class PolynomialScoringFunction(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.W1 = nn.Parameter(torch.randn(10, 10))
#         self.b1 = nn.Parameter(torch.randn(10))
#         self.W2 = nn.Parameter(torch.randn(10, 10))
#         self.b2 = nn.Parameter(torch.randn(10))
#
#     def forward(self, h, r, t):
#         hrt = torch.cat([h, r, t], dim=-1)
#         return self.W2 * (self.W1 * hrt + self.b1)**2 + self.b2
#
# # Define your MLP scoring function
# class MLScoringFunction(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.W1 = nn.Parameter(torch.randn(10, 10))
#         self.b1 = nn.Parameter(torch.randn(10))
#         self.W2 = nn.Parameter(torch.randn(10, 10))
#         self.b2 = nn.Parameter(torch.randn(10))
#
#     def forward(self, h, r, t):
#         return self.W2 * torch.relu(self.W1 * torch.cat([h, r, t], dim=-1) + self.b1) + self.b2
#
# # Your custom dataset
# class MyDataset(Dataset):
#     def __init__(self, filepath):
#         self.data = torch.load(filepath)  # Assuming the data is saved in PyTorch tensor format
#
#     def __getitem__(self, index):
#         return self.data[index]
#
#     def __len__(self):
#         return len(self.data)
#
# # Path to the data
# data_path = 'C:\\Users\\vikrant.singh\\PycharmProjects\\NASKG\\UMLS'
#
# # Load your data
# train_dataset = MyDataset(os.path.join(data_path, 'train.txt'))
# train_dataloader = DataLoader(train_dataset, batch_size=32)
#
# # Model
# model = MyNetwork()
# optimizer = SGD(model.parameters(), lr=0.1)
#
# # Define your scoring functions
# scoring_functions = [
#     LinearScoringFunction(),
#     PolynomialScoringFunction(),
#     MLScoringFunction()
# ]
#
# for scoring_function in scoring_functions:
#     criterion = nn.BCEWithLogitsLoss()  # Adjust based on your specific task
#
#     for epoch in range(100):  # Training loop
#         for batch in train_dataloader:
#             optimizer.zero_grad()
#             inputs, labels = batch
#             h = model(inputs)
#             outputs = scoring_function(h, h, h)  # Replace with your implementation
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#
#     # NAS training
#     algo = PPO(model, 10, 10)
#
#     for _ in range(500):  # Perform 500 epochs
#         # Do forward with gradients
#         logits = algo.record(model, loss_fn=model.criterion, optimizer=model.optimizer)
#         # Use logits to decide action
#         action = algo.choose_action(logits)
#
#     # Apply the architecture chosen by the NAS
#     apply_fixed_architecture(model, algo.export())

#
# import nni
# import torch
# from nni import algorithms
# from nni.algorithms.nas.enas import EnasTrainer
#
# class LinearScoringFunction(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.W = nn.Parameter(torch.randn(10, 10))
#         self.b = nn.Parameter(torch.randn(10))
#
#     def forward(self, h, r, t):
#         return self.W * torch.cat([h, r, t], dim=-1) + self.b
#
#
# def scoring_function(arch, dataset):
#     """A scoring function that computes the accuracy of a neural architecture on a knowledge graph embedding task.
#
#   Args:
#     arch: A neural architecture.
#     dataset: A knowledge graph embedding dataset.
#
#   Returns:
#     The accuracy of the neural architecture on the dataset.
#   """
#
#     model = LinearScoringFunction()
#     model.load_state_dict(nni.get_trial_parameters())
#     model.eval()
#     scores = model(dataset.h, dataset.r, dataset.t)
#     return scores.mean()
#
#
# def main():
#     """The main function that runs the neural architecture search.
#   """
#
#     search_space = {
#         "num_layers": [1, 2, 3],
#         "layer_type": ["MLP", "CNN"],
#         "activation": ["ReLU", "Sigmoid"],
#     }
#
#     tuner = nni.algorithms.nas.enas.ENASSearcher(search_space)
#     tuner.scoring_function = scoring_function
#
#     # Add the dataset path
#     dataset = torch.load("C:\\Users\\vikrant.singh\\PycharmProjects\\NASKG\\UMLS\\train.txt")
#
#     # Run the neural architecture search.
#     tuner.run()
#
#
# if __name__ == "__main__":
#     main()
