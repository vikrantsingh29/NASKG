import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class EmbeddingGenerator(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(EmbeddingGenerator, self).__init__()
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

    def forward(self, heads, relations):
        return self.entity_embeddings(heads), self.relation_embeddings(relations)


class ScoringFunction(nn.Module):
    def __init__(self, embedding_dim):
        super(ScoringFunction, self).__init__()
        self.W1 = nn.Linear(embedding_dim * 3, embedding_dim)
        self.W2 = nn.Linear(embedding_dim, 1)

    def forward(self, h, r, t):
        x = torch.cat([h, r, t], dim=1)
        x = torch.relu(self.W1(x))
        return self.W2(x)


# linear scoring function s(h,r,t) = W[h;r;t] + b, where W is a weight matrix and b is a bias term.
# Define a linear scoring function
class LinearScoringFunction(nn.Module):
    def __init__(self, embedding_dim):
        super(LinearScoringFunction, self).__init__()
        self.linear = nn.Linear(embedding_dim * 3, 1)

    def forward(self, h, r, t):
        x = torch.cat([h, r, t], dim=1)
        return self.linear(x)


# polynomial scoring function of degree 2: s(h, r, t) = W1 * [h;r;t]^2 + W2 * [h;r;t] + b Here, [h;r;t] represents
# the concatenation of the embeddings for the head, relation, and tail, W1 and W2 are weight matrices, and b is a
# bias term.

class PolynomialScoringFunction(nn.Module):
    def __init__(self, input_dim):
        super(PolynomialScoringFunction, self).__init__()
        self.linear1 = nn.Linear(input_dim, 1)
        self.linear2 = nn.Linear(input_dim, 1)

    def forward(self, h, r, t):
        x = torch.cat([h, r, t], dim=1)
        x_squared = x ** 2
        return self.linear1(x_squared) + self.linear2(x)


# Dummy data
# entity_to_id = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7, "I": 8, "J": 9}
# relation_to_id = {"r1": 0, "r2": 1, "r3": 2}
# train_data = [(0, 0, 1), (1, 1, 2), (2, 2, 3), (3, 0, 4), (4, 1, 5), (5, 2, 6), (6, 0, 7), (7, 1, 8), (8, 2, 9),
#               (9, 0, 0)]
# train_labels = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# test_data = [(0, 0, 2), (1, 1, 3), (2, 2, 4), (3, 0, 5)]
# test_labels = [0, 0, 0, 0]

# def read_data(file_path):
#     with open(file_path, 'r') as file:
#         data = []
#         labels = []
#         for line in file:
#             head, relation, tail, label = line.strip().split()  # Assuming the values are separated by whitespace
#             data.append((head, relation, tail))
#             labels.append(int(label))
#         return data, labels

def read_data(file_path):
    with open(file_path, 'r') as file:
        data = []
        for line in file:
            head, relation, tail = line.strip().split()
            data.append((head, relation, tail))
        return data


# def generate_negative_samples(data, entities, ratio=1):
#     negative_samples = []
#     for head, relation, tail in data:
#         for _ in range(ratio):
#             if np.random.random() < 0.5:
#                 negative_head = np.random.choice(entities)
#                 negative_samples.append((negative_head, relation, tail))
#             else:
#                 negative_tail = np.random.choice(entities)
#                 negative_samples.append((head, relation, negative_tail))
#     return negative_samples

# def generate_negative_samples(data, num_entities, ratio=1):
#     if not isinstance(data, list):
#         data = [data]  # If a single triple is passed, convert it into a list
#
#     negative_samples1 = []
#     for head, relation, tail in data:
#         for _ in range(ratio):
#             if np.random.random() < 0.5:
#                 negative_head = np.random.randint(num_entities)  # Randomly select an entity index as negative head
#                 negative_samples1.append((negative_head, relation, tail))
#             else:
#                 negative_tail = np.random.randint(num_entities)  # Randomly select an entity index as negative tail
#                 negative_samples1.append((head, relation, negative_tail))
#     return negative_samples1

def generate_negative_samples(data, num_entities, ratio=1):
    negative_samples1 = []
    for head, relation, tail in data:
        for _ in range(ratio):
            if np.random.random() < 0.5:
                # Randomly replace head
                neg_head = np.random.randint(num_entities)
                negative_samples1.append((neg_head, relation, tail))
            else:
                # Randomly replace tail
                neg_tail = np.random.randint(num_entities)
                negative_samples1.append((head, relation, neg_tail))
    return negative_samples1


# Read data from files
train_data = read_data('C:\\Users\\vikrant.singh\\PycharmProjects\\NASKG\\UMLS\\train.txt')
test_data = read_data('C:\\Users\\vikrant.singh\\PycharmProjects\\NASKG\\UMLS\\test.txt')

# Create mapping from entities and relations to unique ids
entities = set([head for head, _, _ in train_data] + [tail for _, _, tail in train_data])
relations = set([relation for _, relation, _ in train_data])
entity_to_id = {entity: idx for idx, entity in enumerate(entities)}
relation_to_id = {relation: idx for idx, relation in enumerate(relations)}

# Convert the entities and relations in the data to their unique ids
train_data = [(entity_to_id[head], relation_to_id[relation], entity_to_id[tail]) for head, relation, tail in train_data]
test_data = [(entity_to_id[head], relation_to_id[relation], entity_to_id[tail]) for head, relation, tail in test_data]
print(train_data)

# Generate negative samples
# negative_samples = generate_negative_samples(train_data, list(entity_to_id.values()))
num_entities = len(entity_to_id)
negative_samples = generate_negative_samples(train_data, num_entities)
train_data += negative_samples
train_labels = [1] * len(train_data) + [0] * len(negative_samples)

# Parameters
num_relations = len(relation_to_id)
embedding_dim = 50
batch_size = 32  # Define batch size

# Model
embedding_generator = EmbeddingGenerator(num_entities, num_relations, embedding_dim)
scoring_function = ScoringFunction(embedding_dim)
optimizer = optim.SGD(list(embedding_generator.parameters()) + list(scoring_function.parameters()), lr=0.01)
criterion = nn.BCEWithLogitsLoss()


# # Training
# for epoch in range(100):
#     optimizer.zero_grad()
#     total_loss = 0
#     for (head, relation, tail), label in zip(train_data, train_labels):
#         h, r = embedding_generator(torch.tensor([head]), torch.tensor([relation]))
#         t = embedding_generator.entity_embeddings(torch.tensor([tail]))
#         score = scoring_function(h, r, t)
#         # print("score", score)
#         loss = criterion(score, torch.tensor([[label]], dtype=torch.float))
#         total_loss += loss.item()
#         loss.backward()
#
#     optimizer.step()

# Specify the device (CPU or CUDA)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training
for epoch in range(100):
    permutation = torch.randperm(len(train_data))
    for i in range(0, len(train_data), batch_size):
        optimizer.zero_grad()
        indices = permutation[i:i + batch_size]
        batch_heads, batch_relations, batch_tails, batch_labels = zip(
            *[(train_data[index][0], train_data[index][1], train_data[index][2], train_labels[index]) for index in
              indices])

        batch_heads = torch.tensor(batch_heads, dtype=torch.long, device=device)
        batch_relations = torch.tensor(batch_relations, dtype=torch.long, device=device)
        batch_tails = torch.tensor(batch_tails, dtype=torch.long, device=device)
        batch_labels = torch.tensor(batch_labels, dtype=torch.float, device=device).view(-1, 1)

        h, r = embedding_generator(batch_heads, batch_relations)
        t = embedding_generator.entity_embeddings(batch_tails)
        scores = scoring_function(h, r, t)

        loss = criterion(scores, batch_labels)
        print()
        loss.backward()
        optimizer.step()

# Evaluation
ranks = []
num_entities = len(entity_to_id)

for index, (head, relation, tail) in enumerate(test_data):
    h, r = embedding_generator(torch.tensor([head]), torch.tensor([relation]))
    t = embedding_generator.entity_embeddings(torch.tensor([tail]))

    # Generate negative samples
    # negative_samples = [generate_negative_samples((head, relation, tail), num_entities) for _ in range(10)]
    negative_samples = generate_negative_samples([(head, relation, tail)], num_entities, ratio=5)

    scores = []
    print(negative_samples)
    for neg_head, neg_relation, neg_tail in negative_samples:
        h_neg, r_neg = embedding_generator(torch.tensor([neg_head]), torch.tensor([neg_relation]))
        t_neg = embedding_generator.entity_embeddings(torch.tensor([neg_tail]))
        score = scoring_function(h_neg, r_neg, t_neg)
        scores.append(score.item())

    # Include the score of the positive sample
    score = scoring_function(h, r, t)
    scores.append(score.item())

    # Rank the scores
    rank = sorted(scores, reverse=True).index(score.item()) + 1
    ranks.append(rank)

# Calculate Mean Reciprocal Rank (MRR)
mrr = np.mean([1.0 / rank for rank in ranks])
print(f'Mean Reciprocal Rank (MRR): {mrr}')
