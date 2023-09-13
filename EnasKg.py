import torch.optim as optim
import torch.utils.data as data_utils
import numpy as np
import nni
import torch
import logging
import torch.nn as nn
from nni.retiarii import model_wrapper


class NasEmbeddingGenerator(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(NasEmbeddingGenerator, self).__init__()
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        # self.entity_embeddings = LayerChoice([
        #     nn.Embedding(num_entities, embedding_dim),
        #     nn.Embedding(num_entities, embedding_dim * 2)
        # ])
        # self.relation_embeddings = LayerChoice([
        #     nn.Embedding(num_relations, embedding_dim),
        #     nn.Embedding(num_relations, embedding_dim * 2)
        # ])

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


class NasModel(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(NasModel, self).__init__()
        self.nas_embedding_generator = NasEmbeddingGenerator(num_entities, num_relations, embedding_dim)
        self.scoring_function = ScoringFunction(embedding_dim)

    def forward(self, heads, relations, tails):
        h, r = self.nas_embedding_generator(heads, relations)
        t = self.nas_embedding_generator.entity_embeddings(tails)
        return self.scoring_function(h, r, t)

#
# class FunctionSPaces():
#     """ Learning Knowledge Neural Graphs"""
#     """ Learning Neural Networks for Knowledge Graphs"""
#
#     def __init__(self, args):
#         super().__init__(args)
#         self.name = 'FMult'
#         self.entity_embeddings = torch.nn.Embedding(self.num_entities, self.embedding_dim)
#         self.relation_embeddings = torch.nn.Embedding(self.num_relations, self.embedding_dim)
#         self.param_init(self.entity_embeddings.weight.data), self.param_init(self.relation_embeddings.weight.data)
#         self.k = int(np.sqrt(self.embedding_dim // 2))
#         self.num_sample = 50
#         # self.gamma = torch.rand(self.k, self.num_sample) [0,1) uniform=> worse results
#         self.gamma = torch.randn(self.k, self.num_sample)  # N(0,1)
#
#         from scipy.special import roots_legendre, eval_legendre
#         roots, weights = roots_legendre(self.num_sample)
#         self.roots = torch.from_numpy(roots).repeat(self.k, 1).float()  # shape self.k by self.n
#         self.weights = torch.from_numpy(weights).reshape(1, -1).float()  # shape 1 by self.n
#
#
#     def compute_func(self, weights: torch.FloatTensor, x) -> torch.FloatTensor:
#         n = len(weights)
#         # Weights for two linear layers.
#         w1, w2 = torch.hsplit(weights, 2)
#         # (1) Construct two-layered neural network
#         w1 = w1.view(n, self.k, self.k)
#         w2 = w2.view(n, self.k, self.k)
#         # (2) Forward Pass
#         out1 = torch.tanh(w1 @ x)  # torch.sigmoid => worse results
#         out2 = w2 @ out1
#         return out2  # no non-linearity => better results
#
#     def chain_func(self, weights, x: torch.FloatTensor):
#         n = len(weights)
#         # Weights for two linear layers.
#         w1, w2 = torch.hsplit(weights, 2)
#         # (1) Construct two-layered neural network
#         w1 = w1.view(n, self.k, self.k)
#         w2 = w2.view(n, self.k, self.k)
#         # (2) Perform the forward pass
#         out1 = torch.tanh(torch.bmm(w1, x))
#         out2 = torch.bmm(w2, out1)
#         return out2
#
#     def forward(self, idx_triple: torch.Tensor) -> torch.Tensor:
#         # (1) Retrieve embeddings: batch, \mathbb R^d
#         head_ent_emb, rel_ent_emb, tail_ent_emb = self.get_triple_representation(idx_triple)
#         # (2) Compute NNs on \Gamma
#         # Logits via FDistMult...
#         # h_x = self.compute_func(head_ent_emb, x=self.gamma)  # batch, \mathbb{R}^k, |\Gamma|
#         # r_x = self.compute_func(rel_ent_emb, x=self.gamma)  # batch, \mathbb{R}^k, |\Gamma|
#         # t_x = self.compute_func(tail_ent_emb, x=self.gamma)  # batch, \mathbb{R}^k, |\Gamma|
#         # out = h_x * r_x * t_x  # batch, \mathbb{R}^k, |gamma|
#         # (2) Compute NNs on \Gamma
#         self.gamma=self.gamma.to(head_ent_emb.device)
#
#         h_x = self.compute_func(head_ent_emb, x=self.gamma)  # batch, \mathbb{R}^k, |\Gamma|
#         t_x = self.compute_func(tail_ent_emb, x=self.gamma)  # batch, \mathbb{R}^k, |\Gamma|
#         r_h_x = self.chain_func(weights=rel_ent_emb, x=h_x)  # batch, \mathbb{R}^k, |\Gamma|
#         # (3) Compute |\Gamma| predictions
#         out = torch.sum(r_h_x * t_x, dim=1)  # batch, |gamma| #
#         # (4) Average (3) over \Gamma
#         out = torch.mean(out, dim=1)  # batch
#         return out
#
#

def read_data(file_path):
    with open(file_path, 'r') as file:
        data = []
        for line in file:
            head, relation, tail = line.strip().split()
            data.append((head, relation, tail))
        return data


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

# Parameters
num_entities = len(entity_to_id)
negative_samples = generate_negative_samples(train_data, num_entities)
train_data += negative_samples
# train_labels = [1] * len(train_data) + [0] * len(negative_samples)
train_labels = [1] * (len(train_data) - len(negative_samples)) + [0] * len(negative_samples)


# Parameter
# num_relations = len(relation_to_id)


def objective_function():
    # Parameters
    num_entities = len(entity_to_id)
    num_relations = len(relation_to_id)
    embedding_dim = 50
    batch_size = 32  # You can adjust the batch size

    # Model
    nas_embedding_generator = NasEmbeddingGenerator(num_entities, num_relations, embedding_dim)
    scoring_function = ScoringFunction(embedding_dim)
    optimizer = optim.Adam(list(nas_embedding_generator.parameters()) + list(scoring_function.parameters()), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()

    # Convert train_data to tensor format
    train_data_tensor = torch.tensor(train_data, dtype=torch.long)
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.float)

    # print("Hello", train_data_tensor.size())
    # print("Hello 1", train_labels_tensor.size())
    # assert train_data_tensor.size(0) == train_labels_tensor.size(0), "Size mismatch between tensors"
    # Create DataLoader for mini-batches
    dataset = data_utils.TensorDataset(train_data_tensor, train_labels_tensor)
    data_loader = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training
    for epoch in range(100):
        for batch_data, batch_labels in data_loader:
            optimizer.zero_grad() # zeros the gradient
            total_loss = 0

            heads, relations, tails = batch_data[:, 0], batch_data[:, 1], batch_data[:, 2]
            h, r = nas_embedding_generator(heads, relations)
            t = nas_embedding_generator.entity_embeddings(tails)
            scores = scoring_function(h, r, t)
            loss = criterion(scores, batch_labels.view(-1, 1))
            total_loss += loss.item()
            loss.backward()

            optimizer.step()

    # Evaluation
    ranks = []
    num_entities = len(entity_to_id)

    for index, (head, relation, tail) in enumerate(test_data):
        h, r = nas_embedding_generator(torch.tensor([head]), torch.tensor([relation]))
        t = nas_embedding_generator.entity_embeddings(torch.tensor([tail]))

        negative_samples = generate_negative_samples([(head, relation, tail)], num_entities, ratio=5)

        scores = []
        for neg_head, neg_relation, neg_tail in negative_samples:
            h_neg, r_neg = nas_embedding_generator(torch.tensor([neg_head]), torch.tensor([neg_relation]))
            t_neg = nas_embedding_generator.entity_embeddings(torch.tensor([neg_tail]))
            score = scoring_function(h_neg, r_neg, t_neg)
            scores.append(score.item())

        # Include the score of the positive sample
        score = scoring_function(h, r, t)
        scores.append(score.item())

        # Rank the scores
        rank = sorted(scores, reverse=True).index(score.item()) + 1
        ranks.append(rank)

    # Calculate Mean Reciprocal Rank (MRR)
    mrr = np.mean([1.0 / rank for rank in ranks]).item()
    logging.info(f'Mean Reciprocal Rank (MRR): {mrr}')

    # Report the result to NNI
    nni.report_final_result(mrr)


params = nni.get_next_parameter()
objective_function()

# embedding_dim = 50
# batch_size = 32  # Define batch size
#
# # Initialize the model
# model = NasModel(num_entities, num_relations, embedding_dim)
# criterion = nn.BCEWithLogitsLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01)
# # Model
#
# embedding_generator = NasEmbeddingGenerator(num_entities, num_relations, embedding_dim, params)
# scoring_function = ScoringFunction(embedding_dim)
# optimizer = optim.SGD(list(embedding_generator.parameters()) + list(scoring_function.parameters()), lr=0.01)
# criterion = nn.BCEWithLogitsLoss()
# # Apply ENAS
# apply_fixed_architecture(model, "enas")  # enas is the chosen architecture
#
# # Training
# for epoch in range(100):
#     optimizer.zero_grad()
#     total_loss = 0
#     for (head, relation, tail), label in zip(train_data, train_labels):
#         score = model(torch.tensor([head]), torch.tensor([relation]), torch.tensor([tail]))
#         loss = criterion(score, torch.tensor([[label]], dtype=torch.float))
#         total_loss += loss.item()
#         loss.backward()
#     optimizer.step()
#
# # Evaluation
# def objective_function(params):
#     # ... code to create and train the model with the given params ...
#
#     # Evaluation
#     ranks = []
#     num_entities = len(entity_to_id)
#     for index, (head, relation, tail) in enumerate(test_data):
#         h, r = embedding_generator(torch.tensor([head]), torch.tensor([relation]))
#         t = embedding_generator.entity_embeddings(torch.tensor([tail]))
#
#         # Generate negative samples
#         negative_samples = generate_negative_samples([(head, relation, tail)], num_entities, ratio=5)
#
#         scores = []
#         for neg_head, neg_relation, neg_tail in negative_samples:
#             h_neg, r_neg = embedding_generator(torch.tensor([neg_head]), torch.tensor([neg_relation]))
#             t_neg = embedding_generator.entity_embeddings(torch.tensor([neg_tail]))
#             score = scoring_function(h_neg, r_neg, t_neg)
#             scores.append(score.item())
#
#         # Include the score of the positive sample
#         score = scoring_function(h, r, t)
#         scores.append(score.item())
#
#         # Rank the scores
#         rank = sorted(scores, reverse=True).index(score.item()) + 1
#         ranks.append(rank)
#
#     # Calculate Mean Reciprocal Rank (MRR)
#     mrr = np.mean([1.0 / rank for rank in ranks])
#
#     # Return the MRR as the objective to maximize
#     return mrr
#
#
# # NNI will call this function with different params
# nni.report_final_result(objective_function(params))
