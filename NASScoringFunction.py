import torch.optim as optim
import torch.utils.data as data_utils
import numpy as np
import nni
import torch
import logging
import torch.nn as nn


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
        self.W1 = nn.Linear(embedding_dim * 3, embedding_dim)  # Adjust the input size
        self.W2 = nn.Linear(embedding_dim, 1)

    def forward(self, h, r, t):
        x = torch.cat([h, r, t], dim=1)
        x = torch.relu(self.W1(x))
        return self.W2(x)


class NasModel(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(NasModel, self).__init__()
        self.embedding_generator = EmbeddingGenerator(num_entities, num_relations, embedding_dim)
        self.scoring_function = ScoringFunction(embedding_dim)

    def forward(self, heads, relations, tails):
        h, r = self.embedding_generator(heads, relations)
        t = self.embedding_generator.entity_embeddings(tails)
        return self.scoring_function(h, r, t)


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


def objective_function():
    # Parameters
    num_entities = len(entity_to_id)
    num_relations = len(relation_to_id)
    embedding_dim = 50
    batch_size = 32  # You can adjust the batch size

    # Model
    nas_model = NasModel(num_entities, num_relations, embedding_dim)
    optimizer = optim.Adam(nas_model.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()

    # Convert train_data to tensor format
    train_data_tensor = torch.tensor(train_data, dtype=torch.long)
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.float)

    # Create DataLoader for mini-batches
    dataset = data_utils.TensorDataset(train_data_tensor, train_labels_tensor)
    data_loader = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training
    for epoch in range(100):
        for batch_data, batch_labels in data_loader:
            optimizer.zero_grad()  # zeros the gradient
            total_loss = 0

            heads, relations, tails = batch_data[:, 0], batch_data[:, 1], batch_data[:, 2]
            h, r = nas_model.embedding_generator(heads, relations)
            t = nas_model.embedding_generator.entity_embeddings(tails)

            # # Calculate scores for all possible tails
            # scores_all = []
            # for tail_candidate in range(num_entities):
            #     t_cand = nas_model.embedding_generator.entity_embeddings(torch.tensor([tail_candidate]))
            #     t_cand = t_cand.expand_as(h)  # Expand the dimensions of t_cand to match h
            #     score = nas_model.scoring_function(h, r, t_cand).item()
            #     scores_all.append(score)
            #
            # # Calculate the loss
            # positive_scores = torch.tensor([scores_all[tail] for tail in tails])
            # negative_scores = torch.tensor([score for idx, score in enumerate(scores_all) if idx not in tails])
            # loss = -torch.log(torch.sigmoid(positive_scores) + 1e-6) - torch.log(
            #     1 - torch.sigmoid(negative_scores) + 1e-6)
            # loss = torch.mean(loss)
            # total_loss += loss.item()
            # loss.backward()
            #
            # optimizer.step()
            # In your training loop:
            scores_all = []
            for tail_candidate in range(num_entities):
                t_cand = nas_model.embedding_generator.entity_embeddings(torch.tensor([tail_candidate]))
                t_cand = t_cand.expand_as(h)  # Expand the dimensions of t_cand to match h
                score = nas_model.scoring_function(h, r, t_cand).squeeze()  # Remove extra dimensions
                scores_all.append(score)

            # Convert scores_all to a tensor
            scores_all = torch.stack(scores_all, dim=0)  # Concatenate along the first dimension

            # Create labels tensor
            labels = torch.zeros(num_entities, dtype=torch.float)
            labels[tails] = 1.0

            # Calculate the loss
            loss = criterion(scores_all, labels)
            total_loss += loss.item()
            loss.backward()

            optimizer.step()

    # Evaluation
    ranks = []
    num_entities = len(entity_to_id)

    for index, (head, relation, tail) in enumerate(test_data):
        h = nas_model.embedding_generator.entity_embeddings(torch.tensor([head]))
        r = nas_model.embedding_generator.relation_embeddings(torch.tensor([relation]))
        t = nas_model.embedding_generator.entity_embeddings(torch.tensor([tail]))

        # Calculate scores for all possible tails
        scores_all = []
        for tail_candidate in range(num_entities):
            t_cand = nas_model.embedding_generator.entity_embeddings(torch.tensor([tail_candidate]))
            score = nas_model.scoring_function(h, r, t_cand).item()
            scores_all.append(score)

        # Rank the scores
        rank = sorted(scores_all, reverse=True).index(scores_all[tail]) + 1
        ranks.append(rank)

    # Calculate Mean Reciprocal Rank (MRR)
    mrr = np.mean([1.0 / rank for rank in ranks]).item()
    logging.info(f'Mean Reciprocal Rank (MRR): {mrr}')

    # Report the final result to NNI
    nni.report_final_result(mrr)


# Call the objective function to start the training process
objective_function()
