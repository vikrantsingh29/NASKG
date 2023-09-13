import random
import torch
import torch.nn as nn
import torch.optim as optim


# # Assuming entities are represented by integers for simplicity
# all_entities = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# all_relations = [10, 20, 30]
# dummy_triples = [(1, 10, 2), (2, 20, 3), (3, 10, 4)]
#
#
# # Map entities and relations to indices
# entity_to_index = {entity: idx for idx, entity in enumerate(all_entities)}
# relation_to_index = {relation: idx for idx, relation in enumerate(all_relations)}
#
# # Initialize random embeddings
# num_entities = 10
# num_relations = 3
# embedding_dim = 50
#
# entity_embeddings = nn.Embedding(num_entities, embedding_dim)
# relation_embeddings = nn.Embedding(num_relations, embedding_dim)


def read_triples_from_file(filename):
    """
    Reads triples from a file.
    Assumes each line of the file is in the format: head relation tail.
    """
    triples = []
    with open(filename, 'r') as f:
        for line in f:
            h, r, t = line.strip().split()
            triples.append((h, r, t))
    return triples


# Read triples from each file
path = 'C:\\Users\\vikrant.singh\\PycharmProjects\\NASKG\\UMLS'
train_triples = read_triples_from_file(path + '\\train.txt')
test_triples = read_triples_from_file(path + '\\test.txt')
valid_triples = read_triples_from_file(path + '\\valid.txt')

# Combine all triples to extract unique entities and relations
all_triples = train_triples + test_triples + valid_triples

# Get unique entities and relations
all_entities = list(set([h for h, _, _ in all_triples] + [t for _, _, t in all_triples]))
all_relations = list(set([r for _, r, _ in all_triples]))

# Map entities and relations to indices
entity_to_index = {entity: idx for idx, entity in enumerate(all_entities)}
relation_to_index = {relation: idx for idx, relation in enumerate(all_relations)}

# Convert the string triples to their respective indices
train_triples_idx = [(entity_to_index[h], relation_to_index[r], entity_to_index[t]) for h, r, t in train_triples]
test_triples_idx = [(entity_to_index[h], relation_to_index[r], entity_to_index[t]) for h, r, t in test_triples]
valid_triples_idx = [(entity_to_index[h], relation_to_index[r], entity_to_index[t]) for h, r, t in valid_triples]

# Determine the number of unique entities and relations based on the loaded data
num_entities = len(all_entities)
num_relations = len(all_relations)
embedding_dim = 50

# Initialize embeddings with the new determined sizes
entity_embeddings = nn.Embedding(num_entities, embedding_dim)
relation_embeddings = nn.Embedding(num_relations, embedding_dim)

# Loss
margin = 1.0
criterion = nn.MarginRankingLoss(margin=margin)

# Optimizer
optimizer = optim.Adam([
    {'params': entity_embeddings.parameters()},
    {'params': relation_embeddings.parameters()}
], lr=0.0001)


def generate_negative_triples(true_triple, all_entities, num_samples, true_triples_list):
    h, r, t = true_triple
    negative_triples = set()

    while len(negative_triples) < num_samples:
        t_corrupted_idx = random.choice(range(len(all_entities)))  # Instead of choosing an entity, choose its index

        # Check to ensure we don't accidentally generate a positive sample
        # And also check that the corrupted triple isn't a true triple in the dataset
        if (h, r, t_corrupted_idx) not in true_triples_list and t_corrupted_idx != t:
            negative_triples.add((h, r, t_corrupted_idx))

    return list(negative_triples)


def polynomial(embedding, x_samples):
    """
    Computes a polynomial given an embedding and a set of x_samples.
    Returns a polynomial vector representation.
    """
    # The polynomial is in the form of a_0 + a_1*x + a_2*x^2 + ...
    # Each element of the embedding vector will have its own set of coefficients.
    # The result for each dimension will be a vector of size x_samples.
    # return torch.stack([(embedding * (x ** torch.arange(embedding_dim))).sum(-1) for x in x_samples])
    poly_vector = torch.cat(
        [sum([embedding[i] * (x ** i) for i in range(len(embedding))]).unsqueeze(0) for x in x_samples])
    return poly_vector


def complex_num(embedding, x_samples):
    """
    Computes a complex number representation given an embedding and a set of x_samples.
    Returns a complex number representation in the form of c = cos(x) + i*sin(x).
    """
    # Compute the real and imaginary parts
    real_part = (embedding * torch.cos(x_samples.unsqueeze(-1))).sum(-1)  # Shape: [x_samples]
    imag_part = (embedding * torch.sin(x_samples.unsqueeze(-1))).sum(-1)  # Shape: [x_samples]

    # Combine real and imaginary parts
    complex_representation = torch.stack((real_part, imag_part), dim=-1)  # Shape: [x_samples, 2]

    return complex_representation  # Shape: [x_samples, 2]


def forward():
    total_loss = 0.0

    for h_idx, r_idx, t_idx in train_triples_idx:  # Iterate over all positive triples
        x_samples = torch.linspace(-1, 1, 50)  # Generate 50 x_samples between -1 and 1

        # Fetch the embeddings directly using indices
        h = entity_embeddings(torch.tensor([h_idx]))[0]  # Shape: [embedding_dim]
        r = relation_embeddings(torch.tensor([r_idx]))[0]  # Shape: [embedding_dim]
        t = entity_embeddings(torch.tensor([t_idx]))[0]  # Shape: [embedding_dim]

        h_poly = complex_num(h, x_samples)  # Shape: [x_samples, 2]
        t_poly = complex_num(t, x_samples)  # Shape: [x_samples, 2]
        r_h_poly = complex_num(h_poly, x_samples)  # Shape: [x_samples, 2]
        positive_score = torch.trapz(r_h_poly * t_poly, x_samples, dim=0)  # Shape: [2]

        accumulated_loss = 0.0  # Initialize loss for the positive triple
        num_neg_samples = 5  # Number of negative samples to generate
        negative_triples = generate_negative_triples((h_idx, r_idx, t_idx), all_entities, num_neg_samples,
                                                     train_triples_idx)  # Generate negative triples

        for h_neg_idx, r_neg_idx, t_neg_idx in negative_triples:  # Iterate over all negative triples

            h_neg = entity_embeddings(torch.tensor([h_neg_idx]))[0]  # Shape: [embedding_dim]
            r_neg = relation_embeddings(torch.tensor([r_neg_idx]))[0]  # Shape: [embedding_dim]
            t_neg = entity_embeddings(torch.tensor([t_neg_idx]))[0]  # Shape: [embedding_dim]

            h_neg_poly = complex_num(h_neg, x_samples)  # Shape: [x_samples, 2]
            t_neg_poly = complex_num(t_neg, x_samples)  # Shape: [x_samples, 2]
            r_h_neg_poly = complex_num(h_neg_poly, x_samples)  # Shape: [x_samples, 2]
            negative_score = torch.trapz(r_h_neg_poly * t_neg_poly, x_samples, dim=0)  # Shape: [2]

            y = torch.tensor(1, dtype=torch.float32).squeeze() # Positive label
            loss = criterion(positive_score[..., 0], negative_score[..., 0],y)  # Considering only real parts for loss calculation

            accumulated_loss += loss

        accumulated_loss /= num_neg_samples  # Average loss over the negative samples
        accumulated_loss.backward()  # Backpropagate the loss

        optimizer.step()  # Update the parameters
        optimizer.zero_grad()  # Clear the gradients

        total_loss += accumulated_loss.item()  # Accumulate the loss

    return total_loss / len(train_triples_idx)  # Return the average loss


# Training Loop
num_epochs = 10

for epoch in range(num_epochs):
    total_loss = forward()
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss}")
