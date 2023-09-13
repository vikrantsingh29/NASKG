import random
import torch
import torch.nn as nn
import torch.optim as optim

# Assuming entities are represented by integers for simplicity
all_entities = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
all_relations = [10, 20, 30]
dummy_triples = [(1, 10, 2), (2, 20, 3), (3, 10, 4)]


# Map entities and relations to indices
entity_to_index = {entity: idx for idx, entity in enumerate(all_entities)}
relation_to_index = {relation: idx for idx, relation in enumerate(all_relations)}

# Initialize random embeddings
num_entities = 10
num_relations = 3
embedding_dim = 50

entity_embeddings = nn.Embedding(num_entities, embedding_dim)
relation_embeddings = nn.Embedding(num_relations, embedding_dim)

# Loss
margin = 1.0
criterion = nn.MarginRankingLoss(margin=margin)

# Optimizer
optimizer = optim.Adam([
    {'params': entity_embeddings.parameters()},
    {'params': relation_embeddings.parameters()}
], lr=0.001)


def generate_negative_triples(true_triple, all_entities, num_samples, true_triples_list):
    h, r, t = true_triple
    negative_triples = set()

    while len(negative_triples) < num_samples:  # Generate num_samples number of negative triples
        t_corrupted = random.choice(all_entities)  # Corrupt tail

        # Check to ensure we don't accidentally generate a positive sample
        # And also check that the corrupted triple isn't a true triple in the dataset
        if (h, r,
            t_corrupted) not in true_triples_list and t_corrupted != t:  # Ensure the corrupted entity is not the same as the original
            negative_triples.add((h, r, t_corrupted))  # Add the corrupted triple to the set

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
    real_part = (embedding * torch.cos(x_samples.unsqueeze(-1))).sum(-1)
    imag_part = (embedding * torch.sin(x_samples.unsqueeze(-1))).sum(-1)

    # Combine real and imaginary parts
    complex_representation = torch.stack((real_part, imag_part), dim=-1)  # Shape: [x_samples, 2]

    return complex_representation


def forward():
    total_loss = 0.0

    for h_idx, r_idx, t_idx in dummy_triples:
        x_samples = torch.linspace(-1, 1, 50)

        h_emb_idx = entity_to_index[h_idx]
        r_emb_idx = relation_to_index[r_idx]
        t_emb_idx = entity_to_index[t_idx]

        h = entity_embeddings(torch.tensor([h_emb_idx]))[0]
        r = relation_embeddings(torch.tensor([r_emb_idx]))[0]
        t = entity_embeddings(torch.tensor([t_emb_idx]))[0]

        h_poly = complex_num(h, x_samples)
        t_poly = complex_num(t, x_samples)
        r_h_poly = complex_num(h_poly, x_samples)
        positive_score = torch.trapz(r_h_poly * t_poly, x_samples , dim=0)

        accumulated_loss = 0.0
        num_neg_samples = 5
        negative_triples = generate_negative_triples((h_idx, r_idx, t_idx), all_entities, num_neg_samples,
                                                     dummy_triples)

        for h_neg_idx, r_neg_idx, t_neg_idx in negative_triples:
            h_neg_emb_idx = entity_to_index[h_neg_idx]
            r_neg_emb_idx = relation_to_index[r_neg_idx]
            t_neg_emb_idx = entity_to_index[t_neg_idx]

            h_neg = entity_embeddings(torch.tensor([h_neg_emb_idx]))[0]
            r_neg = relation_embeddings(torch.tensor([r_neg_emb_idx]))[0]
            t_neg = entity_embeddings(torch.tensor([t_neg_emb_idx]))[0]

            h_neg_poly = complex_num(h_neg, x_samples)
            t_neg_poly = complex_num(t_neg, x_samples)
            r_h_neg_poly = complex_num(h_neg_poly, x_samples)
            negative_score = torch.trapz(r_h_neg_poly * t_neg_poly, x_samples , dim=0)

            # y = torch.tensor(1, dtype=torch.float32)  # Assuming positive_score should be larger
            y = torch.tensor([1], dtype=torch.float32).expand_as(positive_score)
            loss = criterion(positive_score, negative_score, y)
            accumulated_loss += loss

        accumulated_loss /= num_neg_samples
        accumulated_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += accumulated_loss.item()

    return total_loss


# Training Loop
num_epochs = 100

for epoch in range(num_epochs):
    total_loss = forward()
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss}")
