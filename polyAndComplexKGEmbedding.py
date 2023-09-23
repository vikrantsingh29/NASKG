import random
import torch
import torch.nn as nn
import torch.optim as optim

# # Assuming entities are represented by integers for simplicity
# all_entities = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# all_relations = [10, 20, 30]
# # dummy_triples = [(1, 10, 2), (2, 20, 3), (3, 10, 4)]
# train_triples = [(1, 10, 2), (2, 20, 3), (3, 10, 4), (4, 20, 5), (5, 10, 6)]
# test_triples = [(1, 20, 3), (3, 20, 5), (4, 10, 2)]
#
# # Map entities and relations to indices
# entity_to_index = {entity: idx for idx, entity in enumerate(all_entities)}
# relation_to_index = {relation: idx for idx, relation in enumerate(all_relations)}
#
# # Initialize random embeddings
# num_entities = 10
# num_relations = 3
# embedding_dim = 50





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
path = 'C:\\Users\\vikrant.singh\\PycharmProjects\\NASKG\\Countries-S1'
train_triples_string = read_triples_from_file(path + '\\train.txt')
test_triples_string = read_triples_from_file(path + '\\test.txt')
valid_triples_string = read_triples_from_file(path + '\\valid.txt')

# Combine all triples to extract unique entities and relations
all_triples = train_triples_string + test_triples_string + valid_triples_string

# Get unique entities and relations
all_entities = list(set([h for h, _, _ in all_triples] + [t for _, _, t in all_triples]))
all_relations = list(set([r for _, r, _ in all_triples]))


# Map entities and relations to indices
entity_to_index = {entity: idx for idx, entity in enumerate(all_entities)}
relation_to_index = {relation: idx for idx, relation in enumerate(all_relations)}

# Convert the string triples to their respective indices
train_triples = [(entity_to_index[h], relation_to_index[r], entity_to_index[t]) for h, r, t in train_triples_string]
test_triples = [(entity_to_index[h], relation_to_index[r], entity_to_index[t]) for h, r, t in test_triples_string]
valid_triples = [(entity_to_index[h], relation_to_index[r], entity_to_index[t]) for h, r, t in valid_triples_string]

# Determine the number of unique entities and relations based on the loaded data
num_entities = len(all_entities)
num_relations = len(all_relations)
embedding_dim = 50


entity_embeddings = nn.Embedding(num_entities, embedding_dim)
relation_embeddings = nn.Embedding(num_relations, embedding_dim)


all_entity_indices = list(range(num_entities))

# Loss
margin = 1.0
criterion = nn.MarginRankingLoss(margin=margin)

# Optimizer
optimizer = optim.Adam([
    {'params': entity_embeddings.parameters()},
    {'params': relation_embeddings.parameters()}
], lr=0.001)


def generate_negative_triples(true_triple, all_entity_indices, num_samples, true_triples_list):
    h, r, t = true_triple
    negative_triples = set()

    while len(negative_triples) < num_samples:  # Generate num_samples number of negative triples
        t_corrupted = random.choice(all_entity_indices)  # Corrupt tail

        if (h, r,
            t_corrupted) not in true_triples_list and t_corrupted != t:
            negative_triples.add((h, r, t_corrupted))

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
    # poly_vector = torch.cat([sum([embedding[i] * (x ** i) for i in range(len(embedding))]).unsqueeze(0) for x in x_samples])

    embedding = embedding
    x_samples = x_samples

    # Extend dimensions for broadcasting
    emb_expanded = embedding.unsqueeze(0)  # Shape: [1, embedding_dim]
    x_expanded = x_samples.unsqueeze(1)  # Shape: [num_samples, 1]

    # Calculate polynomial values using broadcasting
    powers_of_x = x_expanded ** torch.arange(len(embedding))  # Shape: [num_samples, embedding_dim]
    poly_vector = (emb_expanded * powers_of_x).sum(dim=-1)  # Element-wise multiplication followed by sum

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
    # complex_representation = torch.stack((real_part, imag_part), dim=-1)  # Shape: [x_samples, 2]
    # Aggregate real and imaginary parts
    complex_representation = (real_part + imag_part) / 2  # Shape: [x_samples]

    return complex_representation


def compute_score(h_idx, r_idx, t_idx):
    x_samples = torch.linspace(-1, 1, 50)

    # h_emb_idx = entity_to_index[h_idx]
    # r_emb_idx = relation_to_index[r_idx]
    # t_emb_idx = entity_to_index[t_idx]

    h = entity_embeddings(torch.tensor([h_idx]))[0]
    r = relation_embeddings(torch.tensor([r_idx]))[0]
    t = entity_embeddings(torch.tensor([t_idx]))[0]

    fh = complex_num(h, x_samples)
    ft = complex_num(t, x_samples)
    fhx = complex_num(fh, x_samples)
    h_r_combined = h * r  # element-wise multiplication of h and r
    frh = complex_num(h_r_combined, x_samples)

    score = torch.trapz(frh * ft, x_samples, dim=0)

    return score


def compute_vtp_score(h_idx, r_idx, t_idx):
    x_samples = torch.linspace(-1, 1, 50)

    # Fetching the embeddings for the head, relation, and tail entities
    h = entity_embeddings(torch.tensor([h_idx]))[0]
    r = relation_embeddings(torch.tensor([r_idx]))[0]
    t = entity_embeddings(torch.tensor([t_idx]))[0]

    # Transform the embeddings using the polynomial function
    fh = complex_num(h, x_samples)
    fr = complex_num(r, x_samples)
    ft = complex_num(t, x_samples)

    # Compute the VTP score using the transformed embeddings
    score = torch.sum(fh * (fr * ft))

    return score

def compute_trilinear_score(h_idx, r_idx, t_idx):
    x_samples = torch.linspace(-1, 1, 50)

    h = entity_embeddings(torch.tensor([h_idx]))[0]
    r = relation_embeddings(torch.tensor([r_idx]))[0]
    t = entity_embeddings(torch.tensor([t_idx]))[0]

    fh = complex_num(h, x_samples)
    fr = complex_num(r, x_samples)
    ft = complex_num(t, x_samples)

    score = torch.sum(fh * fr * ft)  # Element-wise multiplication across the three vectors

    return score


def compute_loss(positive_score, negative_score):
    y = torch.ones_like(positive_score)  # The target tensor assuming positive_score should be larger than negative_score
    loss = criterion(positive_score, negative_score, y)
    return loss


def compute_MRR(test_triples):
    rr_sum = 0.0

    for h_idx, r_idx, t_true_idx in test_triples:
        scores = []

        # Score all entities as potential tails
        for t_idx in all_entity_indices:

            score = compute_trilinear_score(h_idx, r_idx, t_idx)
            scores.append((t_idx, score.item()))

        # Sort entities based on their scores
        ranked_entities = sorted(scores, key=lambda x: x[1], reverse=True)
        rank = [idx for idx, (entity, _) in enumerate(ranked_entities) if entity == t_true_idx][0] + 1
        # Add the reciprocal rank to the sum
        rr_sum += 1.0 / rank

    # Compute the mean reciprocal rank
    mrr = rr_sum / len(test_triples)
    return mrr


def forward(triples):
    total_loss = 0.0

    for h_idx, r_idx, t_idx in triples:
        # Compute positive score
        positive_score = compute_trilinear_score(h_idx, r_idx, t_idx)

        accumulated_loss = 0.0
        num_neg_samples = 5
        negative_triples = generate_negative_triples((h_idx, r_idx, t_idx), all_entity_indices, num_neg_samples,
                                                     train_triples)

        for h_neg_idx, r_neg_idx, t_neg_idx in negative_triples:
            # Compute negative score
            negative_score = compute_trilinear_score(h_neg_idx, r_neg_idx, t_neg_idx)

            # Compute loss for the positive and negative score pair
            loss = compute_loss(positive_score, negative_score)
            accumulated_loss += loss

        accumulated_loss /= num_neg_samples
        accumulated_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += accumulated_loss.item()

    return total_loss


#
# def forward():
#     total_loss = 0.0
#
#     for h_idx, r_idx, t_idx in dummy_triples:
#         x_samples = torch.linspace(-1, 1, 50)
#
#         h_emb_idx = entity_to_index[h_idx]
#         r_emb_idx = relation_to_index[r_idx]
#         t_emb_idx = entity_to_index[t_idx]
#
#         h = entity_embeddings(torch.tensor([h_emb_idx]))[0]
#         r = relation_embeddings(torch.tensor([r_emb_idx]))[0]
#         t = entity_embeddings(torch.tensor([t_emb_idx]))[0]
#
#         h_poly = complex_num(h, x_samples)
#         t_poly = complex_num(t, x_samples)
#         r_h_poly = complex_num(h_poly, x_samples)
#         positive_score = torch.trapz(r_h_poly * t_poly, x_samples , dim=0)
#
#         accumulated_loss = 0.0
#         num_neg_samples = 5
#         negative_triples = generate_negative_triples((h_idx, r_idx, t_idx), all_entities, num_neg_samples,
#                                                      dummy_triples)
#
#         for h_neg_idx, r_neg_idx, t_neg_idx in negative_triples:
#             h_neg_emb_idx = entity_to_index[h_neg_idx]
#             r_neg_emb_idx = relation_to_index[r_neg_idx]
#             t_neg_emb_idx = entity_to_index[t_neg_idx]
#
#             h_neg = entity_embeddings(torch.tensor([h_neg_emb_idx]))[0]
#             r_neg = relation_embeddings(torch.tensor([r_neg_emb_idx]))[0]
#             t_neg = entity_embeddings(torch.tensor([t_neg_emb_idx]))[0]
#
#             h_neg_poly = complex_num(h_neg, x_samples)
#             t_neg_poly = complex_num(t_neg, x_samples)
#             r_h_neg_poly = complex_num(h_neg_poly, x_samples)
#             negative_score = torch.trapz(r_h_neg_poly * t_neg_poly, x_samples , dim=0)
#
#             # y = torch.tensor(1, dtype=torch.float32)  # Assuming positive_score should be larger
#             y = torch.tensor([1], dtype=torch.float32).expand_as(positive_score)
#             loss = criterion(positive_score, negative_score, y)
#             accumulated_loss += loss
#
#         accumulated_loss /= num_neg_samples
#         accumulated_loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()
#
#         total_loss += accumulated_loss.item()
#
#     return total_loss


# Training Loop
num_epochs = 100

for epoch in range(num_epochs):
    total_loss = forward(train_triples)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss}")

# After training, evaluate MRR on test_triples
mrr_value = compute_MRR(test_triples)
print(f"Mean Reciprocal Rank (MRR) on Test Data: {mrr_value}")
