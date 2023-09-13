# import numpy as np
# import numpy as np
# from sklearn.preprocessing import LabelEncoder
#
# # Grid discretization
# N = 10
# M = 10
# grid = np.zeros((N, M))
#
# # Load data
# data_lines = open('C:\\Users\\vikrant.singh\\Downloads\\Hidden Markov Model\\data_vikrant\\data10_14_nrw_18-19.txt', 'r').readlines()
# data = [line.split() for line in data_lines]
#
#
# # Grid parameters
# N, M = 10, 10
# lat_min, lat_max =  48, 52
# lon_min, lon_max = -6, 10
#
#
# # Load data
# def load_data(filename):
#     trajectories = {}
#
#     with open(filename) as f:
#         for line in f:
#     # Parse trajectory data
#     # Add to trajectories dict
#
#     return trajectories
#
#
# # Discretize into grid
# def create_grid(N, M, lat_min, lat_max, lon_min, lon_max):
#     grid = np.zeros((N, M))
#
#     return grid
#
#
# # Preprocess trajectories
# def preprocess(trajectories, grid):
#     for traj in trajectories:
#
#         grid_traj = []
#
#         for lat, lon in traj:
#             # Map to grid cell
#             cell_x = ...
#             cell_y = ...
#             cell = (cell_x, cell_y)
#
#             grid_traj.append(cell)
#
#         trajectories[traj] = grid_traj
#
#     return trajectories
#
#
# # Extract sequences
# def extract_sequences(trajectories, L):
#     sequences = []
#
#     for traj in trajectories:
#
#         for i in range(len(traj) - L + 1):
#             seq = traj[i:i + L]
#             sequences.append(seq)
#
#     return sequences
#
#
# # Build HMM model
# def build_HMM(unique_seqs, sequences):
#     # Learn probabilities
#     start_probs = ...
#     trans_probs = ...
#     emission_probs = ...
#
#     # Create model
#     model = HMM(len(unique_seqs), start_probs, trans_probs, emission_probs)
#
#     return model
#
#
# # Predict next cell
# def predict_next_cell(seq, model):
#     v, _ = viterbi(seq, model)
#
#     obs_probs = ...  # weighted emissions
#
#     return np.argmax(obs_probs)
#
#
# # Preprocess trajectories
# trajectories = {}
# for timestamp, icao24, lat, lon in data:
#     if icao24 not in trajectories:
#         trajectories[icao24] = []
#
#     # Map GPS to grid cell
#     cell_x = int(N * (lon - lon_min) / (lon_max - lon_min))
#     cell_y = int(M * (lat - lat_min) / (lat_max - lat_min))
#     cell = (cell_x, cell_y)
#
#     trajectories[icao24].append(cell)
#
# # Segment into sequences
# L = 3  # sequence length
# for icao24 in trajectories:
#     sequences = []
#     for i in range(len(trajectories[icao24]) - L + 1):
#         seq = trajectories[icao24][i:i + L]
#         sequences.append(seq)
#     trajectories[icao24] = sequences
#
#
# # Get all unique sequences
# all_sequences = []
# for icao24 in trajectories:
#   all_sequences.extend(trajectories[icao24])
# all_sequences = list(set(all_sequences))
#
# # Label encoder to assign id to each unique sequence
# label_encoder = LabelEncoder()
# label_encoder.fit(all_sequences)
#
# # Number of hidden states
# N = len(label_encoder.classes_)
#


#
# # Build model
# model = HMM(N, start_probs, trans_probs, emission_probs)


import numpy as np
from hmmlearn import hmm

N_Cell, M_Cell = 8, 8
lat_min, lat_max = 48, 52
lon_min, lon_max = 6, 10
L = 10  # sequence length


def load_data(filename):
    data_lines = open(filename, 'r').readlines()
    data = [line.split() for line in data_lines]
    return data


def preprocess_trajectories(data):
    trajectories = {}
    for timestamp, icao24, lat, lon in data:
        if icao24 not in trajectories:
            trajectories[icao24] = []

        # Map GPS to grid cell
        cell_x = int(N_Cell * (float(lon) - lon_min) / (lon_max - lon_min))
        cell_y = int(M_Cell * (float(lat) - lat_min) / (lat_max - lat_min))
        cell = (cell_x, cell_y)

        trajectories[icao24].append(cell)
    return trajectories


def segment_into_sequences(trajectories):
    for icao24 in trajectories:
        sequences = []
        for i in range(len(trajectories[icao24]) - L + 1):
            seq = trajectories[icao24][i:i + L]
            sequences.append(seq)
        trajectories[icao24] = sequences
    return trajectories


def get_sequence_to_id_mapping(trajectories):
    all_sequences = []
    for icao24 in trajectories:
        all_sequences.extend(trajectories[icao24])
    unique_sequences = list(set(tuple(seq) for seq in all_sequences))
    sequence_to_id = {seq: i for i, seq in enumerate(unique_sequences)}
    return sequence_to_id


def replace_sequences_with_ids(trajectories, sequence_to_id):
    for icao24 in trajectories:
        trajectories[icao24] = [sequence_to_id[tuple(seq)] for seq in trajectories[icao24]]
    return trajectories


def build_model(trajectories):
    sequence_to_id = get_sequence_to_id_mapping(trajectories)
    trajectories = replace_sequences_with_ids(trajectories, sequence_to_id)

    # Number of hidden states
    N: int = len(sequence_to_id)

    # Emission probabilities - P(cell | sequence)
    emission_probs = np.zeros((N, N_Cell * M_Cell))  # Shape is (no. of unique sequences, 62 * 62)
    for seq, seq_id in sequence_to_id.items():
        counts = np.zeros(N_Cell * M_Cell)
        for cell in seq:
            cell_id = cell[0] * M_Cell + cell[1]  # Convert the cell coordinates to a single index
            counts[cell_id] += 1
        emission_probs[seq_id, :] = counts / sum(counts)

    # Transition probabilities - P(seq_i -> seq_j)
    trans_probs = np.zeros((N, N))
    for icao24 in trajectories:
        seq_ids = trajectories[icao24]
        for i in range(len(seq_ids) - 1):
            id1 = seq_ids[i]
            id2 = seq_ids[i + 1]
            trans_probs[id1, id2] += 1

    trans_probs /= trans_probs.sum(axis=1, keepdims=True)

    # Start probabilities - P(seq_i is start of trajectory)
    start_probs = np.zeros(N)
    for icao24 in trajectories:
        seq_ids = trajectories[icao24]
        start_probs[seq_ids[0]] += 1

    start_probs /= start_probs.sum()

    # Build model
    model = hmm.CategoricalHMM(n_components=N)

    model.startprob_ = start_probs
    model.transmat_ = trans_probs
    model.emissionprob_ = emission_probs

    return model

def predict_next(obs_sequence, model):
    # Decode the sequence of states using Viterbi
    _, state_sequence = model.decode(obs_sequence)

    # Predict the next state using the transition matrix
    next_state = np.argmax(model.transmat_[state_sequence[-1]])

    # Predict the observation for the predicted state using the emission matrix
    next_observation = np.argmax(model.emissionprob_[next_state])

    # Concatenate the predicted observation with the original observation sequence
    new_obs_sequence = np.concatenate([obs_sequence, [next_observation]])

    return new_obs_sequence

# def viterbi(obs, model):
#     # Initialize v and ptr
#
#     N = model.n_components
#     v = np.zeros((N, len(obs)))
#     ptr = np.zeros((N, len(obs)), dtype=int)
#
#     # Initialize base cases
#     v[:, 0] = model.startprob_ * model.emissionprob_[:, obs[0]]
#     ptr[:, 0] = -1
#     print(v.shape , model.transmat_.shape , model.emissionprob_.shape)
#     # Recursive case
#     for t in range(1, len(obs)):
#         for s in range(N):
#             probs = v[:, t - 1] * model.transmat_[:, s] * model.emissionprob_[s, obs[t] - 1]
#             ptr[s, t] = np.argmax(probs)
#             v[s, t] = np.max(probs)
#
#     # Backtrack
#     states = np.zeros(len(obs), dtype=int)
#     states[-1] = np.argmax(v[:, -1])
#
#     for t in range(len(obs) - 2, -1, -1):
#         states[t] = ptr[states[t + 1], [t + 1]]
#
#
#
#     print(states)
#
#     return v, states

#
# def predict(prev_obs, model):
#     # Run Viterbi on observed sequence
#     v, _ = viterbi(prev_obs, model)
#
#     # Initialize next observation probabilities
#     obs_probs = np.zeros(N_Cell * M_Cell)
#
#     # # Sum emission probs from all ending states
#     # for s in range(model.n_components):
#     #     obs_probs += v[-1, s] * model.transmat_[s] * model.emissionprob_[s]
#
#     # Sum emission probs from all ending states
#     for s in range(model.n_components):
#         next_state_probs = model.transmat_[s]  # Probabilities of transitioning to each next state from state s
#         for next_s in range(model.n_components):
#             emission_probs = model.emissionprob_[next_s]  # Emission probabilities for next state next_s
#             obs_probs += v[-1, s] * next_state_probs[next_s] * emission_probs
#
#     # Return most likely next observation
#     return np.argmax(obs_probs)


# def update_prev_obs(prev_obs, predicted_obs):
#     # Remove the oldest observation and add the new predicted observation
#     new_obs = prev_obs[1:] + [predicted_obs]
#     return new_obs


# def evaluate_accuracy(trajectories, model):
#     total_correct = 0
#     total_predictions = 0
#
#     for icao24, trajectory in trajectories.items():
#         # Split the trajectory into observation sequence and ground truth
#         prev_obs = trajectory[:12]
#         true_next_obs = trajectory[12:20]
#
#         # Predict the next 8 observations using the trained model
#         predicted_points = []
#         for _ in range(8):
#             predicted_obs = predict(prev_obs, model)
#             predicted_points.append(predicted_obs)
#             prev_obs = update_prev_obs(prev_obs, predicted_obs)
#
#         # Compare the predicted points with the true next observations
#         correct_predictions = sum(p == t for p, t in zip(predicted_points, true_next_obs))
#         total_correct += correct_predictions
#         total_predictions += len(true_next_obs)
#
#     # Calculate accuracy
#     accuracy = total_correct / total_predictions
#     return accuracy


data = load_data('C:\\Users\\vikrant.singh\\Downloads\\Hidden Markov Model\\data_vikrant\\data10_16_nrw_18-19.txt')
trajectories = preprocess_trajectories(data)
trajectories = segment_into_sequences(trajectories)
model = build_model(trajectories)

#
# # Evaluate the model
# accuracy = evaluate_accuracy(trajectories, model)
# print("Accuracy:", accuracy)
