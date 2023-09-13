import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm
from sklearn.model_selection import train_test_split
import pandas as pd

data_split = pd.read_csv("C:\\Users\\vikrant.singh\\Downloads\\vectorized_filtered_results.txt",
                         sep=" ", header=None)

# Rename the columns based on the actual structure of the data
data_split.columns = ['timestamp', 'icao24', 'latitude', 'longitude']

# Convert latitude and longitude to numeric values
data_split['latitude'] = pd.to_numeric(data_split['latitude'])
data_split['longitude'] = pd.to_numeric(data_split['longitude'])

# Group the data by aircraft ID
trajectories = data_split.groupby('icao24')[['latitude', 'longitude']]

################################# New Code #########################################

from shapely.geometry import LineString
from sklearn.metrics import silhouette_score
from hmmlearn import hmm
import numpy as np


def segment_trajectory(trajectory, tolerance=0.01):
    line = LineString(trajectory)
    simplified_line = line.simplify(tolerance)
    return list(simplified_line.coords)


segmented_trajectories = [segment_trajectory(trajectory) for _, trajectory in trajectories]

print(trajectories)
print("trajectories[10]")
print(segmented_trajectories)
print("segmented_trajectories[10]")

# Combine segmented trajectories and determine lengths for HMM training
train_data_concat = np.concatenate(segmented_trajectories)
lengths = [len(trajectory) for trajectory in segmented_trajectories]

best_score = -1
best_n_components = None
best_model = None
max_states = 10  # This is an example; you can adjust this value based on your needs

for n in range(2, max_states):
    model = hmm.GaussianHMM(n_components=n, covariance_type="tied", n_iter=100)
    model.fit(train_data_concat, lengths)

    predicted_states = model.predict(train_data_concat, lengths)
    score = silhouette_score(train_data_concat, predicted_states)

    if score > best_score:
        best_score = score
        best_n_components = n
        best_model = model

print("best_n_components", best_n_components)


def predict_trajectory(model, initial_trajectory):
    random_state = np.random.RandomState(25)
    states = model.predict(initial_trajectory)
    last_state = states[-1]

    next_states = [np.argmax(model.transmat_[last_state])]
    next_observations = [model._generate_sample_from_state(next_states[-1], random_state=random_state)]

    for _ in range(4):
        next_state = np.argmax(model.transmat_[next_states[-1]])
        next_states.append(next_state)
        next_observations.append(model._generate_sample_from_state(next_state, random_state=random_state))

    return np.concatenate([initial_trajectory, next_observations])


import numpy as np
import matplotlib.pyplot as plt


# 1. Loop through all trajectories and predict the next points.
predictions = []
for trajectory in segmented_trajectories:
    predicted_traj = predict_trajectory(best_model, trajectory)
    predictions.append(predicted_traj)

# 2. Compute the Euclidean error for the last point of each trajectory.
euclidean_errors = []
for true_traj, pred_traj in zip(segmented_trajectories, predictions):
    true_last_point = true_traj[-1]
    pred_last_point = pred_traj[-1]

    error = np.sqrt((true_last_point[0] - pred_last_point[0]) ** 2 + (true_last_point[1] - pred_last_point[1]) ** 2)
    euclidean_errors.append(error)

# 3. Calculate the average Euclidean error for the last point across all trajectories.
average_euclidean_error = np.mean(euclidean_errors)
print(f"Average Euclidean Error for the Last Point: {average_euclidean_error}")

# 4. Plot the given, predicted, and actual trajectories.
for true_traj, pred_traj in zip(segmented_trajectories, predictions):
    plt.figure(figsize=(10, 5))

    # Plot given trajectory
    # plt.plot([point[1] for point in true_traj[:15]], [point[0] for point in true_traj[:15]], 'bo-', label='Given')

    # Plot predicted points
    plt.plot([point[1] for point in pred_traj[:20]], [point[0] for point in pred_traj[:20]], 'ro-', label='Predicted')

    # Plot actual remaining points
    plt.plot([point[1] for point in true_traj[:20]], [point[0] for point in true_traj[:20]], 'go-', label='Actual')

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.title('Latitude and Longitude')
    plt.tight_layout()
    plt.show()

# from sklearn.metrics import mean_absolute_error, mean_squared_error
#
# # Assuming true_trajectory and predicted_trajectory are arrays with true and predicted values
# # You'll need to loop over your test set, predict the trajectories using the best_model,
# # and then compute the following metrics for each:
#
# mae = mean_absolute_error(true_trajectory, predicted_trajectory)
# rmse = np.sqrt(mean_squared_error(true_trajectory, predicted_trajectory))


# train_trajectories, test_trajectories = train_test_split(list(trajectories), test_size=0.2, random_state=42)

# # Combine all training trajectories into one dataset
# combined_train_data = np.vstack([trajectory.values for _, trajectory in train_trajectories])
#
# # Training a single HMM
# model = hmm.GaussianHMM(n_components=25, covariance_type="tied", n_iter=100)
# model.fit(combined_train_data)
#
# # Testing & Prediction
# predictions = {}
# for icao24, trajectory in train_trajectories:
#     # Estimate the states for the first 15 points
#     estimated_states = model.predict(trajectory.iloc[:15].values)
#
#     # Get the last estimated state as the starting point
#     current_state = estimated_states[-1]
#     next_states = [current_state]
#
#     # Sample the next 5 states based on the transition probabilities
#     for _ in range(4):  # We already have the initial state
#         current_state = np.random.choice(
#             range(model.n_components),
#             p=model.transmat_[current_state]
#         )
#         next_states.append(current_state)
#
#     # Predict the next 5 points based on the emission probabilities of the next states
#     predicted_points = [model.means_[state] for state in next_states]
#
#     # Store the predicted points
#     predictions[icao24] = predicted_points[-5:]
#
# # Evaluation
# errors = []
# for icao24, trajectory in train_trajectories:
#     actual_points = trajectory.iloc[15:].values
#     predicted_points = np.array(predictions[icao24])
#     error = np.linalg.norm(actual_points - predicted_points)
#     errors.append(error)
#
# mean_error = np.mean(errors)
# print(f"Mean error: {mean_error:.2f} km")
#
# import matplotlib.pyplot as plt
#
# # Loop through each test trajectory for visualization
# for icao24, trajectory in train_trajectories:
#     latitudes_actual = trajectory['latitude'].tolist()
#     longitudes_actual = trajectory['longitude'].tolist()
#
#     latitudes_pred = [point[0] for point in predictions[icao24]]
#     longitudes_pred = [point[1] for point in predictions[icao24]]
#
#     # Plotting
#     plt.figure(figsize=(10, 5))
#
#     # Plot latitudes and longitudes
#     # plt.plot(longitudes_actual[:15], latitudes_actual[:15], 'bo-', label='Given')
#     plt.plot(longitudes_actual[14:15] + longitudes_pred, latitudes_actual[14:15] + latitudes_pred, 'ro-',
#              label='Predicted')
#     plt.plot(longitudes_actual[14:], latitudes_actual[14:], 'go-', label='Actual')
#     plt.xlabel('Longitude')
#     plt.ylabel('Latitude')
#     plt.legend()
#     plt.title(f'Trajectory {icao24}')
#
#     plt.tight_layout()
#     plt.show()
