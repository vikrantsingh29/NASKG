# import pandas as pd
#
# # Load the data into a DataFrame
# df = pd.read_fwf("C:\\Users\\vikrant.singh\\Downloads\\Hidden Markov Model\\data_vikrant\\data10_16_nrw_13-14.txt")
#
# # Labeling the columns
# df.columns = ['time', 'icao24code', 'lat_long']
#
#
# # Vectorizing the trajectory by splitting the 'lat_long' column
# def vectorize_dataframe(df):
#     # Split the combined 'lat_long' column
#     df[['lat', 'long']] = df['lat_long'].str.split(expand=True).astype(float)
#
#     # Drop the 'lat_long' column
#     df = df.drop('lat_long', axis=1)
#
#     # Vectorize lat and long values
#     reference_points = {}
#     for index, row in df.iterrows():
#         aircraft_id = row['icao24code']
#         if aircraft_id not in reference_points:
#             reference_points[aircraft_id] = (row['lat'], row['long'])
#         df.at[index, 'lat'] -= reference_points[aircraft_id][0]
#         df.at[index, 'long'] -= reference_points[aircraft_id][1]
#
#     return df
#
#
# # Modify df in-place
# df = vectorize_dataframe(df)
#
# print(df)


#####################################################################################


import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("C:\\Users\\vikrant.singh\\Downloads\\vectorized_filtered_results.txt",
                 sep=" ", header=None)

df.columns = ["time", "icao24", "lat", "lng"]

df2 = df
df2['lat'] = df2['lat'].astype(float)
df2['lng'] = df2['lng'].astype(float)

#
d = dict(tuple(df2.groupby("icao24")))
grouped_trajectories = df2.groupby('icao24')

# Create a list to store the trajectories
trajectories = []

# Iterate over each group (trajectory) and extract lat-lng coordinates
for _, group in grouped_trajectories:
    trajectory = group[['lat', 'lng']].values
    trajectories.append(trajectory)



desired_length = 20
#
# trajectories = trajectories[100:120]
trajectories_new = [trajectory[:desired_length] for trajectory in trajectories if
                    len(trajectory) >= desired_length]
data = trajectories_new

new_trajectories = []

#
# Split data into training and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
#
# Concatenate the training data and keep track of the lengths of individual sequences
train_data_concat = np.concatenate(train_data)
lengths = [len(trajectory) for trajectory in train_data]
#
# Fit the model
model = hmm.GaussianHMM(n_components=100, covariance_type="tied", n_iter=400)
model.fit(train_data_concat, lengths)
print(f"The model converged in {model.monitor_.iter} iterations.")
#
#
# Prediction and evaluation
errors = []
euclidean_errors = []
last_point_errors = []
random_state = np.random.RandomState(15)

for trajectory in train_data:
    # Start with the first 12 points
    trajectory_to_predict = trajectory[:15]

    latitudes_pred = []
    longitudes_pred = []
    latitudes_actual = list(trajectory[:15, 0])
    longitudes_actual = list(trajectory[:15, 1])

    # Generate the next 6 points
    for i in range(5):
        # Predict the sequence of hidden states for the past trajectory
        past_states = model.predict(trajectory_to_predict)

        # Get the last hidden state
        last_state = past_states[-1]

        # The next state would be the one that has the highest transition probability from the last state
        next_state = np.argmax(model.transmat_[last_state])

        # Generate an observation from this next state
        next_observation = model._generate_sample_from_state(next_state, random_state=random_state)

        # # Sample the next state based on transition probabilities
        # next_state = np.random.choice(np.arange(model.transmat_.shape[0]), p=model.transmat_[last_state])
        #
        # # Generate the next observation
        # next_observation = model._generate_sample_from_state(next_state, random_state=random_state)

        # Add the generated observation to the trajectory
        trajectory_to_predict = np.concatenate([trajectory_to_predict, [next_observation]])

        # Extract the predicted latitude, longitude, distance, and speed
        lat_pred, lon_pred = next_observation

        # Extract the actual latitude, longitude, distance, and speed for the corresponding point
        lat_actual, lon_actual = trajectory[15 + i]

        # Calculate errors
        error_lat = abs(lat_pred - lat_actual)
        error_lon = abs(lon_pred - lon_actual)
        errors.append([error_lat, error_lon])

        # Calculate the Euclidean distance error and append it to the list
        euclidean_error = np.sqrt((lat_pred - lat_actual) ** 2 + (lon_pred - lon_actual) ** 2)
        euclidean_errors.append(euclidean_error)

        # Store values for plotting
        latitudes_pred.append(lat_pred)
        longitudes_pred.append(lon_pred)
        latitudes_actual.append(lat_actual)
        longitudes_actual.append(lon_actual)

    # Calculate the error of the last point and append it to the list
    last_point_error = np.sqrt(
        (latitudes_pred[-1] - latitudes_actual[-1]) ** 2 + (longitudes_pred[-1] - longitudes_actual[-1]) ** 2)
    last_point_errors.append(last_point_error)

    # Plotting
    plt.figure(figsize=(10, 5))

    # Plot latitudes and longitudes
    # plt.subplot(1, 2, 1)
    # plt.plot(longitudes_actual[:15], latitudes_actual[:15], 'bo-', label='Given')
    plt.plot(longitudes_actual[14:15] + longitudes_pred, latitudes_actual[14:15] + latitudes_pred, 'ro-',
             label='Predicted')
    plt.plot(longitudes_actual[14:], latitudes_actual[14:], 'go-', label='Actual', )
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.title('Latitude and Longitude')

    plt.tight_layout()
    plt.show()

# Calculate average Euclidean error
average_euclidean_error = np.mean(euclidean_errors)
print(f"Average Euclidean Error: {average_euclidean_error}")

# Calculate average last point error
average_last_point_error = np.mean(last_point_errors)
print(f"Average Last Point Error: {average_last_point_error}")
