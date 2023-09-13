import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm
from sklearn.model_selection import train_test_split
import math
import pandas as pd

df = pd.read_fwf("C:\\Users\\vikrant.singh\\Downloads\\concatenated_text.txt")

df.columns = ["time", "icao24", "latlng"]

df2 = pd.concat([df['time'], df['icao24'], df['latlng'].str.split(' ', expand=True)], axis=1)
df2.columns = ["time", "icao24", "lat", "lng"]
df2['lat'] = df2['lat'].astype(float)
df2['lng'] = df2['lng'].astype(float)

d = dict(tuple(df2.groupby("icao24")))
grouped_trajectories = df2.groupby('icao24')

# Create a list to store the trajectories
trajectories = []
print(len(trajectories))
# Iterate over each group (trajectory) and extract lat-lng coordinates
for _, group in grouped_trajectories:
    trajectory = group[['lat', 'lng']].values
    trajectories.append(trajectory)

trajectories = trajectories[200:300]
# Function to calculate distance between two points in meters using Haversine formula
def haversine(lat1, lon1, lat2, lon2):
    R = 6371e3  # Radius of the Earth in meters
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


desired_length = 18

# trajectories = trajectories[100:200]
trajectories_19elementsEach = [trajectory[:desired_length] for trajectory in trajectories if
                               len(trajectory) >= desired_length]
data = trajectories_19elementsEach

new_trajectories = []

# Constants
R = 6371e3  # Radius of the Earth in meters
time_interval = 120  # Time interval in seconds

for trajectory in data:
    new_trajectory = []
    prev_lat, prev_lon = None, None
    for i, (lat, lon) in enumerate(trajectory):
        if i == 0:
            # For the first point, distance and speed are 0
            distance = 0
            speed = 0
        else:
            # Calculate distance using Haversine formula
            phi1 = math.radians(prev_lat)
            phi2 = math.radians(lat)
            delta_phi = math.radians(lat - prev_lat)
            delta_lambda = math.radians(lon - prev_lon)
            a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            distance = R * c

            # Calculate speed
            speed = distance / time_interval

        # Append features to new_trajectory
        new_trajectory.append([lat, lon, distance, speed])

        # Update previous latitude and longitude
        prev_lat, prev_lon = lat, lon

    # Append the new_trajectory to new_trajectories
    new_trajectories.append(np.array(new_trajectory))

print(new_trajectories[0])

# Split data into training and test sets
train_data, test_data = train_test_split(new_trajectories, test_size=0.2, random_state=42)

# Concatenate the training data and keep track of the lengths of individual sequences
train_data_concat = np.concatenate(train_data)
lengths = [len(trajectory) for trajectory in train_data]

# Fit the model
model = hmm.GMMHMM(n_components=10, covariance_type="diag", n_iter=1000)
model.fit(train_data_concat, lengths)

# Prediction and evaluation
errors = []
latitudes_pred = []
longitudes_pred = []
distances_pred = []
speeds_pred = []
latitudes_actual = []
longitudes_actual = []
distances_actual = []
speeds_actual = []

random_state = np.random.RandomState(42)

for trajectory in test_data:
    hidden_state = model.predict(trajectory)[-1]
    next_observation = model._generate_sample_from_state(hidden_state, random_state=random_state)
    log_likelihood = model._compute_log_likelihood(trajectory)
    model_score = model.score(trajectory[:17])
    print("log likelihood =", model_score)

    # Extract the predicted latitude, longitude, distance, and speed
    lat_pred, lon_pred, distance_pred, speed_pred = next_observation

    # Extract the actual latitude, longitude, distance, and speed
    lat_actual, lon_actual, distance_actual, speed_actual = trajectory[-1]

    # Calculate errors
    error_lat = abs(lat_pred - lat_actual)
    error_lon = abs(lon_pred - lon_actual)
    error_distance = abs(distance_pred - distance_actual)
    error_speed = abs(speed_pred - speed_actual)
    errors.append([error_lat, error_lon, error_distance, error_speed])

    # Store values for plotting
    latitudes_pred.append(lat_pred)
    longitudes_pred.append(lon_pred)
    distances_pred.append(distance_pred)
    speeds_pred.append(speed_pred)
    latitudes_actual.append(lat_actual)
    longitudes_actual.append(lon_actual)
    distances_actual.append(distance_actual)
    speeds_actual.append(speed_actual)

# Plotting
plt.figure(figsize=(10, 5))

# Plot latitudes and longitudes
plt.subplot(1, 2, 1)
plt.scatter(longitudes_actual, latitudes_actual, c='green', label='Actual')
plt.scatter(longitudes_pred, latitudes_pred, c='red', label='Predicted')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.title('Latitude and Longitude')

# Plot distances and speeds
plt.subplot(1, 2, 2)
plt.scatter(distances_actual, speeds_actual, c='green', label='Actual')
plt.scatter(distances_pred, speeds_pred, c='red', label='Predicted')
plt.xlabel('Distance')
plt.ylabel('Speed')
plt.legend()
plt.title('Distance and Speed')

plt.tight_layout()
plt.show()

# Calculate average errors
average_errors = np.mean(errors, axis=0)
print(f"Average Error in Latitude: {average_errors[0]}")
print(f"Average Error in Longitude: {average_errors[1]}")
print(f"Average Error in Distance: {average_errors[2]}")
print(f"Average Error in Speed: {average_errors[3]}")
