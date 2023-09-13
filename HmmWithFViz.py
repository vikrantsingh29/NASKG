import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm
from sklearn.model_selection import train_test_split
import math
import pandas as pd

# df = pd.read_fwf("C:\\Users\\vikrant.singh\\Downloads\\concatenated_text.txt")
df = pd.read_fwf("C:\\Users\\vikrant.singh\Desktop\data_vikrant_new\\filtered_trajectory_data.txt")

# df = pd.read_fwf("C:\\Users\\vikrant.singh\\Downloads\\Hidden Markov Model\\data_vikrant\\data10_16_nrw_8-9.txt")

df.columns = ["time", "icao24", "latlng"]

df2 = pd.concat([df['time'], df['icao24'], df['latlng'].str.split(' ', expand=True)], axis=1)
df2.columns = ["time", "icao24", "lat", "lng"]
df2['lat'] = df2['lat'].astype(float)
df2['lng'] = df2['lng'].astype(float)

d = dict(tuple(df2.groupby("icao24")))
grouped_trajectories = df2.groupby('icao24')

# Create a list to store the trajectories
trajectories = []

# Iterate over each group (trajectory) and extract lat-lng coordinates
for _, group in grouped_trajectories:
    trajectory = group[['lat', 'lng']].values
    trajectories.append(trajectory)


# trajectories = trajectories[200:300]
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


desired_length = 20

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

# Split data into training and test sets
train_data, test_data = train_test_split(new_trajectories, test_size=0.2, random_state=42)

# Concatenate the training data and keep track of the lengths of individual sequences
train_data_concat = np.concatenate(train_data)
lengths = [len(trajectory) for trajectory in train_data]

# Fit the model
model = hmm.GaussianHMM(n_components=250, covariance_type="tied", n_iter=400)
model.fit(train_data_concat, lengths)


# Prediction and evaluation
errors = []
euclidean_errors = []
last_point_errors = []
random_state = np.random.RandomState(15)



import folium
from branca.element import Template, MacroElement


def plot_trajectory(latitudes_actual, longitudes_actual, latitudes_pred, longitudes_pred, filename):
    # Create a map centered around the first coordinate
    m = folium.Map(location=[latitudes_actual[0], longitudes_actual[0]], zoom_start=13)

    # Add lines for the given, predicted, and actual trajectories
    folium.PolyLine([(lat, lon) for lat, lon in zip(latitudes_actual[:14], longitudes_actual[:14])], color="blue",
                    weight=2.5, opacity=1, label='Given').add_to(m)
    folium.PolyLine([(lat, lon) for lat, lon in
                     zip(latitudes_actual[13:14] + latitudes_pred, longitudes_actual[13:14] + longitudes_pred)],
                    color="red", weight=2.5, opacity=1, label='Predicted').add_to(m)
    folium.PolyLine([(lat, lon) for lat, lon in zip(latitudes_actual[13:], longitudes_actual[13:])], color="green",
                    weight=2.5, opacity=1, label='Actual').add_to(m)

    # Add a legend
    template = """
    {% macro html(this, kwargs) %}
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css">
      <script src="https://code.jquery.com/jquery-1.12.4.min.js"></script>
      <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js"></script>
    </head>
    <body>
    <div id="maplegend" class="maplegend" 
        style="position: absolute; z-index:9999; border:2px solid grey; background-color:rgba(255, 255, 255, 0.8);
         border-radius:6px; padding: 10px; font-size:14px; right: 20px; bottom: 20px;">

    <div class="legend-title">Legend</div>
    <div class="legend-scale">
      <ul class="legend-labels">
        <li><span style='color:blue;opacity:1;'></span>Given</li>
        <li><span style='color:red;opacity:1;'></span>Predicted</li>
        <li><span style='color:green;opacity:1;'></span>Actual</li>

      </ul>
    </div>
    </div>

    </body>
    </html>

    <style type='text/css'>
      .maplegend .legend-title {
        text-align: left;
        margin-bottom: 5px;
        font-weight: bold;
        font-size: 90%;
        }
      .maplegend .legend-scale ul {
        margin: 0;
        margin-bottom: 5px;
        padding: 0;
        float: left;
        list-style: none;
        }
      .maplegend .legend-scale ul li {
        font-size: 80%;
        list-style: none;
        margin-left: 0;
        line-height: 18px;
        margin-bottom: 2px;
        }
      .maplegend ul.legend-labels li span {
        display: block;
        float: left;
        height: 16px;
        width: 30px;
        margin-right: 5px;
        margin-left: 0;
        border: 1px solid #999;
        }
      .maplegend .legend-source {
        font-size: 80%;
        color: #777;
        clear: both;
        }
      .maplegend a {
        color: #777;
        }
    </style>
    {% endmacro %}"""

    macro = MacroElement()
    macro._template = Template(template)

    m.get_root().add_child(macro)

    # Save the map to an HTML file
    m.save(filename)

    # Display the map
    return m


for trajectory in test_data:
    # Start with the first 14 points
    trajectory_to_predict = trajectory[:14]

    latitudes_pred = []
    longitudes_pred = []
    distances_pred = []
    speeds_pred = []
    latitudes_actual = list(trajectory[:14, 0])
    longitudes_actual = list(trajectory[:14, 1])
    distances_actual = list(trajectory[:14, 2])
    speeds_actual = list(trajectory[:14, 3])

    # Generate the next 6 points
    for i in range(6):
        # Predict the sequence of hidden states for the past trajectory
        past_states = model.predict(trajectory_to_predict)

        # Get the last hidden state
        last_state = past_states[-1]

        # The next state would be the one that has the highest transition probability from the last state
        next_state = np.argmax(model.transmat_[last_state])

        # Generate an observation from this next state
        next_observation = model._generate_sample_from_state(next_state, random_state=random_state)

        # Add the generated observation to the trajectory
        trajectory_to_predict = np.concatenate([trajectory_to_predict, [next_observation]])

        # Extract the predicted latitude, longitude, distance, and speed
        lat_pred, lon_pred, distance_pred, speed_pred = next_observation

        # Extract the actual latitude, longitude, distance, and speed for the corresponding point
        lat_actual, lon_actual, distance_actual, speed_actual = trajectory[14 + i]

        # Calculate errors
        error_lat = abs(lat_pred - lat_actual)
        error_lon = abs(lon_pred - lon_actual)
        error_distance = abs(distance_pred - distance_actual)
        error_speed = abs(speed_pred - speed_actual)
        errors.append([error_lat, error_lon, error_distance, error_speed])

        # Calculate the Euclidean distance error and append it to the list
        euclidean_error = np.sqrt((lat_pred - lat_actual) ** 2 + (lon_pred - lon_actual) ** 2)
        euclidean_errors.append(euclidean_error)

        # Store values for plotting
        latitudes_pred.append(lat_pred)
        longitudes_pred.append(lon_pred)
        distances_pred.append(distance_pred)
        speeds_pred.append(speed_pred)
        latitudes_actual.append(lat_actual)
        longitudes_actual.append(lon_actual)
        distances_actual.append(distance_actual)
        speeds_actual.append(speed_actual)

    # Calculate the error of the last point and append it to the list
    last_point_error = np.sqrt(
        (latitudes_pred[-1] - latitudes_actual[-1]) ** 2 + (longitudes_pred[-1] - longitudes_actual[-1]) ** 2)
    # print("Last point ,Predicted , Actual ", latitudes_pred[-1], latitudes_actual[-1], longitudes_pred[-1],
    #       longitudes_actual[-1])
    last_point_errors.append(last_point_error)

    plot_trajectory(latitudes_actual, longitudes_actual, latitudes_pred, longitudes_pred, f'trajectory_map_{i}.html')
    # # Plotting
    # plt.figure(figsize=(10, 5))
    #
    # # Plot latitudes and longitudes
    # plt.subplot(1, 2, 1)
    # plt.plot(longitudes_actual[:14], latitudes_actual[:14], 'bo-', label='Given')
    # plt.plot(longitudes_actual[13:14] + longitudes_pred, latitudes_actual[13:14] + latitudes_pred, 'ro-',
    #          label='Predicted')
    # plt.plot(longitudes_actual[13:], latitudes_actual[13:], 'go-', label='Actual',)
    # plt.xlabel('Longitude')
    # plt.ylabel('Latitude')
    # plt.legend()
    # plt.title('Latitude and Longitude')
    #
    # # Plot distances and speeds
    # plt.subplot(1, 2, 2)
    # plt.plot(distances_actual[:14], speeds_actual[:14], 'bo-', label='Given')
    # plt.plot(distances_actual[13:], speeds_actual[13:], 'go-', label='Actual')
    # plt.plot(distances_actual[13:14] + distances_pred, speeds_actual[13:14] + speeds_pred, 'ro-', label='Predicted')
    # plt.xlabel('Distance')
    # plt.ylabel('Speed')
    # plt.legend()
    # plt.title('Distance and Speed')
    #
    # plt.tight_layout()
    # plt.show()

# Calculate average Euclidean error
average_euclidean_error = np.mean(euclidean_errors)
print(f"Average Euclidean Error: {average_euclidean_error}")

# Calculate average last point error
average_last_point_error = np.mean(last_point_errors)
print(f"Average Last Point Error: {average_last_point_error}")
