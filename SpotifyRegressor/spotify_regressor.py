# Imports
import xgboost
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('SpotifyTop50.csv')
df = pd.DataFrame(df)

# Remove unnecessary columns from the dataset
df = df.drop(labels = ["TrackName"], axis = 1) # The title of the track will not assist the model in learning how to predict popularity of a song

# Map non numeric values
genre_map = list(set([i for i in df.Genre]))
artist_map = list(set([i for i in df.ArtistName]))

df.Genre = df.Genre.map({i : genre_map.index(i) for i in genre_map})
df.ArtistName = df.ArtistName.map({i : artist_map.index(i) for i in artist_map})

# Scale x values
scaler = StandardScaler()
for col in df.columns:
  if col != 'Popularity':
    df[col] = scaler.fit_transform(df[[col]])

# Initialize x and y lists
x = []
y = list(df.pop("Popularity"))

# Add dataset to x and y lists
for row in range(df.shape[0]):
  rows = []
  for point in range(len(df.loc[0])): # Loop through all columns
    rows.append(df.iloc[row][point])
  x.append(rows)

# Divide the x and y values into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 1)

# Create and train model
model = XGBRegressor(n_estimators = 5000, learning_rate = 0.001)
model.fit(x_train, y_train, early_stopping_rounds = 5, eval_set = [(x_test, y_test)], verbose = 1) # Predicts the popularity rating of music

# View mean squared error of the model
predictions = model.predict(x_test)
mse = mean_squared_error(predictions, y_test)
print("\nMean Squared Error (MSE):", mse)

# Prediction vs. actual value (change the index to view a different input and output set)
index = 0
prediction = model.predict([x_test[index]])[0]

print(f"Model's Prediction on a Sample Input: {prediction}")
print(f"Actual Label on the Same Input: {y_test[index]}\n")

# Calculate model's approximate deviation on test data
def dev_calc(x, y):
  deviation = []
  for val in range(len(x)): # Loop through test values and have model predict on those test values
    error_val = abs(model.predict([x[val]]) - y[val])[0] # Determine the difference between the model's predicted labels and actual labels
    deviation.append(float(error_val)) # Store difference values in a list for plotting
  average_deviation = sum(deviation) / len(deviation)
  return deviation, average_deviation

# Visualize deviation on test data
test_deviation, avg_test_deviation = dev_calc(x_test, y_test)
y_pos = np.arange(len(test_deviation))

plt.figure(figsize = (8, 6))
plt.bar(y_pos, test_deviation, align = 'center')
plt.ylabel('Deviation')
plt.xlabel('Input Index')
plt.title("Model's Deviation on Test Data")

plt.show()

# Visualize deviation on training data
train_deviation, avg_train_deviation = dev_calc(x_train, y_train)
y_pos = np.arange(len(train_deviation))

plt.figure(figsize = (8, 6))
plt.bar(y_pos, train_deviation, align = 'center')
plt.ylabel('Deviation')
plt.xlabel('Input Index')
plt.title("Model's Deviation on Train Data")

plt.show()

# View the average deviation of the model's prediction on each dataset
print(f"\nAverage Deviation on Test Data: {avg_test_deviation} \nAverage Deviation on Train Data: {avg_train_deviation}")
