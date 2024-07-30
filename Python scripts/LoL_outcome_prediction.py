import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Load the dataset
file_path = 'Data/2024_LoL_esports_match_data_from_OraclesElixir.csv'  # Replace with the actual file path
df = pd.read_csv(file_path)

# Filter necessary columns
columns_needed = ["gameid", "side", "position", "champion", "teamname", "result"]
df_filtered = df[columns_needed]

# Pivot the DataFrame to get players in their respective positions for both teams
df_pivot = df_filtered.pivot_table(index="gameid", columns=["side", "position"], values="champion", aggfunc='first')

# Flatten the multi-level columns
df_pivot.columns = ['_'.join(col).strip() for col in df_pivot.columns.values]

# Reset the index to have a columnar structure
df_pivot.reset_index(inplace=True)

# Merge the results to get the win/loss outcome for each team
results = df_filtered.drop_duplicates(subset=["gameid", "teamname"])[["gameid", "side", "teamname", "result"]]

# Get the result for the team on the blue side
blue_result = results[results['side'] == 'Blue'][["gameid", "result"]]
blue_result.rename(columns={"result": "Result"}, inplace=True)

# Merge the result back to the pivoted DataFrame
df_final = df_pivot.merge(blue_result, on="gameid")

# Rename columns to match the desired output format
df_final.columns = [
    'GameID', 
    'BotTeam1', 'JglTeam1', 'MidTeam1', 'SupTeam1', 'TopTeam1', 
    'BotTeam2', 'JglTeam2', 'MidTeam2', 'SupTeam2', 'TopTeam2', 
    'Result'
]

# Output the new DataFrame
print(df_final)

# Convert the data into a numpy array for easier manipulation
data = np.array(df_final)

# Separate the features (X) from the outcomes (y)
X = data[:, 1:11]  # Features
y = data[:, 11]   # Outcome
y = y.astype('float')
# One-hot encode the categorical variables
encoder = OneHotEncoder(sparse_output=False)
X_encoded = encoder.fit_transform(X)

# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Define the model
model = Sequential([
    Dense(32, input_dim=X_train.shape[1], activation='leaky_relu'),
    Dense(1, activation='sigmoid')  # Sigmoid activation for binary classification
])

# Compile the model
model.compile(optimizer='SGD', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')