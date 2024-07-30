import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Example dataset (you need to replace this with your actual dataset)
data = [
    ["Red", "Champ1", "Champ2", "Champ3", "Champ4", "Champ5", "Champ6", "Champ7", "Champ8", "Champ9", "Champ10", "Win"],
    ["Blue", "Champ11", "Champ12", "Champ13", "Champ14", "Champ15", "Champ16", "Champ17", "Champ18", "Champ19", "Champ20", "Loss"],
    # Add more data here...
]

# Convert the data into a numpy array for easier manipulation
data = np.array(data)

# Separate the features (X) from the outcomes (y)
X = data[:, :11]  # Features
y = data[:, 11]   # Outcome

# Use LabelEncoder to convert string labels into integers
le = LabelEncoder()
X[:, 0] = le.fit_transform(X[:, 0])  # Convert 'Red' and 'Blue' to integers
for i in range(1, 11):  # Convert champion names to integers
    le = LabelEncoder()
    X[:, i] = le.fit_transform(X[:, i])

# Convert the outcome to a binary value (Win=1, Loss=0)
y = (y == 'Win').astype(int)

# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define different models to try
models = [
    {
        'name': 'Simple',
        'layers': [
            Dense(64, activation='relu', input_shape=(11,)),
            Dense(1, activation='sigmoid')
        ],
        'optimizer': 'adam',
        'loss': 'binary_crossentropy'
    },
    {
        'name': 'Deep',
        'layers': [
            Dense(128, activation='relu', input_shape=(11,)),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ],
        'optimizer': 'adam',
        'loss': 'binary_crossentropy'
    },
    {
        'name': 'Wide',
        'layers': [
            Dense(256, activation='relu', input_shape=(11,)),
            Dense(1, activation='sigmoid')
        ],
        'optimizer': 'adam',
        'loss': 'binary_crossentropy'
    },
    {
        'name': 'Dropout',
        'layers': [
            Dense(128, activation='relu', input_shape=(11,)),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ],
        'optimizer': 'adam',
        'loss': 'binary_crossentropy'
    },
    {
        'name': 'RMSProp',
        'layers': [
            Dense(64, activation='relu', input_shape=(11,)),
            Dense(1, activation='sigmoid')
        ],
        'optimizer': 'rmsprop',
        'loss': 'binary_crossentropy'
    },
    {
        'name': 'Mean Squared Error',
        'layers': [
            Dense(64, activation='relu', input_shape=(11,)),
            Dense(1, activation='sigmoid')
        ],
        'optimizer': 'adam',
        'loss': 'mean_squared_error'
    }
]

# Train and evaluate each model
for model in models:
    print(f"Training model: {model['name']}")
    
    # Create the model
    m = Sequential(model['layers'])
    
    # Compile the model
    m.compile(optimizer=model['optimizer'], loss=model['loss'], metrics=['accuracy'])
    
    # Train the model
    m.fit(X_train, y_train, epochs=10, batch_size=32, verbose=2)
    
    # Make predictions
    predictions = m.predict(X_test)
    predicted_class = (predictions > 0.5).astype(int)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, predicted_class)
    print(f"Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, predicted_class))
    print()
