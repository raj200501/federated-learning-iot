import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from utils.data_processing import load_data
from models.federated_learning import FederatedLearning, Client, fed_avg
from utils.evaluation import evaluate_model

# Load data
num_clients = 5
(train_data, train_labels), (test_data, test_labels) = load_data()

# Build global model
global_model = tf.keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(train_data.shape[1],)),
    layers.Dense(1, activation='sigmoid')
])
global_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Initialize clients
clients = []
for i in range(num_clients):
    client_data = train_data[i::num_clients]
    client_labels = train_labels[i::num_clients]
