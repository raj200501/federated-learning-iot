import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from utils.data_processing import load_data
from utils.evaluation import evaluate_model

class FederatedLearning:
    def __init__(self, clients, global_model, aggregation_func):
        self.clients = clients
        self.global_model = global_model
        self.aggregation_func = aggregation_func

    def train(self, rounds, epochs, batch_size):
        for round in range(rounds):
            local_weights = []
            for client in self.clients:
                client.train(self.global_model, epochs, batch_size)
                local_weights.append(client.get_weights())
            new_weights = self.aggregation_func(local_weights)
            self.global_model.set_weights(new_weights)
            print(f"Round {round + 1} completed.")

    def evaluate(self, test_data, test_labels):
        evaluate_model(self.global_model, test_data, test_labels)

class Client:
    def __init__(self, model, data, labels):
        self.model = model
        self.data = data
        self.labels = labels

    def train(self, global_model, epochs, batch_size):
        self.model.set_weights(global_model.get_weights())
        self.model.fit(self.data, self.labels, epochs=epochs, batch_size=batch_size, verbose=0)

    def get_weights(self):
        return self.model.get_weights()

def fed_avg(local_weights):
    avg_weights = list()
    for weights_list_tuple in zip(*local_weights):
        avg_weights.append(
            [np.array(weights_).mean(axis=0) for weights_ in zip(*weights_list_tuple)])
    return avg_weights

if __name__ == "__main__":
    num_clients = 5
    (train_data, train_labels), (test_data, test_labels) = load_data()
    global_model = tf.keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(train_data.shape[1],)),
        layers.Dense(1, activation='sigmoid')
    ])
    global_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    clients = []
    for i in range(num_clients):
        client_data = train_data[i::num_clients]
        client_labels = train_labels[i::num_clients]
        client_model = tf.keras.models.clone_model(global_model)
        client_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        clients.append(Client(client_model, client_data, client_labels))

    fl = FederatedLearning(clients, global_model, fed_avg)
    fl.train(rounds=10, epochs=5, batch_size=32)
    fl.evaluate(test_data, test_labels)
