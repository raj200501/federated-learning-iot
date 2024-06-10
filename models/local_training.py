import tensorflow as tf
from tensorflow.keras import layers

def build_local_model(input_shape):
    model = tf.keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=input_shape),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_local_model(model, train_data, train_labels, epochs=5, batch_size=32):
    model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size)
    return model
