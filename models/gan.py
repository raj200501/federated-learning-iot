import tensorflow as tf
from tensorflow.keras import layers

class GAN:
    def __init__(self, noise_dim, data_dim):
        self.noise_dim = noise_dim
        self.data_dim = data_dim
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.gan = self.build_gan()

    def build_generator(self):
        model = tf.keras.Sequential([
            layers.Dense(128, activation='relu', input_dim=self.noise_dim),
            layers.Dense(self.data_dim, activation='tanh')
        ])
        return model

    def build_discriminator(self):
        model = tf.keras.Sequential([
            layers.Dense(128, activation='relu', input_dim=self.data_dim),
            layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def build_gan(self):
        self.discriminator.trainable = False
        model = tf.keras.Sequential([self.generator, self.discriminator])
        model.compile(optimizer='adam', loss='binary_crossentropy')
        return model

    def train(self, data, epochs=10000, batch_size=32):
        half_batch = int(batch_size / 2)
        for epoch in range(epochs):
            noise = np.random.normal(0, 1, (half_batch, self.noise_dim))
            gen_data = self.generator.predict(noise)
            real_data = data[np.random.randint(0, data.shape[0], half_batch)]
            combined_data = np.vstack((gen_data, real_data))
            labels = np.vstack((np.zeros((half_batch, 1)), np.ones((half_batch, 1))))
            d_loss = self.discriminator.train_on_batch(combined_data, labels)
            noise = np.random.normal(0, 1, (batch_size, self.noise_dim))
            labels = np.ones((batch_size, 1))
            g_loss = self.gan.train_on_batch(noise, labels)
            if epoch % 1000 == 0:
                print(f"{epoch} [D loss: {d_loss[0]}] [G loss: {g_loss}]")

if __name__ == "__main__":
    data = pd.read_csv('data/sample_data.csv').drop('label', axis=1).values
    gan = GAN(noise_dim=100, data_dim=data.shape[1])
    gan.train(data, epochs=10000, batch_size=32)
