# src/vae.py
import tensorflow as tf
from tensorflow.keras import layers

class VAE(tf.keras.Model):
    def __init__(self, input_dim: int, latent_dim: int = 16, hidden_dim: int = 256):
        super().__init__()

        self.enc_dense = layers.Dense(hidden_dim, activation="relu")
        self.z_mean = layers.Dense(latent_dim)
        self.z_logvar = layers.Dense(latent_dim)

        self.dec_dense = layers.Dense(hidden_dim, activation="relu")
        self.dec_out = layers.Dense(input_dim, activation="linear")

    def encode(self, x):
        h = self.enc_dense(x)
        z_mean = self.z_mean(h)
        z_logvar = self.z_logvar(h)
        return z_mean, z_logvar

    def reparameterize(self, z_mean, z_logvar):
        eps = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_logvar) * eps

    def decode(self, z):
        h = self.dec_dense(z)
        return self.dec_out(h)

    def call(self, x, training=False):
        z_mean, z_logvar = self.encode(x)
        z = self.reparameterize(z_mean, z_logvar)
        x_hat = self.decode(z)
        return x_hat, z_mean, z_logvar

def vae_loss(x, x_hat, z_mean, z_logvar):
    recon = tf.reduce_mean(tf.reduce_sum(tf.square(x - x_hat), axis=1))
    kl = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar), axis=1))
    return recon + kl, recon, kl