import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import time


class CAE_1D(tf.keras.Model):
	"""1D convolutional autoencoder for the Euler angles"""
	def __init__(self, latent_dim, input_dim):
		super(CAE_1D, self).__init__()
		self.latent_dim = latent_dim
		self.input_dim = input_dim
		self.encoder = tf.keras.Sequential(
				[
						tf.keras.layers.InputLayer(input_shape=(input_dim,1)),
						tf.keras.layers.Conv1D(
								filters=32, kernel_size=100, strides= (25,), activation='relu'),
						tf.keras.layers.Conv1D(
								filters=64, kernel_size=10, strides= (3,), activation='relu'),
						tf.keras.layers.Flatten(),
						tf.keras.layers.Dense(10*latent_dim, activation='relu'),
						# No activation
						tf.keras.layers.Dense(latent_dim),
				]
		)
		self.decoder = tf.keras.Sequential(
				[
						tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
						tf.keras.layers.Dense(units= 10*latent_dim, activation=tf.nn.relu),
						tf.keras.layers.Reshape(target_shape=(10*latent_dim, 1)),
						tf.keras.layers.Conv1DTranspose(
								filters=64, kernel_size=10, strides=2, padding='same',
								activation='relu'),
						tf.keras.layers.Conv1DTranspose(
								filters=32, kernel_size=100, strides=5, padding='same',
								activation='relu'),
						# No activation
						tf.keras.layers.Flatten(),
						tf.keras.layers.Dense(input_dim),
				]
		)
	def encode(self, x):
		return self.encoder(x)
	def decode(self, z):
		return self.decoder(z)
	def __call__(self, x, **kwargs):
		#TODO: keras gives a training argument here. maybe if it's false, you should make the evaluation faster by disabling gradients
		return self.decoder(self.encoder(x))
	
	
