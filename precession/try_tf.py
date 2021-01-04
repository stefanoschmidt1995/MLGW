import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt

#See inspiration here:
#https://www.tensorflow.org/guide/data_performance#the_dataset

class ArtificialDataset(tf.data.Dataset):
	def _generator(num_samples):
		# Opening the file
		time.sleep(0.03)
		while True:
			yield  np.random.uniform(0,10.,(1000,7))#, np.random.uniform(0,10.,(1000,2))

		#for sample_idx in range(num_samples):
		#	# Reading data (line, record) from the file
		#	time.sleep(0.015)

		#	yield (sample_idx,)

	def __new__(cls, num_samples=3):
		return tf.data.Dataset.from_generator(
			cls._generator,
			output_types=tf.dtypes.float64,
			output_shapes=(None,7),
			args=(num_samples,)
		)

class gen():
	def __init__(self,a):
		print(a)
		return

	def __call__(self):
		while True:
			yield np.random.uniform(0,10.,(1000,7)), np.random.uniform(0,10.,(1000,2))

a = gen(4)

dataset = tf.data.Dataset.from_generator(
     a,
     output_signature=(
         tf.TensorSpec(shape=(None,7), dtype=tf.float32),
         tf.TensorSpec(shape=(None,2), dtype=tf.float32))
		).prefetch(tf.data.experimental.AUTOTUNE)


#a = gen()
#a = ArtificialDataset()

#for i in a:
#	print("CIAO")
#	print(type(i))

#for i, val in enumerate(dataset):
#	print(val[0],val[1])
#	if i == 100:
#		break



from precession_helper import *
ranges = np.array([(1.1,10.), (0.,1.), (0.,1.), (0., np.pi), (0., np.pi), (0., 2.*np.pi)])
dataset = angle_generator(20, 100, ranges = ranges, N_batch = 3, replace_step = 10)

tf_dataset = tf.data.Dataset.from_generator(
     dataset,
     output_signature = tf.TensorSpec(shape=(None,9), dtype=tf.float32)
		).prefetch(tf.data.experimental.AUTOTUNE)

for i, val in enumerate(tf_dataset):
	#print(val[0:100,2])
	#print(val[0:100,8])
	print(i)
	#plt.plot(val[0:100,0], val[0:100,7],'o', ms = 1)
	#plt.show()

	if i == 100: break



















