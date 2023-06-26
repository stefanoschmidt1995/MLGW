import tensorflow as tf
from GW_helper import *
import keras.backend as K

	#doesn't work at all!!!!

def np_mismatch_function(logreg_weights, PCA_weights, test_amp):
	"""
	Numpy version of mismatch function. To be wrapped by tensorflow.
	Input:
		logreg_weights	[max (K,), min (K,)]
		PCA_weights		[V (D,K), mu (D,), beta ()]
		test_amp 		(N,D)
	Output:
		loss(y_true, y_pred)	function of two np.array returning a np array with mismatches
	"""
	def loss(y_true, y_pred):
		y_true = np.array(y_true)
		y_pred = np.array(y_pred)
			#un-preprocessing weights
		max_y = np.repeat(logreg_weights[0], y_true.shape[0], axis=0)
		min_y = np.repeat(logreg_weights[1], y_true.shape[0], axis=0)
		y_true = np.multiply(y_true, np.abs(max_y-min_y)) + min_y
		y_pred = np.multiply(y_pred, np.abs(max_y-min_y)) + min_y
			#inverting PCA
		y_true = np.matmul(y_true, PCA_weights[0].T)
		y_true = (y_true+PCA_weights[1] )*PCA_weights[2]
		y_pred = np.matmul(y_pred, PCA_weights[0].T)
		y_pred = (y_pred+PCA_weights[1] )*PCA_weights[2]
			#computing mismatch
		F = compute_mismatch(test_amp, y_true, test_amp, y_pred)
		return np.ndarray(F.shape, buffer = F)
	return loss

def np_mse(y_true, y_pred):
	return np.sum(np.square(y_true-y_pred), axis = 1)

def mismatch_function(logreg_weights, PCA_weights, test_amp):
	"""
	Keras version of mismatch function.
	Input:
		logreg_weights	[max (K,), min (K,)]
		PCA_weights		[V (D,K), mu (D,), beta ()]
		test_amp 		(N,D)
	Output:
		loss(y_true, y_pred)	function of two tf tensors returning a tensor with mismatches
	"""
	def loss(y_true, y_pred):
		if not K.is_tensor(y_pred):
			y_pred = K.constant(y_pred)
		y_true = K.cast(y_true, y_pred.dtype)

			#initializing np tensors to tf tensors
		max_y = K.tf.Variable(K.cast_to_floatx(logreg_weights[0]))
		min_y = K.tf.Variable(K.cast_to_floatx(logreg_weights[1]))
		V_PCA = K.tf.Variable(K.cast_to_floatx(PCA_weights[0].T))
		mu = K.tf.Variable(K.cast_to_floatx(PCA_weights[1]))
		alpha = K.tf.Variable(K.cast_to_floatx(PCA_weights[2]))
		amp = K.tf.Variable(K.cast_to_floatx(test_amp[0,:]))

			#un pre-processing with logreg
		y_true = K.tf.multiply(y_true, K.abs(max_y-min_y)) + min_y #multiply does authomatically the right thing when (N,D)*(D,) !!!!!!
		y_pred = K.tf.multiply(y_pred, K.abs(max_y-min_y)) + min_y
			#inverting PCA
		y_true = K.dot(y_true, V_PCA)
		y_true = (y_true+mu)*alpha
		y_pred = K.dot(y_pred, V_PCA)
		y_pred = (y_pred+mu)*alpha
			#mismatch
		w_pred = K.tf.complex(K.tf.multiply(amp, K.cos(y_pred)), K.tf.multiply(amp, K.sin(y_pred)))
		w_true = K.tf.complex(K.tf.multiply(amp, K.cos(y_true)), K.tf.multiply(amp, K.sin(y_true)))
		overlap = K.tf.real(K.sum(np.multiply(K.tf.conj(w_pred),w_true)))
		norm = K.sqrt(K.tf.multiply(K.tf.real(K.sum(np.multiply(K.tf.conj(w_pred),w_pred))), K.tf.real(K.sum(np.multiply(K.tf.conj(w_true),w_true)))) )
		overlap = 1-K.tf.divide(overlap, norm)

		tf.debugging.check_numerics(overlap, message = "Nan?")

		return overlap
		#return K.mean(K.square(y_pred - y_true), axis=-1)
	return loss

################# tf loss function
def mismatch_function(logreg_weights, PCA_weights, test_amp):
	"""
	Keras version of mismatch function.
	Input:
		logreg_weights	[max (K,), min (K,)]
		PCA_weights		[V (D,K), mu (D,), beta ()]
		test_amp 		(N,D)
	Output:
		loss(y_true, y_pred)	function of two tf tensors returning a tensor with mismatches
	"""
	import tensorflow as tf
	import keras.backend as K

	def loss(y_true, y_pred):
		if not K.is_tensor(y_pred):
			y_pred = K.constant(y_pred)
		y_true = K.cast(y_true, y_pred.dtype)

			#initializing np tensors to tf tensors
		max_y = K.tf.Variable(K.cast_to_floatx(logreg_weights[0]))
		min_y = K.tf.Variable(K.cast_to_floatx(logreg_weights[1]))
		V_PCA = K.tf.Variable(K.cast_to_floatx(PCA_weights[0].T))
		mu = K.tf.Variable(K.cast_to_floatx(PCA_weights[1]))
		alpha = K.tf.Variable(K.cast_to_floatx(PCA_weights[2]))
		amp = K.tf.Variable(K.cast_to_floatx(test_amp[0,:]))

			#un pre-processing with logreg
		y_true = K.tf.multiply(y_true, K.abs(max_y-min_y)) + min_y #multiply does authomatically the right thing when (N,D)*(D,) !!!!!!
		y_pred = K.tf.multiply(y_pred, K.abs(max_y-min_y)) + min_y
			#inverting PCA
		y_true = K.dot(y_true, V_PCA)
		y_true = (y_true+mu)*alpha
		y_pred = K.dot(y_pred, V_PCA)
		y_pred = (y_pred+mu)*alpha
			#mismatch
		w_pred = K.tf.complex(K.tf.multiply(amp, K.cos(y_pred)), K.tf.multiply(amp, K.sin(y_pred)))
		w_true = K.tf.complex(K.tf.multiply(amp, K.cos(y_true)), K.tf.multiply(amp, K.sin(y_true)))
		overlap = K.tf.real(K.sum(np.multiply(K.tf.conj(w_pred),w_true)))
		norm = K.sqrt(K.tf.multiply(K.tf.real(K.sum(np.multiply(K.tf.conj(w_pred),w_pred))), K.tf.real(K.sum(np.multiply(K.tf.conj(w_true),w_true)))) )
		overlap = 1-K.tf.divide(overlap, norm)
		
		return overlap
		#return K.mean(K.square(y_pred - y_true), axis=-1)
	return loss


