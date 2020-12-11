#!/home/stefano.schmidt/ML_ODE/env/bin/python3
#/home/stefano.schmidt/dev/bin/python3.6
"""
First attempt to solve a ODE with ML.
The solution is written as a parametric function of time and initial data and by an optimization problem, it is computed the best matching value for the hyperparameters.
"""
#You can go to Caltech with: stefano.schmidt@ldas-pcdev11.ligo.caltech.edu
#You can go to Nikhef server with: stefano.schmidt@stro.nikhef.nl

#For how to build a model: https://keras.io/api/models/model/
#For how to fit a model in a custom way: https://keras.io/api/optimizers/

#For gradients w.r.t. inputs: https://stackoverflow.com/questions/53649837/how-to-compute-loss-gradient-w-r-t-to-model-inputs-in-a-keras-model
#Or probably simpler
#	grads = K.gradients(model.output, model.input)
#as in https://stackoverflow.com/questions/49312989/keras-calculating-derivatives-of-model-output-wrt-input-returns-none

#For how to implement a training procedure with L-BFGS method, you can see this: https://gist.github.com/piyueh/712ec7d4540489aad2dcfb80f9a54993 (function function_factory() )
#and the tf reference page: https://www.tensorflow.org/probability/api_docs/python/tfp/optimizer/lbfgs_minimize

try:
	import silence_tensorflow.auto #awsome!!!! :)
except:
	pass

import cProfile
#import re
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import os
os.environ["KMP_WARNINGS"] = "FALSE" 


#plotting function
def plot_solution(model, N_sol, seed, folder = ".", show = False):
	np.random.seed(seed)
	X = np.random.uniform(-1,1,(N_sol,7))
	X[:,1:4] = [1.,0,1.]

	times = np.linspace(0.,10.,200)
	X_t = np.zeros((N_sol, times.shape[0],3))
	X_t_rec = np.zeros((N_sol, times.shape[0],3))


	for i in range(N_sol):
		X_t_rec[i,:,:] = model.ODE_solution(times, X[i,1:4], X[i,4:])
		X_t[i,:,:] = scipy.integrate.odeint(model.ODE_derivative_np, np.array(X[i,1:4]), times, args = (np.array(X[i,4:]),), tfirst = True)

	plt.figure()
	for i in range(N_sol):
		plt.plot(X_t[i,:,0],X_t[i,:,1], c = 'r')
		plt.plot(X_t_rec[i,:,0],X_t_rec[i,:,1], c = 'b')
	plt.xlabel(r"$L_x$")
	plt.ylabel(r"$L_y$")

	plt.savefig(folder+"/Lxy.pdf", transparent =True)

	plt.figure()
	for i in range(N_sol):
		plt.plot(times,X_t[i,:,0],  c = 'r')
		plt.plot(times,X_t_rec[i,:,0], c = 'b')
	plt.xlabel(r"$t$")
	plt.ylabel(r"$L_x$")

	plt.savefig(folder+"/Lx.pdf", transparent =True)

	plt.figure()
	for i in range(N_sol):
		plt.plot(times,X_t[i,:,1],  c = 'r')
		plt.plot(times,X_t_rec[i,:,1], c = 'b')
	plt.xlabel(r"$t$")
	plt.ylabel(r"$L_y$")

	plt.savefig(folder+"/Ly.pdf", transparent =True)


	plt.figure()
	for i in range(N_sol):
		plt.plot(times,X_t[i,:,2],  c = 'r')
		plt.plot(times,X_t_rec[i,:,2], c = 'b')
	plt.xlabel(r"$t$")
	plt.ylabel(r"$L_z$")

	plt.savefig(folder+"/Lz.pdf", transparent =True)

	if show:
		plt.show()
	else:
		plt.close('all')


#defining a Model
class MyModel(tf.keras.Model):

	def __init__(self):
		super(MyModel, self).__init__()
		self.history = []
		self.metric = []
		self.epoch = 0

		self._list = []

		#self._list.append(tf.keras.layers.Dense(128, activation=tf.nn.sigmoid) )
		#self._list.append(tf.keras.layers.Dense(128, activation=tf.nn.sigmoid) )
		#self._list.append(tf.keras.layers.Dense(128, activation=tf.nn.sigmoid) )
		self._list.append(tf.keras.layers.Dense(128*4, activation=tf.nn.sigmoid) )
		self._list.append(tf.keras.layers.Dense(128*2, activation=tf.nn.sigmoid) )
		self._list.append(tf.keras.layers.Dense(128, activation=tf.nn.sigmoid) )
		self._list.append(tf.keras.layers.Dense(128, activation=tf.nn.sigmoid) )
		self._list.append(tf.keras.layers.Dense(3, activation=tf.keras.activations.linear))

		#self.call(tf.ones((1000, 4))) #This calls build() 
		self.optimizer = tf.keras.optimizers.Adam(learning_rate = 5e-4)
		self.build(input_shape = (None, 7)) #This is required to specify the input shape of the model and to state which are the trainable paramters

	def call(self, inputs):
		"Inputs: [t, X_0 (3,), Omega (3,)]"
		output = inputs
		for l in self._list:
			output = l(output)
		return output #(N,3)

	def get_solution(self, inputs):
		"Inputs are (N,7)"
		return inputs[:,1:4] + tf.transpose(tf.math.multiply( tf.transpose(self.call(inputs)), 1-tf.math.exp(-inputs[:,0])) ) #(N,3)

	def __ok_inputs(self, inputs):
		if not isinstance(inputs, tf.Tensor):
			inputs = tf.convert_to_tensor(inputs, dtype=tf.float32) #(N,7)
			if inputs.ndim == 1:
				inputs = inputs[None,:]
		return inputs

	def ODE_derivative_np(self,t, X, Omega):
		return self.ODE_derivative(tf.convert_to_tensor(t,dtype = tf.float32), tf.convert_to_tensor(X,dtype = tf.float32), tf.convert_to_tensor(Omega,dtype = tf.float32)).numpy()

	def ODE_derivative(self, t, X, Omega):
		if len(X.shape) != len(Omega.shape) and len(X.shape) > 1:
			Omega = tf.repeat(Omega[None,:], X.shape[0], axis = 0)
		output = tf.linalg.cross(Omega, X) #(N,3)
		#output = Omega
		return output


	def ODE_solution(self,t, X_0, Omega):
		"Numpy interface for the solution of the ODE with ML. Accepts a list of times, the initial conditions (3,) and Omega (3,)."
		X_0 = np.array(X_0)
		Omega = np.array(Omega)
		assert Omega.shape == (3,)
		assert X_0.shape == (3,)
		X = np.repeat([[*X_0, *Omega]],len(t), axis = 0) #(T,3)
		X = np.concatenate([np.array(t)[:, None],X], axis = 1) #(T,4)
		X = self.__ok_inputs(X) #casting to tf
		res = self.get_solution(X) #(T,3)
		return res.numpy()
	
	def loss(self, X):
		"""
		Loss function: takes an array X (N,1+3+3) with values to test the model at. X[0,:] = [t, (x0)_0, (x0)_1, (x0)_2, (Omega)_0, (Omega)_1, (Omega)_2]
		Input should be tensorflow only.
		"""
		Omega = X[:,4:]
		with tf.GradientTape() as g:
			g.watch(X)
			out = self.get_solution(X)
		
		grad = g.batch_jacobian(out, X)[:,:,0] #d/dt #(N,3)
		F = self.ODE_derivative(None, out, Omega)

			#loss can be multiplied by exp(-alpha*t) for "regularization"
		loss = tf.math.square(grad - F) #(N,3)
		loss = tf.transpose(tf.math.multiply(tf.transpose(loss), tf.math.exp(-1.*X[:,0]))) #(N,3)
		loss = tf.reduce_sum(loss, axis = 1) /X.shape[1] #(N,)
		return loss

	@tf.function#(jit_compile=True) #very useful for speed up
	def grad_update(self, X):
		"Input should be tensorflow only."
		with tf.GradientTape() as g:
			g.watch(self.trainable_weights)
			loss = tf.reduce_sum(self.loss(X))/X.shape[0]

		gradients = g.gradient(loss, self.trainable_weights)
		self.optimizer.apply_gradients(zip(gradients, model.trainable_weights))
		return loss

	def get_random_X(self, N_batch):
		t = tf.random.uniform((N_batch,1), minval=-0.1, maxval=10., dtype=tf.dtypes.float32)
		L_0x = tf.random.uniform((N_batch,1), minval=-1, maxval=1., dtype=tf.dtypes.float32)
		L_0y = tf.random.uniform((N_batch,1), minval=-1, maxval=1., dtype=tf.dtypes.float32)
		L_0z = tf.random.uniform((N_batch,1), minval=-1, maxval=1., dtype=tf.dtypes.float32)
		Omega_x = tf.random.uniform((N_batch,1), minval=-1., maxval=1., dtype=tf.dtypes.float32)
		Omega_y = tf.random.uniform((N_batch,1), minval=-1., maxval=1., dtype=tf.dtypes.float32)
		Omega_z = tf.random.uniform((N_batch,1), minval=-1., maxval=1., dtype=tf.dtypes.float32)

		return tf.concat([t, L_0x, L_0y, L_0z, Omega_x, Omega_y, Omega_z], axis = 1) #(N_batch, 7)
	
	def fit(self, N_epochs, model_folder = None, plot_function = None):
		N_batch = 20000
		save_step = 20000
		epoch_0 = self.epoch
		for i in range(N_epochs):
			X = self.get_random_X(N_batch)

			loss = self.grad_update(X)

			if i % (save_step/10) == 0: #saving history
				self.epoch = epoch_0 + i
				self.history.append((self.epoch, loss.numpy()))
				print(self.epoch, loss.numpy())
				if model_folder is not None:
					self.save_weights("{}/{}".format(model_folder, model_folder)) #overwriting the newest

			if i == 0: continue

			if model_folder is not None:
				if i%save_step ==0: #computing metric loss
					metric = 0.
					N_avg = 100 #batch size to compute the metric at
					X = self.get_random_X(N_avg)
					times = np.linspace(0.,10.,100)
					for j in range(N_avg):
							#solving ODE for solution
						X_t = scipy.integrate.odeint(self.ODE_derivative_np, X[j,1:4].numpy(), times, args = (X[j,4:],), tfirst = True)

						X_t_NN = self.ODE_solution(times, X[j, 1:4], X[j, 4:]) #(D,)
						plt.plot(times, X_t_NN)

						metric += np.mean(np.square(X_t -X_t_NN))

					self.metric.append((self.epoch, metric/N_avg))
					print("\tMetric: {} {}".format(self.metric[-1][0],self.metric[-1][1]))

				if i % save_step == 0:
					self.save_weights("{}/{}/{}".format(model_folder, str(self.epoch), model_folder)) #saving to arxiv
					if plot_function is not None:
						plot_function(self, 10, 0, "{}/{}".format(model_folder, str(self.epoch)))
					np.savetxt(model_folder+"/"+model_folder+".loss", np.array(self.history))
					np.savetxt(model_folder+"/"+model_folder+".metric", np.array(self.metric))
						
					
		return self.history

	def load_everything(self, path):
		"Loads model and try to read metric and loss"
		print(path)
		self.load_weights(path)
		try:
			self.history = np.loadtxt(path+".loss").tolist()
			self.epoch = int(self.history[-1][0])
		except:
			self.epoch = 0
			pass

		try:
			self.metric = np.loadtxt(path+".metric").tolist()
		except:
			pass

		return

##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################

what_to_do = 'load'

model_name = "model_omega_fit_GPU"
model_file = "{}/{}".format(model_name, model_name)

model = MyModel()
print(model.summary())

history = None
if what_to_do == 'load':
	model.load_everything(model_file)
elif what_to_do == 'fit':
	history = model.fit(int(1e7), model_name, plot_solution)
	model.save_weights(model_file)
elif what_to_do == 'fitload':
	model.load_everything(model_file)
	history = model.fit(200000, model_name, plot_solution)
	model.save_weights(model_file)
else:
	print("Nothing meaningful was asked")
	quit()

#quit()

try:
	history = np.array(model.history)
	metric = np.array(model.metric)
	plt.figure()
	plt.plot(history[:,0], history[:,1], c = 'b')
	plt.plot(metric[:,0], metric[:,1], c = 'r')
	plt.ylabel('loss/metric')
	plt.yscale('log')
	plt.savefig(model_name+"/loss.pdf", transparent =True)
except:
	pass



#plt.show()

########### Testing



plot_solution(model, 10, 3, model_name, True)

quit()








