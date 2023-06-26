###################
#	File holding definition for class greedy_reduction which implements a greedy algorithm for reducing a function over a set of basis functions
###################

import numpy as np
from GW_helper import *
import copy
import matplotlib.pyplot as plt

class GA_reduction:
	"""
	This class implements a version of greedy algorithm for reducing a function as linear combination of a small number of basis function.
	It requires a set of functions to be initialized - those function forming the dictionary of the algorithm. Then given a test function it computes the projection coefficents over the dictionary function of the model.
	"""
	def __init__(self, x_grid, scalar_product, exp_list = None):
		"""
		Initialize the model with the specification of grid point at which every function is evaluated. A list of exponents can be given if one wants the basis functions to be in the form of phi = x**alpha.
		A scalar product among funcions must be specified. It must accept as arguments the two sets of functions (N,N_grid) returning a vector of scalar products (N,)
		Input:
			x_grid (N_grid,)	vector holding values of x points in the domain every function is evaluated at
			scalar_product		Function for the scalar product used for reduction: <(N,N_grid),(N,N_grid) > = (N,)
			exp_list			list of exponents for the default dictionary built upon power laws
		Output:
			None
		"""
		self.x_grid = copy.copy(x_grid)
		self.scalar_product = scalar_product
#		self.scalar_product = np.vectorize(np.frompyfunc(scalar_product, 2, 1))
		if exp_list is not None:
			self.basis_dict = np.power(np.outer(self.x_grid, np.ones((len(exp_list),))), exp_list).T #(D,N_grid)
			self.basis_dict = np.divide(self.basis_dict.T, np.sqrt(self.scalar_product(self.basis_dict, self.basis_dict))).T #to normalize
			self.dict_exist = True
		else:
			self.dict_exist = False
		return None

	def get_dictionary(self):
		"""
		Returns the dictionary of basis functions.
		Input:
		Output:
			basis_dict (D,N_grid)	dictionary of basis function for the model (if exist)
		"""
		if self.dict_exist:
			return self.basis_dict
		else:
			return None

	def set_dictionary(self, basis_dict):
		"""
		Set a custom dictionary.	
		Input:
			baisis_dict (D,N_grid)	set of D basis functions composing the dictionary. The basis functions must be evaluated at points in x_grid
		Output:
			None
		"""
		self.basis_dict = np.array(basis_dict)
		self.basis_dict = np.divide(self.basis_dict.T, np.sqrt(self.scalar_product(self.basis_dict, self.basis_dict))).T #to normalize (in case it hasn't been done before)
		#self.basis_dict = np.concatenate((self.basis_dict,-self.basis_dict), axis =0)
		self.dict_exist = True
		return None

	def reconstruct_function(self, coeff_matrix):
		"""
		Reconstruct the explicit values of functions encoded in coeff_matrix (N,D) with formula:
			f_rec[i,j] = coeff_matrix[i,k] basis_dict[k,j]
		D must be equal to the number of basis function in the model.
		Input:
			coeff_matrix (N,D) 	matrix holdinf low dimensional representation of the function in terms of basis functions in the dictionary
		Output:
			f_rec (N,N_grid)	reconstructed function
		"""
		if coeff_matrix.ndim == 1:
			coeff_matrix = np.reshape(coeff_matrix, (1,coeff_matrix.shape[0]))
		if not 	self.dict_exist:
			raise RuntimeError("No dictionary is given yet: can't reconstruct the function!")
		if coeff_matrix.shape[1] != self.basis_dict.shape[0]:
			raise TypeError("Invalid shape for coefficient matrix "+str(coeff_matrix.shape)+"! Shape (N,"+str(self.basis_dict.shape[0])+") required")
		return np.matmul(coeff_matrix,self.basis_dict) #(N,N_grid) = (N,D)*(D,N_grid)

	def get_red_coefficients(self, F, M_iter):
		"""
		Applies the greedy algorithm for computing the best projection coefficients of the given function F onto the basis functions in the dictionary. It stops after M_iter iterations
		Input:
			F (N,N_grid)	Function(s) to compute the reduction coeffiecients for
			M_iter			Number of iterations after which the GA shall be stopped
		Output:
			red_coeff (N,D)	Returns the projections coefficients of F over the dictionary that give the best reconstruction of F
		"""
		if F.ndim == 1:
			F = np.reshape(F, (1,F.shape[0]))
		if not 	self.dict_exist:
			raise RuntimeError("No dictionary is given yet: can't reduce the function!")
		if F.shape[1] != self.basis_dict.shape[1]:
			raise TypeError("Invalid shape for coefficient matrix "+str(F.shape)+"! Shape (N,"+str(self.basis_dict.shape[1])+") required")
		
		red_coeff = np.zeros((F.shape[0],self.basis_dict.shape[0]))

		for n_iter in range(M_iter):
			temp_coeff_set = np.zeros((F.shape[0],self.basis_dict.shape[0]))
			for i in range(self.basis_dict.shape[0]): #it's similar to outer... You can do better I'm sure
				temp_coeff_set[:,i] = self.scalar_product(F, self.basis_dict[i,:])
			ind_to_choose = np.argmax(np.abs(temp_coeff_set), axis = 1)
			#print(ind_to_choose)

				#set of all values for the indices chosen with argmax - to make predictions
			chosen_coeff_set = np.zeros((F.shape[0],self.basis_dict.shape[0]))
			for i in range(F.shape[0]): 
				chosen_coeff_set[i,ind_to_choose[i]] = temp_coeff_set[i,ind_to_choose[i]]

			#print(chosen_coeff_set[5,ind_to_choose[5]])
			#plt.figure(0)
			#plt.title("To fit")
			#plt.plot(self.x_grid, F[2,:], label = str(n_iter))
			#plt.plot(self.x_grid, self.basis_dict[ind_to_choose[2],:], label ="basis iter "+str(n_iter))
			#plt.legend()
			#plt.figure(1)
			#plt.title("Reconstruction attempt")
			#plt.plot(self.x_grid, self.reconstruct_function(chosen_coeff_set)[2,:], label = str(n_iter))
			#plt.legend()

			F = F - self.reconstruct_function(chosen_coeff_set)
			red_coeff = red_coeff + chosen_coeff_set

		#plt.show()
		return red_coeff















		










