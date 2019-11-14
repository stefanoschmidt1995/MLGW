###################
#	Some tries of reducing GW WF as linear combination of some basis functions with greedy algorith (GA)
###################

from GW_helper import *
import matplotlib.pyplot as plt
from ML_routines import *
from greedy_reduction import *

N_grid = 512
theta_vector, amp_dataset, ph_dataset, frequencies = load_dataset("./datasets/GW_dataset_full_range.gz", N_data =10, N_grid = N_grid, shuffle=False) #loading dataset

df = (np.max(frequencies)-np.min(frequencies)) / N_grid
ph_dataset = ph_dataset
#quit()
print("Loaded "+ str(theta_vector.shape[0])+" data with N_grid = "+str(amp_dataset.shape[1]))

exp_set = [1.,-5/3.,-2/3., -1/3., -1., 1/3., 2/3., 5/3]
exp_set = np.linspace(-30,30,50)

#scalar_product = lambda a,b: compute_scalar(a, np.ones(a.shape), b , np.ones(b.shape), df = df)
def scalar_product(a,b):
	return compute_scalar(a, np.zeros(a.shape), b , np.zeros(b.shape), df = df)


greedy_red = GA_reduction(frequencies, scalar_product, exp_set)
dic = np.concatenate((greedy_red.get_dictionary(), ph_dataset[0:2,:]), axis = 0)
greedy_red.set_dictionary(dic)
print(greedy_red.get_dictionary().shape)

red_coeff = greedy_red.get_red_coefficients(ph_dataset, 300)
rec_ph = greedy_red.reconstruct_function(red_coeff)

print(compute_mismatch(np.ones((N_grid,)), rec_ph, np.ones((N_grid,)), ph_dataset))

plt.figure(0)
plt.title("Phase with GA")
for i in range(2,3):
	plt.plot(frequencies, ph_dataset[i,:], label = 'true '+str(i))
	plt.plot(frequencies, rec_ph[i,:], label = 'reconstructed '+str(i))
plt.legend()
plt.show()

quit()








freq_basis = np.power(np.outer(frequencies, np.ones((len(exp_set),))), exp_set).T #saving basis functions
freq_basis = np.divide(freq_basis.T, np.sqrt(compute_scalar(freq_basis, np.ones((N_grid,)), freq_basis ,np.ones((N_grid,)), df = df))).T #to normalize

print(compute_scalar(freq_basis, np.ones((N_grid,)), freq_basis ,np.ones((N_grid,)), df = 1.)) #to normalize

print(freq_basis.shape)
#print(freq_basis)
coeff_set = np.zeros((ph_dataset.shape[0],freq_basis.shape[0]))

	#you do this ONLY if the base you're working on is orthonormal
for i in range(freq_basis.shape[0]):
	coeff_set[:,i] = compute_scalar(freq_basis[i,:], np.ones((N_grid,)), ph_dataset ,np.ones((N_grid,)), df = df)

	#greedy algorithm
f_it = ph_dataset
coeff_set = np.zeros((ph_dataset.shape[0],K))
for i in range(K):
	temp_coeff_set = np.zeros((ph_dataset.shape[0],freq_basis.shape[0]))
	for i in range(freq_basis.shape[0]):
		temp_coeff_set[:,i] = compute_scalar(freq_basis[i,:], np.ones((N_grid,)), f_it ,np.ones((N_grid,)), df = df)

rec_ph = np.matmul(coeff_set,freq_basis)







