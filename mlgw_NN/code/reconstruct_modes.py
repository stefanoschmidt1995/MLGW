import sys
import os
sys.path.append("/home/tim.grimbergen/my_code_v2")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from sklearn.metrics import mean_squared_error
from GW_generator_NN import GW_generator_NN
from GW_helper import generate_mode, compute_optimal_mismatch
import matplotlib.pyplot as plt
import time
import tensorflow as tf
tf.compat.v1.disable_eager_execution() #to speed up WF predictions (~4x speed up!)

modes = ["21","22","33","44","55"]
model_loc = "/home/tim.grimbergen/full_models_test/test_full/"
save_loc = "/home/tim.grimbergen/modes_test/testBIG/"
to_save = True
time_batch = True

gen_NN = GW_generator_NN(folder = model_loc)

N = 4000
batch_size = 100
f_min = 15

real_param_list = np.zeros((N,7))
param_list = np.zeros((N,4))
time_comp = np.zeros((N,2))
start_times = np.zeros(N)
F = np.zeros((len(modes),N))

if to_save:
	total_time = 0
	long_time_grid = np.array([1000000])
	short_time_grid = np.array([-1000000])

for i in range(N):
	q = np.random.uniform(1,10) #model trained for q \in (1,10), s1,s2\in (-0.9, 0.9)
	s1 = np.random.uniform(-0.9,0.9)
	s2 = np.random.uniform(-0.9,0.9)
	
	M = 8 * ( q / (1+q)**2)**(-3/8)
	param_list[i] = np.array([q,s1,s2,M]) #separate modes dont need iota and phi
	params = [M*q/(1+q),M/(1+q),s1,s2,1,0,0] #m1, m2, s1, s2, d, iota, phi_0
	theta = np.array(params) #D = 7 : [m1, m2, spin1_z , spin2_z, D_L, inclination, phi_0]
	real_param_list[i] = theta
	t = time.time()
	times, mode_dict = generate_mode(params[0],params[1], s1=params[2],s2 = params[3],d=params[4], t_coal = None,                                           t_step = 5e-5, f_min = f_min, t_min = None,
											  verbose = False, approx = "IMRPhenomTPHM")
	time_comp[i,0] = time.time()-t
	
	start_times[i]=times[0]
	
	if to_save:
		if times[0] < long_time_grid[0]:
			long_time_grid = times
		if times[0] > short_time_grid[0]:
			short_time_grid = times
	
	amp_prefactor = 4.7864188273360336e-20*20/1 #scale pred mode amplitude because only happens at the stage where modes are combined in gen_WF
	t = time.time()
	h_p_pred, h_c_pred = gen_NN.get_modes(theta, times, [(int(mode[0]),int(mode[1])) for mode in modes], "realimag")
	h_pred = amp_prefactor*(h_p_pred + h_c_pred*1j)
	time_comp[i,1] = time.time()-t
	
	for j,mode in enumerate(modes):
		h_pred = amp_prefactor*(h_p_pred[:,j] + h_c_pred[:,j]*1j)
		#print(h_pred)
		#h_pred = h_pred / np.max(np.abs(h_pred))
		h_real = mode_dict[(int(mode[0]),int(mode[1]))]
		#h_real = h_real / np.max(np.abs(h_real))

		F_opt, _ = compute_optimal_mismatch(h_pred, h_real)
		print("done with " + str(i+1)+" mode "+ mode + ", mismatch was " + str(F_opt[0]))
		F[j,i] = F_opt[0]
	
	if False:
		plt.figure(figsize=(15,8))
		plt.plot(times, h_real.real, label="real")
		plt.plot(times, h_pred.real, label="pred")
		plt.legend()
		plt.savefig("./test_figures_modes_3/"+mode+"/real_pred_WF"+str(q)+".png")
		plt.close()

for i,mode in enumerate(modes):
	plt.figure(figsize=(6,4))
	plt.scatter(param_list[:,0], np.log(F[i,:])/np.log(10))
	plt.title("mismatch median: " + str(np.median(F[i,:])))
	plt.savefig("./test_figures_modes_3/"+mode+"/mismatch_massratio"+str(N)+".png")
	plt.xlim(1,10)
	plt.xlabel("mass_ratio q")
	plt.ylabel("log(F)")
	plt.close()

	plt.figure(figsize=(6,4))
	plt.scatter(start_times, np.log(F[i,:])/np.log(10))
	plt.savefig("./test_figures_modes_3/"+mode+"/mismatch_starttime"+str(N)+".png")
	plt.xlabel("start_times")
	plt.ylabel("log(F)")
	plt.close()

	plt.figure(figsize=(8,5))
	plt.title("Speed up histogram all modes: true/model")
	plt.hist(time_comp[:,0] / time_comp[:,1], rwidth=1, bins = N)
	plt.xlabel("log(t_true / t_model)")
	plt.savefig("./test_figures_modes_3/"+mode+"/speed_up_histogram"+str(N)+".png")
	plt.close()

	plt.figure(figsize=(8,5))
	plt.title("starttime, massratio")
	plt.scatter(param_list[:,0], start_times)
	plt.xlabel("mass ratio")
	plt.ylabel("start times")
	plt.savefig("./test_figures_modes_3/"+mode+"/start_mass"+str(N)+".png")
	plt.close()

if time_batch:
	#times the time it takes to calculate WF in one batch (on longest time grid)
	gen_NN = GW_generator_NN(folder = model_loc, frozen=False, batch_size=batch_size)
	full_batch_time_short = 0
	full_batch_time_long = 0
	for j in range(N//batch_size):
		t = time.time()
		h_p_pred, h_c_pred = gen_NN.get_modes(real_param_list[j*batch_size:(j+1)*batch_size],
											  long_time_grid,
											  [(int(mode[0]),int(mode[1])) for mode in modes], "realimag")
		full_batch_time_long += time.time()-t
		t = time.time()
		h_p_pred, h_c_pred = gen_NN.get_modes(real_param_list[j*batch_size:(j+1)*batch_size],
											  short_time_grid,
											  [(int(mode[0]),int(mode[1])) for mode in modes], "realimag")
		full_batch_time_short += time.time()-t

save_dir_name = str(N)+"modes"
os.mkdir(save_loc+save_dir_name)

if to_save:
	np.savetxt(save_loc+save_dir_name + "/short_time_grid", short_time_grid)
	np.savetxt(save_loc+save_dir_name + "/long_time_grid", long_time_grid)
	np.savetxt(save_loc+save_dir_name + "/parameters", param_list)
	np.savetxt(save_loc+save_dir_name + "/time_comp", time_comp)
	np.savetxt(save_loc+save_dir_name + "/start_times", start_times)
	np.savetxt(save_loc+save_dir_name + "/mismatches", F)
	
	#ToDo: write an info-file containing parameters of data and total time of true WF generation.
	with open(save_loc+save_dir_name+"/info.txt", "w") as f:
		f.write("total time in seconds for true WF generation: "+str(np.sum(time_comp[:,0]))+'\n')
		f.write("total time in seconds for pred WF generation (1by1): "+str(np.sum(time_comp[:,1]))+'\n')
		f.write("total time in seconds for "+str(N//batch_size)+" batches (long grid): "+str(full_batch_time_long)+'\n')
		f.write("total time in seconds for "+str(N//batch_size)+" batches (short grid): "+str(full_batch_time_short)+'\n')
		f.write("f_min: "+str(f_min)+'\n')
		f.write("N: "+str(N)+'\n')
		f.write("Batch size: " + str(batch_size)+'\n')
		f.write("model_loc: " + model_loc+'\n')
