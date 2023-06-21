import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(sys.path[0])),'MLGW-master','mlgw_v2')) #ugly way of adding mlgw_v2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pycbc.filter
from pycbc.types import timeseries
import time
from sklearn.metrics import mean_squared_error #not used but need for error resolve
import numpy as np
from GW_generator_NN import GW_generator_NN
from GW_helper import generate_waveform, compute_optimal_mismatch
import matplotlib.pyplot as plt
import tensorflow as tf
tf.compat.v1.disable_eager_execution() #to speed up WF predictions (~4x speed up!)

def get_random_antenna_patterns():
	"""
	Get random antenna patterns
	"""
	N  =1
	polarization = np.random.uniform(0, 2*np.pi, N)
	latitude = np.arcsin(np.random.uniform(-1.,1., N)) #FIXME: check this is right!
	longitude = np.random.uniform(-np.pi, np.pi, N)
	
	theta = np.pi/2 - np.asarray(latitude)
	
	F_p = - 0.5*(1 + np.cos(theta)**2)* np.cos(2*longitude)* np.cos(2*polarization)
	F_p -= np.cos(theta)* np.sin(2*longitude)* np.sin(2*polarization) 
	F_c = 0.5*(1 + np.cos(theta)**2)* np.cos(2*longitude)* np.sin(2*polarization)
	F_c -= np.cos(theta)* np.sin(2*longitude)* np.cos(2*polarization) 

	return F_p, F_c

model_loc = "/home/tim.grimbergen/full_models_test/test_full/"
save_loc = "/home/tim.grimbergen/full_test_wf/test4/"
to_save = True
time_batch = True
save_fig = True
new_mismatch = True

gen_NN = GW_generator_NN(folder = model_loc, frozen=False)

N = 10000
batch_size = 200
f_min = 15
#t_coal = 1.8905370620077373

param_list = np.zeros((N,6))
real_param_list = np.zeros((N,7))
time_comp = np.zeros((N,2))
start_times = np.zeros(N)
F = np.zeros(N)
total_time = 0
long_time_grid = np.array([1000000])
short_time_grid = np.array([-1000000])

for i in range(N): #this is effectively the case batch_size = 1
	q = np.random.uniform(1,10) #model trained for q \in (1,10), s1,s2\in (-0.9, 0.9)
	s1 = np.random.uniform(-0.9,0.9)
	s2 = np.random.uniform(-0.9,0.9)
	iota = np.arccos(np.random.uniform(-1,1)) #polar angle, problem: gives weird results
	phi_0 = np.random.uniform(0,2*np.pi) #azimuthal angle, no problem
	#iota = 0
	#phi_0 = 0
	
	M = 8 * ( q / (1+q)**2)**(-3/8) #why isn't t_coal equal to ~-40 for all WF's? is the approximation formula that bad? changed 7.852093850813758 to 8 so WF is always inside prediction grid
	params = [M*q/(1+q),M/(1+q),s1,s2,1,iota,phi_0] #m1, m2, s1, s2, d, iota, phi_0
	theta = np.array(params) #D = 7 : [m1, m2, spin1_z , spin2_z, D_L, inclination, phi_0]
	param_list[i] = np.array([q,s1,s2,iota,phi_0, M])
	real_param_list[i] = theta
	
	t = time.time()
	times, h_p_real, h_c_real = generate_waveform(params[0],params[1], s1=params[2],s2 = params[3],d=params[4], iota = params[5],
												phi_0=params[6], t_coal = None, t_step = 5e-5, f_min = f_min, t_min = None,
																					verbose = False, approx = "IMRPhenomTPHM")
	
	time_comp[i,0] = time.time() - t
	
	if times[0] < long_time_grid[0]:
		long_time_grid = times
	if times[0] > short_time_grid[0]:
		short_time_grid = times
	
	t = time.time()
	theta[6] = np.pi/2 - theta[6]
	h_p_pred, h_c_pred = gen_NN.get_WF(theta, times, [(2,2),(3,3),(2,1),(4,4),(5,5)])
	#h_p_pred = np.array([np.random.uniform() for _ in range((len(times)))]) #for line profiler purposes
	#h_c_pred = np.array([np.random.uniform() for _ in range((len(times)))])
	time_comp[i,1] = time.time()-t
	#h_p_pred, h_c_pred = gen_NN.get_WF(theta, times, [(2,2)])
	
	if new_mismatch == True:
		F_p, F_c = get_random_antenna_patterns()
		h_pred = F_p*h_p_pred+F_c*h_c_pred

		#Putting all of them into pycbc
		h_p_real = timeseries.TimeSeries(h_p_real, delta_t = np.diff(times)[0])
		h_c_real = timeseries.TimeSeries(-h_c_real, delta_t = np.diff(times)[0])
		h_pred = timeseries.TimeSeries(h_pred, delta_t = np.diff(times)[0])

		h_p_real = h_p_real / np.sqrt(pycbc.filter.matchedfilter.sigmasq(h_p_real))
		h_c_real = h_c_real / np.sqrt(pycbc.filter.matchedfilter.sigmasq(h_c_real))
		h_pred = h_pred / np.sqrt(pycbc.filter.matchedfilter.sigmasq(h_pred))

		hplus_timeseries = pycbc.filter.matchedfilter.matched_filter(h_pred, h_p_real)
		hcross_timeseries = pycbc.filter.matchedfilter.matched_filter(h_pred, h_c_real)
		hpc = pycbc.filter.matchedfilter.overlap_cplx(h_p_real, h_c_real, psd = None).real

		match = pycbc.filter.matchedfilter.compute_max_snr_over_sky_loc_stat_no_phase( np.array(hplus_timeseries), np.array(hcross_timeseries), hpc, hpnorm=1, hcnorm=1)

		mismatch = 1 - np.max(np.array(match))
		print(i+1, mismatch)
		F[i] = mismatch
	if new_mismatch == False:
		h_real = h_p_real - h_c_real*1j
		h_pred = h_p_pred + h_c_pred*1j
	
		F_opt, _ = compute_optimal_mismatch(h_pred, h_real)
		print(i+1, F_opt[0])
		F[i]=F_opt
	start_times[i] = times[0]

#plt.figure(figsize=(15,8))
#plt.plot(times, h_real.real, label="real")
#plt.plot(times, h_pred.real, label="pred")
#plt.legend()
#plt.savefig("./test_figures_3/full/true_pred22.png")
if save_fig:
	plt.figure(figsize=(6,4))
	plt.scatter(param_list[:,0], np.log(F)/np.log(10))
	plt.title("mismatch median: " + str(np.median(F)))
	plt.savefig("./test_figures_7/full/mismatch_massratio"+str(N)+".png")
	plt.xlim(1,10)
	plt.xlabel("mass_ratio q")
	plt.ylabel("log(F)")
	plt.close()

	plt.figure(figsize=(8,5))
	plt.title("Speed up histogram: median = "+str(np.median(time_comp[:,0] / time_comp[:,1])))
	plt.hist(time_comp[:,0] / time_comp[:,1], rwidth=1, bins = N)
	plt.xlabel("t_true / t_model")
	plt.savefig("./test_figures_7/full/speed_up_histogram"+str(N)+".png")
	plt.close()

	plt.figure(figsize=(8,5))
	plt.title("Mismatch Histogram: median = "+str(np.median(F)))
	plt.hist(np.log(F)/ np.log(10), rwidth=1, bins = int(np.sqrt(N)), range=(-7,0))
	plt.xlabel("log(mismatch)")
	plt.savefig("./test_figures_7/full/mismatch_histogram"+str(N)+".png")
	plt.close()

	plt.figure(figsize=(6,4))
	plt.scatter(start_times, np.log(F)/np.log(10))
	plt.savefig("./test_figures_7/full/mismatch_starttime"+str(N)+".png")
	plt.xlabel("start_times")
	plt.ylabel("log(F)")
	plt.close()


if time_batch:
	gen_NN = GW_generator_NN(folder = model_loc, frozen=False, batch_size=N)
	full_batch_time_long = 0
	full_batch_time_short = 0
	for j in range(N//batch_size): #make sure batch_size is not too large or else memory problems
		t = time.time()
		h_p_pred, h_c_pred = gen_NN.get_WF(real_param_list[j*batch_size:(j+1)*batch_size], long_time_grid, [(2,2),(3,3),(2,1),(4,4),(5,5)])
		full_batch_time_long += time.time()-t
		t = time.time()
		h_p_pred, h_c_pred = gen_NN.get_WF(real_param_list[j*batch_size:(j+1)*batch_size], short_time_grid, [(2,2),(3,3),(2,1),(4,4),(5,5)])
		full_batch_time_short += time.time()-t

#print([len(data_times[row]) for row in range(N)])
save_dir_name = str(N)+"test_WF"
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
		f.write("total time in seconds for "+str(N//batch_size)+" batches (long grid): "+str(full_batch_time_long)+'\n')
		f.write("total time in seconds for "+str(N//batch_size)+" batches (short grid): "+str(full_batch_time_short)+'\n')
		f.write("f_min: "+str(f_min)+'\n')
		f.write("N: "+str(N)+'\n')
		f.write("Batch size: "+str(batch_size)+'\n')
		f.write("model_loc: " + model_loc+'\n')
