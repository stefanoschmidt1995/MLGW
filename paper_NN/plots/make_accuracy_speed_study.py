"""
Script to test the accuracy and the performance a mlgw model
"""
import mlgw
from mlgw.GW_helper import generate_waveform

import numpy as np
import matplotlib.pyplot as plt
import lal
import lalsimulation as lalsim
import timeit

import argparse
import sys
import os
import warnings
import json

import lalsimulation as lalsim
from tqdm import tqdm

import pycbc.filter
from pycbc.types import timeseries

###################################

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

#########################################################################

lal_cmd = """
hp, hc = lalsim.SimInspiralChooseTDWaveform( 
		m1*lalsim.lal.MSUN_SI, #m1
		m2*lalsim.lal.MSUN_SI, #m2
		0, 0, s1z,
		0, 0, s2z,
		d*1e6*lalsim.lal.PC_SI, #distance to source (in pc)
		iota, #inclination
		np.pi/2. - phi_ref, #phi ref
		0., #longAscNodes
		0., #eccentricity
		0., #meanPerAno
		t_step, # time incremental step
		f_min, # lowest value of freq
		f_min, #some reference value of freq
		lal.CreateDict(), #some lal dictionary
		approx #approx method for the model
		)
	"""
mlgw_cmd = "hp_mlgw, hc_mlgw = generator.get_WF(theta, times, modes = modes)"
mlgw_cmd_batches = "hp_mlgw, hc_mlgw = generator.get_WF(np.repeat(theta[None,:], 100, axis =0), times, modes = modes)"

#########################################################################

parser = argparse.ArgumentParser(__doc__)

parser.add_argument(
	"--model-folder", type = str, required = True,
	help="Folder to load the model from (if an int is given, one of the default package models is used)")

parser.add_argument(
	"--n-wfs", type = int, required = False, default = 1000,
	help="Number of WFs to generate")

parser.add_argument(
	"--verbose", action = 'store_true', default = False,
	help="Whether to print some output")

parser.add_argument(
	"--plot", action = 'store_true', default = False,
	help="Whether to plot the comparison between modes")

parser.add_argument(
	"--output-file", type = str, required = True,
	help="Output file where all the results are store (in json)")


args = parser.parse_args()

if args.model_folder.isnumeric():
	args.model_folder = int(args.model_folder)

generator = mlgw.GW_generator(args.model_folder)	#initializing the generator with standard model
generator.get_WF([10, 3, 0.3, -0.6], np.linspace(-8,0.01, 1000), modes = None) #The first call to the WF is always the slowest...

		#getting random theta
		#Safe zone: m1_range = (10, 50)
M_range = (30., 80.)
q_range = (1.,10.)
s1_range = (-0.9,0.9)
s2_range = (-0.9,0.9)
d_range = (.5,100.)
i_range = (0, np.pi) 
phi_0_range = (0, 2*np.pi)
f_range = (10, 20)
LALpars = lal.CreateDict()
approx = lalsim.SimInspiralGetApproximantFromString("IMRPhenomTPHM")
#approx = lalsim.SimInspiralGetApproximantFromString("SEOBNRv4PHM")
#approx = lalsim.SimInspiralGetApproximantFromString("SEOBNRv4HM_ROM")
t_step = 1/(2*4096.) #srate = 4069 Hz

modes = [(2,2), (2,1), (3,3), (4,4), (5,5)]

low_list = [M_range[0],q_range[0], s1_range[0], s2_range[0], d_range[0], i_range[0], phi_0_range[0]]
high_list = [M_range[1],q_range[1], s1_range[1], s2_range[1], d_range[1], i_range[1], phi_0_range[1]]

print("Saving output to file: ", args.output_file)
if os.path.isfile(args.output_file):
	warnings.warn("The file '{}' already exists: new entries will be appended. Is this intended?".format(args.output_file))
json_file = open(args.output_file, 'a')

for i in tqdm(range(args.n_wfs), disable = args.verbose, desc = 'Loops on WFs'):
			#computing test WFs
			#add support also for lal WFs, 'cause you need to check!
	theta = np.random.uniform(low = low_list, high = high_list, size = (7, ))
	theta[:2] = theta[0]*theta[1]/(1+theta[1]), theta[0]/(1+theta[1]) #from M, q to m1, m2
	f_min = np.random.uniform(*f_range)

	m1, m2, s1z, s2z, d, iota, phi_ref  = theta

	new_dict_row = {'m1':m1, 'm2':m2, 's1z':s1z, 's2z':s2z, 'd':d, 'iota':iota, 'phi_ref':phi_ref, 'f_min':f_min}
	if args.verbose: print("it: {} - theta {} ".format( i, theta[:4]))
	
		########
		# Generating the modes
	hlm = lalsim.SimInspiralChooseTDModes(0.,
			t_step,
			m1*lalsim.lal.MSUN_SI,
			m2*lalsim.lal.MSUN_SI,
			0.,
			0.,
			s1z,
			0.,
			0.,
			s2z,
			f_min,
			f_min,
			1e6*lalsim.lal.PC_SI,
			LALpars,
			5,			#lmax
			approx
		)
	prefactor = 4.7864188273360336e-20
	nu =  np.divide(m1/m2, np.square(1+m1/m2))
	amp_prefactor = prefactor*(m1+m2)#/1.*nu

		#Setting up the time grid			
	h_22 = lalsim.SphHarmTimeSeriesGetMode(hlm,2,2).data.data/amp_prefactor
	times = np.linspace(0.0, len(h_22)*t_step, len(h_22)) #time grid at which wave is computed
	times = times - times[np.argmax(np.abs(h_22))]
	amp_mlgw, ph_mlgw = generator.get_modes(theta, times, modes, out_type = 'ampph')

	new_dict_row['T'] = np.abs(times[-1]-times[0])
	new_dict_row['srate'] = 1/t_step
			
		#####
		# Generating the waveform (& timing)
	exec(lal_cmd)
	exec(mlgw_cmd)
	hc_mlgw = -hc_mlgw #to make it compatible with lal conventions

	time_lal = timeit.timeit(lal_cmd, globals = globals(), number = 2)/2
	time_mlgw = timeit.timeit(mlgw_cmd, globals = globals(), number = 2)/2
	time_mlgw_batches = timeit.timeit(mlgw_cmd_batches, globals = globals(), number = 2)/200

	if args.verbose: print("\tTime lal | mlgw | mlgw batches: {:.5f} {:.5f} ({:.3f}) {:.5f} ({:.3f})".format(time_lal, time_mlgw, time_lal/time_mlgw, time_mlgw_batches, time_lal/time_mlgw_batches))
	
	new_dict_row['time_lal'] = time_lal
	new_dict_row['time_mlgw'] = time_mlgw
	new_dict_row['time_mlgw_100'] = time_mlgw_batches
	
		#####
		# Match computations
	#TODO: implement here match computation

	F_p, F_c = get_random_antenna_patterns()
	h_pred = F_p*hp.data.data+F_c*hc.data.data

		#Putting all of them into pycbc
	hp_pycbc = timeseries.TimeSeries(hp_mlgw, delta_t = t_step)
	hc_pycbc = timeseries.TimeSeries(hc_mlgw, delta_t = t_step)
	h_pred = timeseries.TimeSeries(h_pred, delta_t = t_step)

	hp_pycbc = hp_pycbc / np.sqrt(pycbc.filter.matchedfilter.sigmasq(hp_pycbc))
	hc_pycbc = hc_pycbc / np.sqrt(pycbc.filter.matchedfilter.sigmasq(hc_pycbc))
	h_pred = h_pred / np.sqrt(pycbc.filter.matchedfilter.sigmasq(h_pred))

	hplus_timeseries = pycbc.filter.matchedfilter.matched_filter(h_pred, hp_pycbc)
	hcross_timeseries = pycbc.filter.matchedfilter.matched_filter(h_pred, hc_pycbc)
	hpc = pycbc.filter.matchedfilter.overlap_cplx(hp_pycbc, hc_pycbc, psd = None).real

	match = pycbc.filter.matchedfilter.compute_max_snr_over_sky_loc_stat_no_phase( np.array(hplus_timeseries), np.array(hcross_timeseries), hpc, hpnorm=1, hcnorm=1)

	mismatch = 1 - np.max(np.array(match))
	if args.verbose: print('\tMismatch', mismatch)
	new_dict_row['mismatch'] = mismatch

		#Match mode by node
	for j, (l,m) in enumerate(modes):
		lm = str(l)+str(m)
		hlm_IMR = lalsim.SphHarmTimeSeriesGetMode(hlm, l, m).data.data/amp_prefactor
		hlm_mlgw = amp_mlgw[:,j]*np.exp(1j*ph_mlgw[:,j])
		mismatch_mode = 1 - pycbc.filter.match(timeseries.TimeSeries(hlm_mlgw.real, delta_t = t_step),
					timeseries.TimeSeries(hlm_IMR.real, delta_t = t_step), psd = None)[0]
		if args.verbose: print('\t\t', lm, mismatch_mode)

		new_dict_row['mismatch_{}'.format(lm)] = mismatch_mode
	
	json_file.write(json.dumps(new_dict_row) + '\n')
	
	if args.plot:
		ids_plot, = np.where(times<0.01)
		fig, ax_list = plt.subplots(nrows = 2, ncols = 1, sharex = True)
		ax_list[0].plot(times[ids_plot], hp_mlgw[ids_plot], label = 'mlgw')
		ax_list[0].plot(times[ids_plot], hp.data.data[ids_plot], label = 'lal')
		ax_list[1].plot(times[ids_plot], hc_mlgw[ids_plot], label = 'mlgw')
		ax_list[1].plot(times[ids_plot], hc.data.data[ids_plot], label = 'lal')
		ax_list[0].legend()
		#plt.show()
		
	
		for j, (l,m) in enumerate(modes):
			lm = str(l)+str(m)
			
			hlm_IMR = lalsim.SphHarmTimeSeriesGetMode(hlm, l, m).data.data/amp_prefactor
			ph_IMR = np.unwrap(np.angle(hlm_IMR))
			
			fig, ax_list = plt.subplots(num=int(lm+'0'), nrows = 3, ncols = 1, sharex = True)

			ax_list[0].set_title("Amp mode {}".format(lm))
			ax_list[0].plot(times[ids_plot], amp_mlgw[ids_plot,j], label = "mlgw")	
			ax_list[0].plot(times[ids_plot], np.abs(hlm_IMR)[ids_plot], label = "IMR")
			ax_list[0].legend(loc = 'upper right')
			
			ax_list[1].set_title("Ph mode {}".format(lm))
			ax_list[1].plot(times[ids_plot], ph_mlgw[ids_plot,j], label = "mlgw")
			ax_list[1].plot(times[ids_plot], ph_IMR[ids_plot], label = "IMR")

			ax_list[2].plot(times[ids_plot], (ph_mlgw[ids_plot,j] - ph_IMR[ids_plot])- (ph_mlgw[0,0] - ph_IMR[0]), label = "mlgw - IMR")
			ax_list[2].legend(loc = 'upper left')
			plt.tight_layout()
		plt.show()

json_file.close()

#Read the file with `pd.read_json('test.json', lines = True)`




