import numpy as np
import pycbc.filter
from pycbc.types import timeseries
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '../../mlgw_v2')

import GW_generator as g



def get_random_antenna_patterns():
	"""
	Get random antenna patterns
	"""
	N =1
	polarization = np.random.uniform(0, 2*np.pi, N)
	latitude = np.arcsin(np.random.uniform(-1.,1., N)) #FIXME: check this is right!
	longitude = np.random.uniform(-np.pi, np.pi, N)
	
	theta = np.pi/2 - np.asarray(latitude)
	
	F_p = - 0.5*(1 + np.cos(theta)**2)* np.cos(2*longitude)* np.cos(2*polarization)
	F_p -= np.cos(theta)* np.sin(2*longitude)* np.sin(2*polarization) 
	F_c = 0.5*(1 + np.cos(theta)**2)* np.cos(2*longitude)* np.sin(2*polarization)
	F_c -= np.cos(theta)* np.sin(2*longitude)* np.cos(2*polarization) 

	return F_p, F_c


#########################
model = g.GW_generator(0)

theta = [50, 10, 0.4, -0.5, 100, 2, 0.3]
theta_signal = [50, 10, 0.4, -0.5, 100, 2, 0.3]
times = np.linspace(-10, 0.02, 10000)

hp, hc = model.get_WF(theta, times, modes = None)

hp_signal, hc_signal = model.get_WF(theta_signal, times, modes = None)
F_p, F_c = get_random_antenna_patterns()
h = F_p*hp_signal+F_c*hc_signal

	#Putting all of them into pycbc
hp = timeseries.TimeSeries(hp, delta_t = np.diff(times)[0])
hc = timeseries.TimeSeries(hc, delta_t = np.diff(times)[0])
h = timeseries.TimeSeries(h, delta_t = np.diff(times)[0])

	#Normalizing
hp = hp / np.sqrt(pycbc.filter.matchedfilter.sigmasq(hp))
hc = hc / np.sqrt(pycbc.filter.matchedfilter.sigmasq(hc))
h = h / np.sqrt(pycbc.filter.matchedfilter.sigmasq(h))

hplus_timeseries = pycbc.filter.matchedfilter.matched_filter(h, hp)
hcross_timeseries = pycbc.filter.matchedfilter.matched_filter(h, hc)
hpc = pycbc.filter.matchedfilter.overlap_cplx(hp, hc, psd = None).real

match = pycbc.filter.matchedfilter.compute_max_snr_over_sky_loc_stat_no_phase(np.array(hplus_timeseries), np.array(hcross_timeseries),
	hpc, hpnorm=1, hcnorm=1)

match = np.max(np.array(match))
print(match)

#plt.plot(hp.sample_times, hp)
plt.plot(hplus_timeseries.sample_times, np.abs(hplus_timeseries))
plt.show()
