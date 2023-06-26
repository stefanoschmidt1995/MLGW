import matplotlib.pyplot as plt 
import numpy as np
import scipy.stats
import sys
import time
sys.path.insert(1, '../mlgw_v1') #folder in which every relevant routine is saved

from MLGW_generator import *
from GW_helper import * 	#routines for dealing with datasets

generator = MLGW_generator("../mlgw_v1/TD_model_TEOBResumS")

N_waves = 10

frequencies = np.linspace(20,1000, 3500)
theta = np.zeros((N_waves,14))
true_amp = np.zeros((N_waves, len(frequencies)))
true_ph = np.zeros((N_waves, len(frequencies)))
rec_amp = np.zeros((N_waves, len(frequencies)))
rec_ph = np.zeros((N_waves, len(frequencies)))

for i in range(N_waves):
	q = np.random.uniform(1.,4.9)
	m2 = 10#np.random.uniform(1.,10.)
	spin1_z = 0.#np.random.uniform(-0.8,0.8)
	spin2_z = 0.#np.random.uniform(-0.8,0.8)
	d = 1#np.random.uniform(.5, 50.)
	inclination = 0.#np.random.uniform(0, 3.14)

	theta[i,:] = [q*m2, m2, 0,0, spin1_z, 0,0, spin2_z, d, inclination, 0,0,0,0]

	full_freq = np.arange(20, 1000, 5e-2)
		#getting true wave
	LALpars = lal.CreateDict()
	approx = lalsim.SimInspiralGetApproximantFromString("IMRPhenomPv2")
	hptilde, hctilde = lalsim.SimInspiralChooseFDWaveform( #where is its definition and documentation????
			q*m2*lalsim.lal.MSUN_SI, #m1
			m2*lalsim.lal.MSUN_SI, #m2
			0., 0., spin1_z, #spin vector 1
			0., 0., spin2_z, #spin vector 2
			d*1e6*lalsim.lal.PC_SI, #distance to source
			inclination, #inclination
			0., #phi ref
			0., #longAscNodes (for precession)
			0., #eccentricity
			0., #meanPerAno (for precession)
			5e-2, # frequency incremental step
			20, # lowest value of frequency
			1000, # highest value of frequency
			20, #some reference value of frequency (??)
			LALpars, #some lal dictionary
			approx #approx method for the model
	)
	true_h = np.array(hptilde.data.data)+1j*np.array(hctilde.data.data) #complex waveform
	true_h = true_h[int(20/5e-2):int(1000/5e-2)]
	temp_amp = (np.abs(true_h).real)
	temp_ph = (np.unwrap(np.angle(true_h)).real)

			#bringing waves on the chosen grid
	temp_amp = np.interp(frequencies, full_freq, temp_amp)
	temp_ph = np.interp(frequencies, full_freq, temp_ph)
	temp_ph = temp_ph - temp_ph[0] #all frequencies are shifted by a constant to make the wave start at zero phase!!!! IMPORTANT

			#removing spourious gaps (if present)
	(index,) = np.where(temp_amp/temp_amp[0] < 5e-3) #there should be a way to choose right threshold...
	if len(index) >0:
		temp_ph[index] = temp_ph[index[0]-1]
	true_h = np.multiply(temp_amp, np.exp(1j*temp_ph))

	true_amp[i,:] = true_h.real #temp_amp
	true_ph[i,:] = true_h.imag #temp_ph
	
		#generating surrogate wave
	lambd = 1.2
	new_hptilde, new_hctilde = lalsim.SimInspiralChooseFDWaveform( #where is its definition and documentation????
			lambd*q*m2*lalsim.lal.MSUN_SI, #m1
			lambd*m2*lalsim.lal.MSUN_SI, #m2
			0., 0., spin1_z, #spin vector 1
			0., 0., spin2_z, #spin vector 2
			d*1e6*lalsim.lal.PC_SI, #distance to source
			inclination, #inclination
			0., #phi ref
			0., #longAscNodes (for precession)
			0., #eccentricity
			0., #meanPerAno (for precession)
			5e-2, # frequency incremental step
			20, # lowest value of frequency
			1000, # highest value of frequency
			20, #some reference value of frequency (??)
			LALpars, #some lal dictionary
			approx #approx method for the model
	)
	true_h = np.array(new_hptilde.data.data)+1j*np.array(new_hctilde.data.data) #complex waveform
	true_h = true_h[int(20/5e-2):int(1000/5e-2)]
	temp_amp = true_h.real #(np.abs(true_h).real)
	temp_ph = true_h.imag #(np.unwrap(np.angle(true_h)).real)

			#bringing waves on the chosen grid
	temp_amp = np.interp(frequencies, full_freq, temp_amp)
	temp_ph = np.interp(frequencies, full_freq, temp_ph)
	temp_ph = temp_ph - temp_ph[0] #all frequencies are shifted by a constant to make the wave start at zero phase!!!! IMPORTANT

			#removing spourious gaps (if present)
	(index,) = np.where(temp_amp/temp_amp[0] < 5e-3) #there should be a way to choose right threshold...
	if len(index) >0:
		temp_ph[index] = temp_ph[index[0]-1]
	true_h = np.multiply(temp_amp, np.exp(1j*temp_ph))

	rec_amp[i,:] = temp_amp
	rec_ph[i,:] = temp_ph


F = compute_mismatch(true_amp, true_ph, rec_amp, rec_ph)
print("Avg fit mismatch (avg,max,min,std): ", np.mean(F), np.max(F), np.min(F), np.std(F))

#quit()

N_plots = 5
indices = np.random.choice(range(N_plots), size=N_plots ,replace = False)
for i in range(N_plots):
	plt.figure(i+1, figsize=(15,10))
	h_rec = rec_amp[indices[i]] * np.exp(1j*rec_ph[indices[i]])
	h_true = true_amp[indices[i]] * np.exp(1j*true_ph[indices[i]])


	plt.title("(q,s1,s2) = "+str(theta[indices[i],:]))

	#plt.plot(frequencies, (rec_ph[indices[i]] - true_ph[indices[i]]))
	#plt.plot(np.divide(h_rec.real, lambd*frequencies))

	plt.plot(frequencies*lambd*((1+theta[indices[i],0]))*10, 	rec_amp[indices[i]], label = "Rec")
	plt.plot(frequencies*(1+theta[indices[i],0])*10, 			true_amp[indices[i]], label = "True")
	#plt.plot((1+theta[indices[i],0])*frequencies, true_ph[indices[i]].real, label = "True")
	#plt.plot(lambd*(1+theta[indices[i],0])*frequencies, 98./68.*rec_ph[indices[i]].real, label = "rec")
	#plt.xscale("log")
	plt.legend()
	plt.savefig("../pictures/rec_WFs/WF_"+str(i)+".jpeg")

plt.show()







