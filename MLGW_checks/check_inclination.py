import mlgw.GW_generator as gen
import numpy as np
import matplotlib.pyplot as plt
import lalsimulation as lalsim
import lal
import sys
sys.path.insert(1, '../routines') #folder in which every relevant routine is saved

from GW_generator import * 	#routines for dealing with datasets
from GW_helper import * 	#routines for dealing with datasets


def generate_waveform(m1,m2, s1=0.,s2 = 0.,d=1., iota = 0.,phi_0=0.):
	q = m1/m2
	mtot = (m1+m2)#*lal.MTSUN_SI
	mc = (m1*m2)**(3./5.)/(m1+m2)**(1./5.)
	mc /= 1.21 #M_c / 1.21 M_sun
	t_step =  5e-5
	t_coal = 0.25

#	f_min = 134 * mc **(-5./8.)*(1.2)**(-3./8.) * mtot**(-3.8)
	f_min = .9* ((151*(t_coal)**(-3./8.) * (((1+q)**2)/q)**(3./8.))/mtot)
		#in () there is the right scaling formula for frequency in order to get always the right reduced time
		#it should be multiplied by a prefactor for dealing with some small variation in spin
		#sounds like a good choice... and also it is able to reduce erorrs in phase reconstruction
	print(m1,m2,s1,s2,d, iota)
	print(f_min)

	hptilde, hctilde = lalsim.SimInspiralChooseTDWaveform( #where is its definition and documentation????
		m1*lalsim.lal.MSUN_SI, #m1
		m2*lalsim.lal.MSUN_SI, #m2
		0., 0., s1, #spin vector 1
		0., 0., s2, #spin vector 2
		d*1e6*lalsim.lal.PC_SI, #distance to source
		iota, #inclination
		phi_0, #phi ref
		0., #longAscNodes
		0., #eccentricity
		0., #meanPerAno
		t_step, # time incremental step
		f_min, # lowest value of freq
		f_min, #some reference value of freq (??)
		lal.CreateDict(), #some lal dictionary
#		lalsim.GetApproximantFromString('IMRPHenomPv2') #approx method for the model
		lalsim.GetApproximantFromString('SEOBNRv2_opt') #approx method for the model
		)
	h =  (hptilde.data.data+1j*hctilde.data.data)
	#ph = np.unwrap(np.angle(h))
	#h =  h * np.exp(1j*(-ph[0]-phi_0))
	(indices, ) = np.where(np.abs(h)!=0) #trimming zeros of amplitude
	h = h[indices]

	time_full = np.linspace(0.0, h.shape[0]*t_step, h.shape[0])  #time actually
	t_m =  time_full[np.argmax(np.abs(h))]
	time_full = time_full - t_m

	amp = np.abs(h)
	ph = np.unwrap(np.angle(h))
	#ph = ph - ph[np.argmax(amp)]
	#ph = ph - ph[0]

	#h = amp*np.exp(1j*ph)
#	return  time, rescaled_time, h
	return time_full, amp, ph, h


m1 = 30#58.65530865199289
m2 = 29.675514740929415
s1 = 0.18690056579201775
s2 = 0.5866442879771859
cosiota =  1#-0.7351822826155849
logdistance = 6.172280270128777
d = np.exp(logdistance)
phi_0 = 5.819

print("iota, phi_0: ", np.arccos(cosiota), phi_0)

times, amp, ph, h = generate_waveform(m1,m2,s1,s2,d, np.arccos(cosiota), phi_0)

generator = GW_generator("../definitive_code/TD_model")
rec_amp, rec_ph = generator.get_WF(np.array([m1,m2, s1, s2, d, np.arccos(cosiota), phi_0]), plus_cross = False,
				x_grid = times, red_grid = False)

h_p_rec, h_c_rec = generator.get_WF(np.array([m1,m2,0., 0., s1, 0., 0., s2, d, np.arccos(cosiota), phi_0, 0,0,0]), plus_cross = True,
				x_grid = times, red_grid = False)
h_rec = h_p_rec +1j* h_c_rec

	#cehck
#theta_vector_test, amp_dataset_test, ph_dataset_test, red_test_times = create_dataset_TD(1, N_grid = int(9.9e4),
#					filename = None,
#		            t_coal = .4, q_range = m1/m2, m2_range = (m2,m2), s1_range = s1, s2_range = s2,
#		            t_step = 1e-5, lal_approximant = "SEOBNRv2_opt", alpha = 1.)

F, phi_ref = compute_optimal_mismatch(h[np.newaxis,:-500], h_rec[:,:-500])
#F, phi_ref = compute_optimal_mismatch(h[:-500], rec_amp[0,:-500]*np.exp(1j*rec_ph[0,:-500]))
#F, phi_ref = compute_optimal_mismatch(h,h)
F_bad = compute_mismatch(np.abs(h), np.unwrap(np.angle(h)), np.abs(h_rec)[0,:], np.unwrap(np.angle(h_rec))[0,:])
print(F,F_bad, phi_ref)

plt.figure()
plt.title("h")
plt.plot(times, h,"--", label = "EOB")
plt.plot(times, h_rec[0,:]*np.exp(1j*phi_ref),"-", label = "reconstructed")
plt.legend()

plt.figure()
h_amp_ph = (rec_amp*np.exp(1j*rec_ph))
F_amp_ph, phi_ref_amp_ph = compute_optimal_mismatch(h_amp_ph[:,:-500], h_rec[:,:-500])
print(F_amp_ph, phi_ref_amp_ph)
plt.title("h amp ph")
plt.plot(times, h_rec[0,:],"--", label = "h rec")
plt.plot(times, (rec_amp*np.exp(1j*(rec_ph)))[0,:] ,"-", label = "amp ph rec")
plt.legend()

plt.figure()
plt.title("ph diff")
plt.plot(times, np.unwrap(np.angle(h))- np.unwrap(np.angle(h_rec[0,:]*np.exp(1j*phi_ref) )),"-", label = "reconstructed")


plt.show()








