import mlgw.GW_generator as gen
import numpy as np
import matplotlib.pyplot as plt
import lalsimulation as lalsim
import lal
import sys
sys.path.insert(1, '../mlgw_v1') #folder in which every relevant routine is saved

from GW_generator import * 	#routines for dealing with datasets
from GW_helper import * 	#routines for dealing with datasets


m1 = 30#58.65530865199289
m2 = 29.675514740929415
s1 = 0.18690056579201775
s2 = 0.5866442879771859
cosiota =  -0.7351822826155849
logdistance = 6.172280270128777
d = np.exp(logdistance)
phi_0 = 5.819

print("iota, phi_0: ", np.arccos(cosiota), phi_0)

times, amp, ph, h = generate_waveform(m1,m2,s1,s2,d, np.arccos(cosiota), phi_0)

generator = GW_generator("../mlgw_v1/TD_model_TEOBResumS")
rec_amp, rec_ph = generator.get_WF(np.array([m1,m2, s1, s2, d, np.arccos(cosiota), phi_0]), plus_cross = False,
				t_grid = times, red_grid = False)

h_p_rec, h_c_rec = generator.get_WF(np.array([m1,m2,0., 0., s1, 0., 0., s2, d, np.arccos(cosiota), phi_0, 0,0,0]), plus_cross = True,
				t_grid = times, red_grid = False)
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








