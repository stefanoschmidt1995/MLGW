import mlgw.GW_generator as gen
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, '../mlgw_v1') #folder in which every relevant routine is saved
import scipy.optimize

from GW_generator import * 	#routines for dealing with datasets
from GW_helper import * 	#routines for dealing with datasets

def compute_mismatch(phi_ref, h_true, theta, times, generator):
	theta = np.array(theta)
	theta[6] = phi_ref
	h_p, h_c = generator.get_WF(theta, times)
	h_rec = h_p +1j* h_c
	return compute_optimal_mismatch(h_rec,h_true, False)[0][0]

generator = GW_generator("../mlgw_v1/TD_model_TEOBResumS")

m1 = 58.65530865199289
m2 = 29.675514740929415
s1 = 0.18690056579201775
s2 = -0.5866442879771859
try:

	iota = float(sys.argv[1])
except:
	iota =  0
logdistance = 6.172280270128777
d = np.exp(logdistance)
phi_0 = 0.

print("Trying with iota = ", iota)

iota_range = np.linspace(0,np.pi,10)

plt.figure()
t_m0 = 0.

for iota in iota_range:
	theta = np.array([m1,m2,s1,s2,d, iota, 0.])

	times, h_p, h_c, t_m = generate_waveform_TEOBResumS(m1,m2,s1,s2,d, iota, 0., t_min = 10., t_step = 1e-4)
	if iota ==0:
		t_m0 = t_m
	
	h_true = h_p+1j*h_c
	res = scipy.optimize.minimize_scalar(compute_mismatch, bounds = [0.,2*np.pi], args = (h_true, theta, times+(t_m-t_m0), generator), method = "Brent")
	

	print("F: ", res['fun'])

	h_rec = generator.get_WF(theta, times)[0]+1j*generator.get_WF(theta, times)[1]
	std_opt, phi_opt = compute_optimal_mismatch(h_true, h_rec)

	plt.scatter(iota, std_opt[0], c = 'g') #optimal mismatch computed with trash method
	plt.scatter(iota, res['fun'], c = 'k') #optimal mismatch (correct)
	plt.scatter(iota, res['x'], c = 'r')   #optimal phi_ref

	"""theta[6] = res['x']
	h_p_rec, h_c_rec = generator.get_WF(theta, times)
	plt.figure()
	plt.title(r"$h_+$")
	plt.plot(times, h_true.real, '-')
	plt.plot(times, h_p_rec, '--')
	plt.figure()
	plt.title(r"$h_x$")
	plt.plot(times, h_true.imag, '-')
	plt.plot(times, h_c_rec, '--')
	plt.figure()
	plt.title(r"$A$")
	plt.plot(times, np.abs(h_true), '-')
	plt.plot(times, np.sqrt(np.square(h_p_rec)+np.square(h_c_rec)), '--')
	plt.show()
	theta[6] = 0#"""

plt.show()

plt.plot(times, (h_rec*np.exp(1j*phi_opt)).imag, '-')
theta[6] = res['x']
print(theta)
plt.plot(times, generator.get_WF(theta, times)[1], '--')
plt.show()



print(compute_mismatch(0.1050023, h_true, theta, times, generator))















