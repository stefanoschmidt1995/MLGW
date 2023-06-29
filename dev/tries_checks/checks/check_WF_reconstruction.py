import numpy as np
import matplotlib.pyplot as plt

import lal
import lalsimulation as lalsim

import sys
sys.path.insert(0, '../../..')

import mlgw
from mlgw import GW_generator as g

###############################################
#lalsim.SimIMRPhenomTPHM_L0Modes

m1, m2 = 50, 10
s1x, s1y, s1z = 0, 0, -0.4
s2x, s2y, s2z = 0, 0, 0.8
iota, phi = np.pi/2, 2.

model = g(3)

modes_lal = lalsim.SimIMRPhenomTPHM_L0Modes(
#modes_lal, alpha, beta, gamma, _ = lalsim.SimIMRPhenomTPHM_JModes(
	m1*lalsim.lal.MSUN_SI, #m1
	m2*lalsim.lal.MSUN_SI, #m2
	s1x, s1y, s1z,
	s2x, s2y, s2z,
	1e6*lalsim.lal.PC_SI,
	iota, 
	1e-4, #delta T
	20, 20., #f_min, f_ref
	phi, #phi_ref
	lal.CreateDict(),
	False #only 22
)

hp_lal, hc_lal = lalsim.SimInspiralChooseTDWaveform(
	m1*lalsim.lal.MSUN_SI, #m1
	m2*lalsim.lal.MSUN_SI, #m2
	s1x, s1y, s1z,
	s2x, s2y, s2z,
	1e6*lalsim.lal.PC_SI,
	iota,  phi, #iota, phi_ref
	0,0,0,
	1e-4, #delta T
	20, 20., #f_min, f_ref
	lal.CreateDict(),
	lalsim.GetApproximantFromString('IMRPhenomTPHM')
)

phi_diff = {(2,2):0, (2,1):np.pi/2,
	(3,3): -np.pi/2, (4,4):np.pi, (5,5): np.pi/2}

h_plus, h_cross = 0, 0
h_plus_mlgw, h_cross_mlgw = 0, 0

	#getting the 22 time peak
tmp_mode = modes_lal
while tmp_mode:
	if (tmp_mode.l, tmp_mode.m) == (2,2):
		t_grid = np.linspace(0, tmp_mode.mode.deltaT*tmp_mode.mode.data.length, tmp_mode.mode.data.length)
		t_peak_22 = t_grid[np.argmax(np.abs(tmp_mode.mode.data.data))]
		amp_lm_mlgw, _ = model.get_modes([m1, m2, s1z, s2z], t_grid-t_peak_22, modes = (2,2))
		amp_scale_factor = np.max(np.abs(tmp_mode.mode.data.data))/np.max(amp_lm_mlgw)
		break
	tmp_mode = tmp_mode.next	

tmp_mode = modes_lal
while tmp_mode:# is not None:
	l,m = tmp_mode.l, tmp_mode.m
	if np.max(np.abs(tmp_mode.mode.data.data))>1e-23 and m>0:
		print(l, m)

		#t_grid = np.linspace(0, tmp_mode.mode.deltaT*tmp_mode.mode.data.length, tmp_mode.mode.data.length)
		amp_lm, ph_lm = np.abs(tmp_mode.mode.data.data), np.unwrap(np.angle(tmp_mode.mode.data.data))
		#ph_lm = ph_lm-ph_lm[0]

			#FIXME: wrong time alignment
			#Old model ph_lal = -ph_mlgw; w/o nu scaling
			#New model ph_lal = ph_mlgw; w nu scaling
			
		amp_lm_mlgw, ph_lm_mlgw = model.get_modes([m1, m2, s1z, s2z], t_grid-t_peak_22, modes = (l,m))
		ph_lm_mlgw = ph_lm_mlgw + phi_diff[(l,m)]
		amp_prefactor = 4.7864188273360336e-20*(m1+m2)/1.
		nu = m1*m2/(m1+m2)**2
		amp_lm_mlgw *= amp_prefactor*nu
		
		h_lm_mlgw_real, h_lm_mlgw_imag = model._GW_generator__set_spherical_harmonics((l,m), amp_lm_mlgw, ph_lm_mlgw, np.array(iota), np.pi/2.- np.array(phi))
		h_plus_mlgw = h_plus_mlgw + h_lm_mlgw_real
		h_cross_mlgw = h_cross_mlgw + h_lm_mlgw_imag
		
		print('\t', ph_lm[0]-ph_lm_mlgw[0])

			# setting spherical harmonics: amp, ph, D_L,iota, phi_0
		h_lm_real, h_lm_imag = model._GW_generator__set_spherical_harmonics((l,m), amp_lm, ph_lm, np.array(iota), np.pi/2.- np.array(phi))
		h_plus = h_plus + h_lm_real
		h_cross = h_cross + h_lm_imag

			#Plotting stuff
		plt.figure()
		plt.title('l,m = {}, {}'.format(l,m))
		plt.plot(t_grid, ph_lm, label = 'lal')
		plt.plot(t_grid, ph_lm_mlgw, label = 'mlgw')
		inset = plt.gca().inset_axes([.1,.1,.5,.32])
		inset.plot(t_grid[:100], ph_lm[:100], label = 'lal')
		inset.plot(t_grid[:100], ph_lm_mlgw[:100], label = 'mlgw')
		plt.legend()
		plt.savefig('mode_alignment/{}{}.png'.format(l,m))
		
		plt.figure()
		plt.plot(t_grid[:-2000], (ph_lm_mlgw-ph_lm)[:-2000], label = 'lal')
		#plt.show()
	tmp_mode = tmp_mode.next

plt.figure()
plt.plot(t_grid, h_plus, label = 'rec WF')
plt.plot(t_grid, hp_lal.data.data.real, label = 'lal')
plt.plot(t_grid, h_plus_mlgw, label = 'mlgw')
plt.legend()
plt.show()




























