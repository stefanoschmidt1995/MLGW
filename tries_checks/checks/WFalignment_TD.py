import numpy as np
import lal
import lalsimulation as lalsim
import sys
sys.path.insert(1, '../mlgw_v1') #folder in which every relevant routine is saved

from GW_helper import * 	#routines for dealing with datasets

def align_ph(amp, ph = None, merger = False):
	if ph is None:
		amp = np.abs(amp)
		ph = np.unwrap(np.angle(amp))
	if merger:
		ph = ph - ph[np.argmax(amp)]
	else:
		ph = ph - ph[0]
	#return amp*np.exp(1j*ph)
	return amp, ph

def reg_phase(wf, threshold = 1e-4):
	amp = np.abs(wf)
	ph = np.unwrap(np.angle(wf))
	(index,) = np.where(amp/np.max(amp) < threshold) #there should be a way to choose right threshold...
	if len(index) > 0:
		#print(index, ph.shape)
		amp[index[0]:] = amp[index[0]-1]
		ph[index[0]:] = ph[index[0]-1]
	else:
		index = [len(ph)-1]
	return amp*np.exp(1j*ph), index[0]

#It might be a good idea to generate WFs by scaling start frequency: in this way every signal has almost the same number of cycles (sounds like a good property...)

def generate_waveform(m1,m2, s1=0.):
	q = m1/m2
	mtot = (m1+m2)#*lal.MTSUN_SI
	mc = (m1*m2)**(3./5.)/(m1+m2)**(1./5.)
	mc /= 1.21 #M_c / 1.21 M_sun
	print(m1,m2,mc,mtot)
	print( mc **(-5./8.) * mtot**(-3./8.), (((1+q)**2)/q)**(3./8.)/mtot)

#	f_min = 134 * mc **(-5./8.)*(1.2)**(-3./8.) * mtot**(-3.8)
	f_min = .9* ((151*(.25)**(-3./8.) * (((1+q)**2)/q)**(3./8.))/mtot)
		#in () there is the right scaling formula for frequency in order to get always the right reduced time
		#it should be multiplied by a prefactor for dealing with some small variation in spin
		#sounds like a good choice... and also it is able to reduce erorrs in phase reconstruction
	print(f_min)

	hptilde, hctilde = lalsim.SimInspiralChooseTDWaveform( #where is its definition and documentation????
		m1*lalsim.lal.MSUN_SI, #m1
		m2*lalsim.lal.MSUN_SI, #m2
		0., 0., s1, #spin vector 1
		0., 0., 0., #spin vector 2
		1.*1e6*lalsim.lal.PC_SI, #distance to source
		0., #inclination
		0., #phi ref
		0., #longAscNodes
		0., #eccentricity
		0., #meanPerAno
		5e-5, # time incremental step
		f_min, # lowest value of freq
		f_min, #some reference value of freq (??)
		lal.CreateDict(), #some lal dictionary
#		lalsim.GetApproximantFromString('IMRPHenomPv2') #approx method for the model
		lalsim.GetApproximantFromString('SEOBNRv2_opt') #approx method for the model
		)

	h =  (hptilde.data.data+1j*hctilde.data.data)
	(indices, ) = np.where(np.abs(h)!=0) #trimming zeros of amplitude
	h = h[indices]

	time_full = np.linspace(0.0, h.shape[0]*5e-5, h.shape[0])  #time actually
	t_m =  time_full[np.argmax(np.abs(h))]
	time_full = time_full - t_m #[-???,??]

		#building a proper time grid
	split_point = -0.01
	time = np.hstack( (np.linspace(time_full[0], split_point, 3000),np.linspace(split_point, time_full[-1], 1500))  )
	#time = time_full

	#time_pos = np.logspace(np.log10(1e-2), np.log10(np.max(time_full)), 1000)
	#time_neg = np.flip(-np.logspace(np.log10(1e-2), np.log10(-np.min(time_full)), 4000))
	#time = np.hstack((time_neg,time_pos))

	#print("time neg/pos: ",time_neg,time_pos)
	#print(time_full, time)


	#time = np.linspace(hptilde.data.length*0.9, hptilde.data.length, hptilde.data.length*.1) #time actually
	rescaled_time = time/mtot
	amp = np.abs(h)
	ph = np.unwrap(np.angle(h))
	amp = np.interp(time, time_full, amp)
	ph = np.interp(time, time_full, ph)
	ph = ph - ph[0] #is it fine to shift waves at will?? Can I do this or am I losing some physical informations?? (this is the purpose of phi_ref!!!!)

	h = amp*np.exp(1j*ph)
#	return  time, rescaled_time, h
	return time, rescaled_time, amp, ph

q = 1.
m1 = 5.0
m1c = (m1*q*m1)**(3./5.)/(m1+m1*q)**(1./5.)
m2 = 10.0
m2c = (m2*q*m2)**(3./5.)/(m2+m2*q)**(1./5.)
t1,tr1,amp1, ph1 = generate_waveform(q*m1,m1)
t2,tr2,amp2, ph2 = generate_waveform(q*m2,m2, .5)
m1tot = (1+q)*m1
m2tot = (1+q)*m2

amp1,ph1 = align_ph(amp1,ph1, True)
amp2,ph2 = align_ph(amp2, ph2, True)

t1_merger = np.argmax(amp1)
t2_merger = np.argmax(amp2)

print("end: ",(t1-t1[t1_merger])[-1]/m1tot)

	#better to do interpolation in amp /ph space rather than in h space
		#Interpolation is more able to track amplitude behaviour than in the case of a wiggly function
amp3 = m2/m1*np.interp((t2-t2[t2_merger])/m2tot, (t1-t1[t1_merger])/m1tot, amp1 )
ph3 = np.interp((t2-t2[t2_merger])/m2tot, (t1-t1[t1_merger])/m1tot, ph1 )

#wf3 = m2/m1*np.interp((t2-t2[t2_merger])/m2, (t1-t1[t1_merger])/m1, wf1) #don't do this: much more unstable!!

amp3, ph3 = align_ph(amp3, ph3, True)

	#setting a huge grid to make things better
#"""
t1_huge = np.linspace(t1[0],t1[-1], 100000)
amp1 = np.interp(t1_huge, t1, amp1)
ph1 = np.interp(t1_huge, t1, ph1)
t1 = t1_huge

t2_huge = np.linspace(t2[0],t2[-1], 100000)
amp2 = np.interp(t2_huge, t2, amp2)
ph2 = np.interp(t2_huge, t2, ph2)
amp3 = np.interp(t2_huge, t2, amp3)
ph3 = np.interp(t2_huge, t2, ph3)
t2 = t2_huge
#"""


t1_merger = np.argmax(amp1)
t2_merger = np.argmax(amp2)


wf1 = amp1*np.exp(1j*ph1)
wf2 = amp2*np.exp(1j*ph2)
wf3 = amp3*np.exp(1j*ph3)

print(wf1.shape, wf2.shape, wf3.shape)
print(t1.shape,t2.shape)

print("Merger times: ", t1[t1_merger], t2[t2_merger])

import matplotlib.pyplot as plt

fig = plt.figure()
plt.title('rescaled times')
ax = fig.add_subplot(111)
ax.plot((t1-t1[t1_merger])/m1tot, wf1/m1tot, color='b')
ax.plot((t2-t2[t2_merger])/m2tot, wf2/m2tot, color='k')
#ax.plot(tr2, wf3, color='r')

fig = plt.figure()
plt.title('interpolated prediction')
ax = fig.add_subplot(111)
ax.plot((t2-t2[t2_merger]), wf2, color = 'k')
ax.plot((t2-t2[t2_merger]), wf3, color = 'red')

print(compute_mismatch(np.abs(wf2), np.unwrap(np.angle(wf2)), np.abs(wf3), np.unwrap(np.angle(wf3))))

#plt.show()
#quit()

fig = plt.figure()
plt.title('amplitudes')
ax = fig.add_subplot(111)
ax.plot((t2-t2[t2_merger])/m2tot, np.abs(amp2)/np.max(amp2), color='k')
ax.plot((t2-t2[t2_merger])/m2tot, np.abs(amp3)/np.max(amp3), color='r')


fig = plt.figure()
plt.title('phases')
ax = fig.add_subplot(111)
ax.plot((t2-t2[t2_merger]), np.unwrap(np.angle(wf2))/ np.unwrap(np.angle(wf1)), color='k')
#ax.plot((t1-t1[t1_merger])/m1tot, np.unwrap(np.angle(wf1)), color='b')
#ax.plot((t2-t2[t2_merger]), np.unwrap(np.angle(wf3)), color='r')

fig = plt.figure()
plt.title('phase difference')
ax = fig.add_subplot(111)
ax.plot((t2-t2[t2_merger]), np.unwrap(np.angle(wf3)) - np.unwrap(np.angle(wf2)), color='k')


plt.show()
