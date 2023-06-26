import lalsimulation,lal
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, '/home/stefano/Desktop/Stefano/scuola/uni/tesi_magistrale/code/mlgw_v2')
import GW_generator as gen
import GW_helper 


nqcCoeffsInput=lal.CreateREAL8Vector(10) ##This will be unused, but it is necessary
m1 = 67.59289191947715
m2 = 52.901795866814176
phi_c = 1.662251934907694
f_start22 = 8. #Frequency of the 22 mode at which the signal starts
distance =47.319574599147124
spin1_z = 0.1893378340991678
spin2_z =  0.94369664350014335
deltaT = 1./16384.


sphtseries, dyn, dynHi = lalsimulation.SimIMRSpinAlignedEOBModes(deltaT, m1*lal.MSUN_SI, m2*lal.MSUN_SI, f_start22, distance*lal.PC_SI, spin1_z, spin2_z,41, 0., 0., 0.,0.,0.,0.,0.,0.,1.,1.,nqcCoeffsInput, 0) 

sp = sphtseries
hlms = []

while sp is not None:
	#try:
	print("lm", sp.l, sp.m)
	hlms.append(sp.mode.data.data)
	sp = sp.next
	#except:
	#	break


id_ = np.argmax(np.abs(hlms[-1]))

times = np.arange(0,len(hlms[-1])*deltaT,deltaT) 
times = times - times[id_]


times_TEOB, h_p_TEOB, h_c_TEOB, hlm, t_m = GW_helper.generate_waveform_TEOBResumS(m1,m2, spin1_z, spin2_z, distance/1e6, f_min = f_start22,
								verbose = False, t_step = 1e-4, modes = [(3,3),(2,2)] )
modes_to_k = lambda modes:[int(x[0]*(x[0]-1)/2 + x[1]-2) for x in modes] # [(l,m)] -> [k]

g = gen.GW_generator('/home/stefano/Documents/Stefano/scuola/uni/tesi_magistrale/code/mlgw_package/mlgw/TD_models/model_0/')

amp, ph = g.get_modes([m1,m2,spin1_z, spin2_z], times,  (3,3))
h_p, h_c = g.get_WF([m1,m2,spin1_z, spin2_z,distance/(1e6),0.,0.], times)
q = m1/m2
nu = np.divide(q, np.square(1+q)) #symmetric mass ratio

prefactor = 4.7864188273360336e-20 # G/c^2*(M_sun/Mpc)
prefactor = prefactor*(m1+m2)/(distance/(1e6))

	#amp
#plt.plot(times, amp, label = "mlgw")
#plt.plot(times, np.abs(hlms[2])/prefactor/nu, label ="SEOB")
#plt.plot(times, np.abs(hlms[-1])/prefactor/nu, label ="SEOB_22")

	#ph
plt.plot(times, ph, label = "mlgw")
plt.plot(times, np.unwrap(np.angle(hlms[2])), label ="SEOB")
plt.plot(times_TEOB, hlm[str(modes_to_k([(3,3)])[0])][1], label = "TEOB")

#factor = np.sqrt(5./(64.*np.pi))*4.
#plt.plot(times, np.abs(hlms[-1]), label ="SEOB")
#plt.plot(times, np.abs(h_p+1j*h_c)/(factor)*nu, label = 'mlgw')
plt.legend()

plt.show()


