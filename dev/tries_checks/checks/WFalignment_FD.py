import numpy as np
import lal
import lalsimulation as lalsim

#export LAL_DATA_PATH=/home/stefano/Documents/Stefano/scuola/uni/tesi_magistrale/code/data_ROM/

def align_ph(wf):
	amp = np.abs(wf)
	ph = np.unwrap(np.angle(wf))
	ph = ph - ph[0]
	return amp*np.exp(1j*ph)

def generate_waveform(m1,m2):
    mtot = (m1+m2)*lal.MTSUN_SI


    f_min = 20.0
    f_max = 2048.0
    df    = 1./32.

    f_rescaled_min = f_min*mtot
    f_rescaled_max = f_max*mtot
    df_rescaled = mtot*df

    hptilde, hctilde = lalsim.SimInspiralChooseFDWaveform( #where is its definition and documentation????
        m1*lalsim.lal.MSUN_SI, #m1
        m2*lalsim.lal.MSUN_SI, #m2
        0., 0., .5, #spin vector 1
        0., 0., 0., #spin vector 2
        1.*1e6*lalsim.lal.PC_SI, #distance to source
        0., #inclination
        0., #phi ref
        0., #longAscNodes
        0., #eccentricity
        0., #meanPerAno
        1e-3, # frequency incremental step
        f_min, # lowest value of frequency
        f_max, # highest value of frequency
        f_min, #some reference value of frequency (??)
        lal.CreateDict(), #some lal dictionary
#        lalsim.GetApproximantFromString('IMRPHenomPv2') #approx method for the model
        lalsim.GetApproximantFromString('SEOBNRv4_ROM') #approx method for the model
        )

    frequency = np.linspace(0.0, f_max, hptilde.data.length)
    rescaled_frequency = frequency*mtot
    print(mtot)
    return  frequency, rescaled_frequency, hptilde.data.data+1j*hctilde.data.data

q = 15.
m1 = 5.0
m1c = (m1*q*m1)**(3./5.)/(m1+m1*q)**(1./5.)
m2 = 15.0
m2c = (m2*q*m2)**(3./5.)/(m2+m2*q)**(1./5.)
m1tot = (1+q)*m1
m2tot = (1+q)*m2
f1,fr1,wf1 = generate_waveform(m1,m1)
f2,fr2,wf2 = generate_waveform(m2,m2)

#wf2 = np.interp(fr1,fr2,wf1)

wf1 = align_ph(wf1)
wf2 = align_ph(wf2)

amp1= np.abs(wf1)
amp2= np.abs(wf2)
ph1 = np.unwrap(np.angle(wf1))
ph2 = np.unwrap(np.angle(wf2))

#wf3 = (m1c/m2c)**(-2./6.)*np.interp(f1,f1*m1/m2,wf1)*m2/m1

#wf3 = m2/m1*np.interp(fr2, fr1, wf1)
#phi = np.interp(f1/m2, f1/m1, phi)
#wf3 = np.interp(f2, f1/m2, wf3)

print(amp1,amp2)

	#mistery???
t1 = 2.18 * (1.21/m1c)**(5./3.) * (100/f1[np.nonzero(amp1)[0][0]])**(8./3.)
t2 = 2.18 * (1.21/m2c)**(5./3.) * (100/f2[np.nonzero(amp2)[0][0]])**(8./3.)

#print(t1,t2)

import matplotlib.pyplot as plt

fig = plt.figure()
plt.title('ph')
ax = fig.add_subplot(111)
#ax.plot(fr1, np.unwrap(np.angle(wf1*np.exp(-1j*2*np.pi*f1*t1))).real, color='b')
#ax.plot(fr2, np.unwrap(np.angle(wf2*np.exp(-1j*2*np.pi*f2*t2))).real, color='k')
ax.plot(fr1, np.unwrap(np.angle(wf1)), color='b')
ax.plot(fr2, np.unwrap(np.angle(wf2)), color='k')

fig = plt.figure()
plt.title('amp')
ax = fig.add_subplot(111)
ax.plot(fr1, np.abs(wf1), color='b')
ax.plot(fr2, np.abs(wf2), color='k')
#ax.plot(fr2, wf3, color='r')

plt.show()
quit()


fig = plt.figure()
plt.title('interpolated prediction')
ax = fig.add_subplot(111)
ax.plot(f2, wf2, color = 'k')
ax.plot(f2, wf3, color = 'red')

plt.show()
