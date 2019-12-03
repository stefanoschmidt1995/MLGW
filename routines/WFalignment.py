import numpy as np
import lal
import lalsimulation as lalsim

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
        0., 0., 0.5, #spin vector 1
        0., 0., 0., #spin vector 2
        1.*1e6*lalsim.lal.PC_SI, #distance to source
        0., #inclination
        0., #phi ref
        0., #longAscNodes
        0., #eccentricity
        0., #meanPerAno
        df, # frequency incremental step
        f_min, # lowest value of frequency
        f_max, # highest value of frequency
        f_min, #some reference value of frequency (??)
        lal.CreateDict(), #some lal dictionary
        lalsim.GetApproximantFromString('IMRPHenomPv2') #approx method for the model
        )

    frequency = np.linspace(0.0, f_max, hptilde.data.length)
    rescaled_frequency = frequency*mtot
    return  frequency, rescaled_frequency, hptilde.data.data+1j*hctilde.data.data

m1 = 3.0
m1c = (m1*m1)**(3./5.)/(m1+m1)**(1./5.)
m2 = 20.0
m2c = (m2*m2)**(3./5.)/(m2+m2)**(1./5.)
f1,fr1,wf1 = generate_waveform(m1,m1)
f2,fr2,wf2 = generate_waveform(m2,m2)

wf3 = (m1c/m2c)**(-2./6.)*np.interp(f1*m2/m1,f1,wf1)*m2/m1

import matplotlib.pyplot as plt

fig = plt.figure()
plt.title('rescaled freqs')
ax = fig.add_subplot(111)
ax.plot(fr1, wf1, color='b')
ax.plot(fr2, wf2, color='k')
ax.plot(fr2, wf3, color='r')

fig = plt.figure()
plt.title('interpolated prediction')
ax = fig.add_subplot(111)
ax.plot(f2, wf2, color = 'k')
ax.plot(f2, wf3, color = 'red')

plt.show()
