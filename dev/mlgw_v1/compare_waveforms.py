import numpy as np
import lal
import lalsimulation as lalsim

import matplotlib.pyplot as plt

from GW_generator import *
from GW_helper import * 	#routines for dealing with datasets

generator = GW_generator("./TD_model_TEOBResumS")

m1 = 50.0
m2 = 45.0
d = 1

spin1z = 0.0
spin2z = 0.0

t_step = 5e-5
f_min  = 20.0

for inclination in np.linspace(-np.pi/2,np.pi/2,10):

    hptilde, hctilde = lalsim.SimInspiralChooseTDWaveform( #where is its definition and documentation????
        m1*lalsim.lal.MSUN_SI, #m1
        m2*lalsim.lal.MSUN_SI, #m2
        0., 0., spin1z, #spin vector 1
        0., 0., spin2z, #spin vector 2
        d*1e6*lalsim.lal.PC_SI, #distance to source
        inclination, #inclination
        0., #phi ref
        0., #longAscNodes
        0., #eccentricity
        0., #meanPerAno
        t_step, # time incremental step
        f_min, # lowest value of time
        f_min, #some reference value of time (??)
        lal.CreateDict(), #some lal dictionary
        lalsim.GetApproximantFromString('SEOBNRv2_opt') #approx method for the model
    )
    times = t_step*np.linspace(0.0,len(hptilde.data.data),len(hptilde.data.data))
    times -= times[np.argmax(hptilde.data.data**2+hctilde.data.data**2)]

    theta = np.column_stack((m1/m2, spin1z, spin2z))    
    amp, ph = generator.get_raw_WF(theta)
	

    hpml, hcml = generator(times,
                           m1,
                           m2,
                           0.0,
                           0.0,
                           spin1z,
                           0.0,
                           0.0,
                           spin2z,
                           d,
                           inclination,
                           0.0,
                           0.0,
                           0.0,
                           0.0)

    h = amp * np.exp(1j*ph)
    
    plt.plot(times, hptilde.data.data, 'r')
    #plt.plot(times, np.squeeze(hpml), 'b')
    plt.plot(generator.times, h.real[0,:])
    plt.show()
    exit()
