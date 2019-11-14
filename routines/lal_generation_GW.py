import lalsimulation as lalsim
import lal
import matplotlib.pyplot as plt
import numpy as np

#maggiore gravitational waves vol 1 per cose un po' teoriche...
#leggi nested sampling meglio
#leggi articoli https://arxiv.org/abs/0907.0700 + https://arxiv.org/abs/1801.08009
#studia meglio metodi probabilistici a rete...

M = [5,100] #m1>m2
spin = [-.9,.9]


m1 = 10
m2 = 1
spin1z = 0.2
spin2z = 0.4
d = 1
LALpars = lal.CreateDict()
approx = lalsim.SimInspiralGetApproximantFromString('IMRPhenomD')

print(approx)

hptilde, hctilde = lalsim.SimInspiralChooseFDWaveform(
                    m1*lalsim.lal.MSUN_SI,
                    m2*lalsim.lal.MSUN_SI,
                    0, 0, spin1z,
                    0, 0, spin2z,
                    d*1e6*lalsim.lal.PC_SI,
                    0,
                    0,
                    0, #longAscNodes
                    0, #eccentricity
                    0, #meanPerAno
                    1/16,
                    10,
                    2048,
                    #4096,
                    30,
                    LALpars,
                    approx)

print(hptilde.data.data.shape, (2048)/0.1)

hptilde2, hctilde2 = lalsim.SimInspiralChooseFDWaveform(
                    m1*lalsim.lal.MSUN_SI,
                    m2*lalsim.lal.MSUN_SI,
                    0, 0, spin1z,
                    0, 0, .19,
                    d*1e6*lalsim.lal.PC_SI,
                    0,
                    0,
                    0, #longAscNodes
                    0, #eccentricity
                    0, #meanPerAno
                    .1,
                    30,
                    2048,
                    30,
                    LALpars,
                    approx)

#f = [i for i in np.arange(0,2049,.1)]

h = np.array(hptilde.data.data)+1j*np.array(hctilde.data.data) #amplitude
ph = np.unwrap(np.angle(h)) #phase (why unwrap?????)

h2 = np.array(hptilde2.data.data)+1j*np.array(hctilde2.data.data) #amplitude
ph2 = np.unwrap(np.angle(h2)) #phase

#print(np.arange(30,2048, 0.1))

plt.plot(np.arange(10,2048, 1/16),h[100:int(2048/(1/16))])
#plt.plot((np.abs(h2)*1e21))
#plt.yscale("log")
plt.show()


