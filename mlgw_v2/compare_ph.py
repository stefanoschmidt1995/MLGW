import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

from GW_helper import *
from GW_generator import *

modes_to_k = lambda modes:[int(x[0]*(x[0]-1)/2 + x[1]-2) for x in modes] # [(l,m)] -> [k]

modes_ = [(2,2),(3,1), (3,2),(3,3), (4,3), (4,1)]

m1, m2 = 30,30
s1,s2 = 0.3,-0.5

t, hp, hc, hlm, t_m = generate_waveform_TEOBResumS(m1, m2, s1, s2, d=1., iota = 0., t_coal = 0.4, t_step = 5e-5, f_min = None, t_min = None, modes = modes_, verbose = False, path_TEOBResumS = None)

print(hlm.keys())
t = np.array(t)

hlm1 = hlm[str(modes_to_k([(2,2)])[0])][1]
hlm2 = hlm[str(modes_to_k([(3,1)])[0])][1]
hlm3 = hlm[str(modes_to_k([(3,2)])[0])][1]

#plt.plot(t, hlm1)
#plt.plot(t, 2.*hlm2)

gen = GW_generator()
mlgw_amp, mlgw_ph = gen.get_modes([m1,m2,s1,s2], t, modes_, align22 = False)
print(mlgw_ph.shape)

id_0 = np.argmin(np.abs(t))
#plt.plot(t[:id_0], hlm1[:id_0]-2*hlm2[:id_0])

plt.plot(t[:id_0], hlm2[:id_0]-.5*hlm1[:id_0]-(hlm2[0]-2*hlm1[0]), label = 'true')

plt.plot(t[:id_0], mlgw_ph[:id_0,1]-.5*mlgw_ph[:id_0,0], label = 'mlgw')

plt.legend(loc = "upper left")
plt.show()





