import numpy as np
import matplotlib.pyplot as plt

from precession_helper import *

import sys
sys.path.insert(0,'../../mlgw_v2')
from GW_helper import *

def unwrap_general(p, discont=np.pi, modulo = 2*np.pi, axis=-1):
	import numpy.core.numeric as _nx
	p = np.asarray(p)
	nd = p.ndim
	dd = np.diff(p, axis=axis)
	slice1 = [slice(None, None)]*nd	 # full slices
	slice1[axis] = slice(1, None)
	slice1 = tuple(slice1)
	ddmod = np.mod(dd + modulo/2, modulo) - modulo/2
	_nx.copyto(ddmod, modulo/2, where=(ddmod == -modulo/2) & (dd > 0))
	ph_correct = ddmod - dd
	_nx.copyto(ph_correct, 0, where=abs(dd) < discont)
	up = np.array(p, copy=True, dtype='d')
	up[slice1] = p[slice1] + ph_correct.cumsum(axis)
	return up

#alpha, beta = get_alpha_beta(1.4, .4, .14, .75, 1.34, 3., 500., times)

#create_dataset_alpha_beta(50, "angles.dat", 10000, 1000, (1.1,10.))

params, alpha, beta, times = load_dataset("validation_angles.dat", n_params = 6, N_data = 10)

print(times)

#alpha_unwrap = unwrap_general(alpha, modulo = 0.1)
#beta_unwrap = unwrap_general(beta, modulo = 0.001)
#print(np.allclose(alpha_unwrap, beta_unwrap))

x = np.sin(3*times)
y = np.cos(3*times)

alpha_unwrap = np.arctan2(x,y)
beta_unwrap = np.unwrap(alpha_unwrap)


plt.figure()
plt.plot(times, alpha.T)

plt.figure()
plt.plot(times, beta.T)

plt.figure()
plt.title("alpha")
plt.plot(alpha_unwrap.T)

plt.figure()
plt.title("beta")
plt.plot(beta_unwrap.T)

plt.show()


