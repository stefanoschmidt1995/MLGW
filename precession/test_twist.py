import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0,'../mlgw_v2')

import GW_generator as gen

g = gen.GW_generator()

theta = np.array([20,15,0.,.9,.0,.8,0.,-0.2])#,[20,15,0.,.3,.1,.1,0.,-0.2]]
t_grid = np.linspace(-8.,0.1,10000)
#modes = [(2,2),(2,-2),(2,1),(2,-1)]
modes = [(2,1),(2,1)]

print(g.list_modes())

theta_22 = np.concatenate([theta[None,:2], np.linalg.norm(theta[None,2:5],axis = 1)[:,None], np.linalg.norm(theta[None,5:8],axis = 1)[:,None]] , axis = 1)[0,:]
h_p, h_c = g.get_twisted_modes(theta,t_grid, modes)
amp_22, ph_22 = g.get_modes(theta_22,t_grid, (2,2), out_type = 'ampph')

print(h_p.shape)

plt.plot(t_grid, amp_22, label = '3,3 - NP')
plt.plot(t_grid, h_p[:,0], label = '3,3')
#plt.plot(t_grid, h_p[:,1], label = '3,-3')

plt.legend()
plt.show()
