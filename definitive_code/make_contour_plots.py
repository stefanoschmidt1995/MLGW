import matplotlib.pyplot as plt 
import matplotlib.colors as colors
import numpy as np
import scipy.stats
import sys
import time
sys.path.insert(1, '../routines') #folder in which every relevant routine is saved

from MLGW_generator import *
from GW_helper import * 	#routines for dealing with datasets

generator = MLGW_generator("TD", "./models_TD_short_al_merger")

n_points = 3

q = np.linspace(1.,2.,n_points)
m2 = np.linspace(10.,25.,n_points)
s1 = np.linspace(-0.8,0.6,n_points)
s2 = np.linspace(-0.8,0.6,n_points)

full_grid = np.meshgrid(q,m2,s1,s2) #q,m2,s1,s2

F = np.zeros((len(q),len(m2),len(s1),len(s2)))

for q_, m2_, s1_, s2_ in np.nditer(full_grid):
	theta_vector_test, amp_dataset_test, ph_dataset_test, red_test_times = create_dataset_TD(1, N_grid = int(9.9e4),
				filename = None,
                t_coal = .015, q_range = q_, m2_range = m2_, s1_range = s1_, s2_range = s2_,
                t_step = 5e-5, lal_approximant = "SEOBNRv2_opt")

	rec_amp_dataset, rec_ph_dataset = generator.get_WF(theta_vector_test, plus_cross = False, x_grid = red_test_times, red_grid = True)

	id_q = np.where(q == q_)[0]
	id_m2 = np.where(m2 == m2_)[0]
	id_s1 = np.where(s1 == s1_)[0]
	id_s2 = np.where(s2 == s2_)[0]

	F[id_q,id_m2,id_s1,id_s2] = compute_mismatch(amp_dataset_test, ph_dataset_test, rec_amp_dataset, rec_ph_dataset)
	print(q_, m2_, s1_, s2_, F[id_q,id_m2,id_s1,id_s2])

	#saving F to file
np.save("mismatch_grid.dat", F)
print("Mean mismatch: ",np.mean(F))

	#contours plot for masses
print("Computing mass plot")
F_m = np.ones((len(q)+1,len(m2)+1))
for i_q, i_m2 in np.nditer(np.meshgrid(range(len(q)), range(len(m2)))):
	F_m[i_q,i_m2] = np.mean(F[i_q,i_m2,:,:])
	print(i_q,i_m2, F_m[i_q,i_m2])

set_grid = lambda grid: np.append(grid,grid[-1]+(grid[1]-grid[0])) - (grid[1]-grid[0])/2.

	#setting proper grids to heal plt lacks
q_grid = set_grid(q)
m2_grid = set_grid(m2)

plt.figure(0, figsize = (30,20))
plt.title("Mismatch of masses")
plt.pcolormesh(*np.meshgrid(q_grid,m2_grid), F_m.T, norm=colors.LogNorm(vmin=F_m.min(), vmax=F_m.max()))
#plt.pcolormesh(F_m, edgecolors = 'face')
plt.colorbar()
plt.xlabel("$q $")
plt.ylabel("$m_2 (M_{sun})$")
plt.savefig("../pictures/color_mesh_plot/masses.jpeg")

	#contours plot for spins
print("Computing spins plot")
F_s = np.ones((len(s1)+1,len(s2)+1))
for i_s1, i_s2 in np.nditer(np.meshgrid(range(len(s1)), range(len(s2)))):
	F_s[i_s1,i_s2] = np.mean(F[:,:,i_s1,i_s2])
	print(i_s1,i_s2, F_s[i_s1,i_s2])

	#setting proper grids to heal plt lacks
s1_grid = set_grid(s1)
s2_grid = set_grid(s2)

plt.figure(1, figsize = (30,20))
plt.title("Mismatch of spins")
plt.pcolormesh(*np.meshgrid(s1_grid,s2_grid), F_s.T, norm=colors.LogNorm(vmin=F_s.min(), vmax=F_s.max()))
#plt.pcolormesh(F_m, edgecolors = 'face')
plt.colorbar()
plt.xlabel("$s_1$")
plt.ylabel("$s_2$")
plt.savefig("../pictures/color_mesh_plot/spins.jpeg")
plt.show()














