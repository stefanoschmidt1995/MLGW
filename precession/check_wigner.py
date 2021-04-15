"""
This script checks the robustness of the my implementation of Wigner functions.
The benchmark is from the package spherical_functions: https://github.com/moble/spherical_functions
Apparently everything works quite well (apart from a complex conjugate)
"""

import spherical_functions as sf #goes only with numpy == 1.20.2
import numpy as np
import sys
sys.path.insert(0,'../mlgw_v2')

import GW_generator as gen
g = gen.GW_generator(1)

for i in range(10000):

	alpha, beta, gamma =  np.array([np.random.uniform(0,2*np.pi)]),  np.array([np.random.uniform(0,2*np.pi)]), np.array([np.random.uniform(0,2*np.pi)])
	ell = np.random.randint(5) + 2
	mp,m = [np.random.randint(ell+1)],[np.random.randint(ell+1)]
	mp[0] *= (np.random.randint(2)*2-1)
	m[0] *= (np.random.randint(2)*2-1)
	
	wig_package = sf.Wigner_D_element(alpha[0], beta[0], gamma[0], ell, mp[0], m[0])

	wig_mine = g._GW_generator__get_Wigner_D_matrix(ell,mp,m, alpha,beta,gamma)

	all_close = np.allclose(wig_package, np.conj(wig_mine), rtol = 1e-5)
	print("Iteration: ",i)
	print("l,m,mp ",ell,mp,m)
	print("\t",all_close, wig_mine, wig_package)

	assert all_close

#print("package wigner ", wig_package)
#print("my wigner ", wig_mine)
