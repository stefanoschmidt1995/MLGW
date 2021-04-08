#TODO: try to understand how the Wigner matrices work. It is not very clear how the inverse of Wigner's matrix is defined

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0,'../mlgw_v2')

import GW_generator as gen

g = gen.GW_generator()

m_list = [-2,-1,1,2]
beta = np.array([0.])

d_mprime_m = np.zeros((len(m_list),len(m_list)))

for i, m_prime_ in enumerate(m_list):
	for j, m_ in enumerate(m_list):
		d_mprime_m[i,j] = g._GW_generator__get_Wigner_d_function(2, m_list[i],m_list[j], beta)

print(d_mprime_m)
print(np.einsum('ij,ik->jk',d_mprime_m,d_mprime_m))
