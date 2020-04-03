import sys
sys.path.insert(1, '../routines')
import matplotlib.pyplot as plt
import numpy as np
import GW_generator as gen

###It does not work for spins...

model = gen.GW_generator("/home/stefano/Desktop/Stefano/scuola/uni/tesi_magistrale/code/definitive_code/TD_model")

#std_q = lambda q: [q*5.,5.,.1,.3, 1e-19, 0, 0]
std_q = lambda q: np.array([[3.,q,-0.1]])

t = np.linspace(-2,0.05, 5000)
def get_WF(q):
	print(std_q(q))
	#hp, hc = model.get_WF(std_q(q), t)
	#h= (hp+1j*hc)
	amp, ph = model.get_raw_WF(std_q(q))
	return amp[0,:]

def get_grads(q):
	#g_hp, g_hc = model.get_grads(std_q(q0),t)
	g_amp, g_ph = model.get_raw_grads(std_q(q))
	return g_amp[0,:,0]

def get_red_coeff(q):
	g_amp, g_ph = model.get_red_coefficients(std_q(q))
	return g_amp[0,:]

epsilon = .00005
q0= .3
WF_4pe = get_WF(q0+epsilon)
WF_4me = get_WF(q0-epsilon)
WF_4pe = get_red_coeff(q0+epsilon)
WF_4me = get_red_coeff(q0-epsilon)

#plt.plot(model.get_time_grid(), (WF_4pe- WF_4me)/(2*epsilon),'o', ms =2, label = "numerical")
#plt.plot(model.get_time_grid(), get_grads(q0) ,'o', ms =2, label = "analytical")
plt.plot(get_grads(q0) ,'o', ms =10, label = "analytical")
plt.plot((WF_4pe- WF_4me)/(2*epsilon),'o', ms =5, label = "numerical")
plt.legend()
plt.show()






