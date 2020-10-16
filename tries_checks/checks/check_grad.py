import sys
sys.path.insert(1, '../mlgw_v1')
import matplotlib.pyplot as plt
import numpy as np
import GW_generator as gen

###RAW WF WORKS WELL...
###let's see the rest!!

model = gen.GW_generator("/home/stefano/Desktop/Stefano/scuola/uni/tesi_magistrale/code/mlgw_v1/TD_model")

#std_q = lambda q: np.array([[q*20./(1+q),20./(1+q),.1,.3,1e-5,-.3,0.4]]) #q
#std_q = lambda q: np.array([[5.*q/6.,q/6,-0.1,.3,1e-5,2.4234,0.4]]) #total mass
std_q = lambda q: np.array([[15,45, q,-.63,1e-5,-.3,0.4],[20, 10,.1,.32,4,1.3,2.4]]) #spin
#std_q = lambda q: np.array([[3.,-.5, q]])
#std_q = lambda q: np.array([[45,23,-0.1,.3,q,1.4234,0.4]]) #distance
#std_q = lambda q: np.array([[23,53,-0.1,.3,0.2342,q,0.4]]) #iota
#std_q = lambda q: np.array([[45,23,-0.1,.3,0.2342,0.4,q]]) #phi

t = np.linspace(-4,0.05, 5000)
def get_WF(q):
	print(std_q(q))
	#amp, ph = model.get_WF(std_q(q), t, plus_cross = False)
	hp, hc = model.get_WF(std_q(q), t, plus_cross = True)
	#h= (hp+1j*hc)
	#amp, ph = model.get_raw_WF(std_q(q))
	return hp[0,:]

def get_grads(q):
	#g_amp, g_ph = model._GW_generator__grads_theta(std_q(q0),t)
	#g_hp, g_hc = model._GW_generator__grads_theta(std_q(q0),t)
	g_hp, g_hc = model.get_grads(std_q(q0),t)
	#g_amp, g_ph = model.get_raw_grads(std_q(q))
	return g_hp[0,:,2]

def get_red_coeff(q):
	g_amp, g_ph = model.get_red_coefficients(std_q(q))
	return g_ph[0,:]

epsilon = .0000005
q0= .1
WF_4pe = get_WF(q0+epsilon)
WF_4me = get_WF(q0-epsilon)
#WF_4pe = get_red_coeff(q0+epsilon)
#WF_4me = get_red_coeff(q0-epsilon)

#plt.plot(model.get_time_grid(), (WF_4pe- WF_4me)/(2*epsilon),'o', ms =2, label = "numerical")
#plt.plot(model.get_time_grid(), get_grads(q0) ,'o', ms =2, label = "analytical")
#plt.plot(t, get_WF(q0) ,'o', ms =2, label = "WF")
plt.plot(t,get_grads(q0) ,'o', ms =2, label = "analytical")
plt.plot(t,(WF_4pe- WF_4me)/(2*epsilon),'o', ms =2, label = "numerical")
plt.legend()
plt.show()






