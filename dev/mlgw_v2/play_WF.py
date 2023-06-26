###################
#	Interactive plot to display in real time the WF dependence on the physical parameters.
###################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons

try:
	from GW_generator import *
except:
	from mlgw.GW_generator import *

gen = GW_generator() #loading the generator

	#initializing the plot
fig, ax = plt.subplots(figsize = (6.4*1.3,4.8), nrows = 2, ncols = 1)
plt.subplots_adjust(left=0.2, bottom=0.25, hspace = 0.4)
all_modes = [str(mode) for mode in gen.list_modes()]
modes = gen.list_modes()
t_0 = 0
t = np.linspace(-10**(t_0), .01, 1000)
m1_0 = 20.
m2_0 = 20.
s1_0 = 0.
s2_0 = 0.
dist_0 = 1.
iota_0 = 0.
scale_0 = -18.4 

h_p, h_c = gen.get_WF([m1_0, m2_0, s1_0, s2_0, dist_0, iota_0, 0.], t, modes = modes)
l_p, = ax[0].plot(t, h_p, lw=2)
l_c, = ax[1].plot(t, h_c, lw=2)

ax[0].set_ylim([-10**(scale_0),10**(scale_0)])
ax[1].set_ylim([-10**(scale_0),10**(scale_0)])
ax[0].set_xlim([-10**(t_0),0.01])
ax[1].set_xlim([-10**(t_0),0.01])
ax[0].set_ylabel(r"$h_+$")
ax[1].set_ylabel(r"$h_{\times}$")

	#setting the interactive sliders
axcolor = 'lightgoldenrodyellow'
ax_m1 = plt.axes([0.25, 0.15, 0.25, 0.03], facecolor=axcolor)
ax_m2 = plt.axes([0.25, 0.1, 0.25, 0.03], facecolor=axcolor)
ax_s1 = plt.axes([0.63, 0.15, 0.25, 0.03], facecolor=axcolor)
ax_s2 = plt.axes([0.63, 0.1, 0.25, 0.03], facecolor=axcolor)
ax_dist = plt.axes([0.25, 0.05, 0.25, 0.03], facecolor=axcolor)
ax_iota = plt.axes([0.63, 0.05, 0.25, 0.03], facecolor=axcolor)

s_m1 = Slider(ax_m1, r'$m_1/M_\odot$', 5., 100.0, valinit=m1_0, valstep=0.1)
s_m2 = Slider(ax_m2, r'$m_2/M_\odot$', 5., 100.0, valinit=m2_0, valstep = 0.1)
s_s1 = Slider(ax_s1, r'$s_1$', -0.8, 0.8, valinit=s1_0, valstep=0.01)
s_s2 = Slider(ax_s2, r'$s_2$', -0.8, 0.8, valinit=s2_0, valstep = 0.01)
s_dist = Slider(ax_dist, r'$d_L/Mpc$', 0.1, 10, valinit=dist_0, valstep=0.01)
s_iota = Slider(ax_iota, r'$\iota$', 0., np.pi, valinit=iota_0, valstep = 0.01)

ax_scale = plt.axes([0.058, .37, 0.03, 0.1], facecolor=axcolor)
s_scale = Slider(ax_scale, 'Scale', -20,-17, valinit= scale_0, valstep = 0.01, orientation = "vertical")

ax_time = plt.axes([0.058, .15, 0.03, 0.1], facecolor=axcolor)
s_time = Slider(ax_time, r'$\log(t_{min}/s)$', -1, 1.3, valinit= t_0, valstep = 0.01, orientation = "vertical")

ax_buttons = plt.axes([0.025, 0.55, 0.1, 0.35], facecolor=axcolor)
buttons = CheckButtons(ax_buttons, [mode for mode in all_modes], actives = [True for i in range(len(all_modes))])

def update(val):
	"Updates the plots with the values of physical paramters set by the sliders"
	m1 = s_m1.val
	m2 = s_m2.val
	s1 = s_s1.val
	s2 = s_s2.val
	dist = s_dist.val
	iota = s_iota.val
	h_p, h_c = gen.get_WF([m1, m2, s1, s2, dist, iota, 0.], t, modes = modes)
	l_p.set_ydata(h_p)
	l_c.set_ydata(h_c)
	fig.canvas.draw_idle()
	return


def update_modes(label):
	"Changes the modes to be included in the WFs"
	if buttons.get_status()[all_modes.index(label)]:
		modes.append(gen.list_modes()[all_modes.index(label)])
	else:
		modes.remove(gen.list_modes()[all_modes.index(label)])
	update(0.)
	return

def update_view(val):
	"Changes the y scale of the plot"
	scale = s_scale.val
	ax[0].set_ylim([-10**(scale),10**(scale)])
	ax[1].set_ylim([-10**(scale),10**(scale)])
	return

def update_time(val):
	"Changes the starting point of the time grid and updates the plot"
	global t
	t_min = -10**(s_time.val)
	t = np.linspace(t_min, 0.01, int(np.abs(t_min*1000.)))
	l_p.set_xdata(t)
	l_c.set_xdata(t)
	ax[0].set_xlim([t_min,0.01])
	ax[1].set_xlim([t_min,0.01])
	update(0.)
	return

	#activating the sliders
s_m1.on_changed(update)
s_m2.on_changed(update)
s_s1.on_changed(update)
s_s2.on_changed(update)
s_dist.on_changed(update)
s_iota.on_changed(update)

buttons.on_clicked(update_modes)
s_scale.on_changed(update_view)
s_time.on_changed(update_time)


plt.show()





