"""
To generate a mismatch dictionary

	python test_mismatch_analytical_approx.py --n-points 100 --t-mismatch 0.05 --out-file analytical_angles_mismatch_cut_before_merger.json
	python test_mismatch_analytical_approx.py --n-points 100 --t-mismatch 0.0 --out-file analytical_angles_mismatch_cut_t_zero.json

To plot the generated files:
	python3 test_mismatch_analytical_approx.py --out-file analytical_angles_mismatch_cut_before_merger.json
	python3 test_mismatch_analytical_approx.py --out-file analytical_angles_mismatch_cut_t_zero.json
"""


import numpy as np
import matplotlib.pyplot as plt
import mlgw
from mlgw.GW_helper import load_dataset, make_set_split
from mlgw.ML_routines import PCA_model
from mlgw.GW_helper import compute_optimal_mismatch
from mlgw.precession_helper import angle_manager
import json
from tqdm import tqdm
import argparse
from pathlib import Path
import os

def scalar_prod(a, b):
	a, b = np.fft.fft(a), np.fft.fft(b)
	return 1-np.abs(np.vdot(a,b)/np.sqrt(np.vdot(a,a)*np.vdot(b,b)))

###################################################################################################

parser = argparse.ArgumentParser(description='Computes the match between the approximated version and the true version of the angles')
parser.add_argument('--n-points', type=int, required = False, default = 100,
	help='Number of points to populate the histogram with')
parser.add_argument('--job-id', type=int, required = False, default = None,
	help='Id for the job, if given, it will change the seed and append the job_id to the output file')
parser.add_argument('--out-file', type=str, required = False, default = 'analytical_angles_mismatch_results.json',
	help='JSON file to write the output')
parser.add_argument('--t-mismatch', type=float, required = False, default = 0.05,
	help='Time to merger after which you want to stop the match calculation')
args = parser.parse_args()

gen = mlgw.GW_generator()
modes = [(2,2), (2,1), (3,3), (4,4), (5,5)]

verbose = True
show_plots = not False


if isinstance(args.job_id, int):
	np.random.seed(args.job_id)
	#filename = Path(args.out_file).stem
	#out_file = args.out_file.replace(filename, filename+'_{}'.format(args.job_id))

out_file = args.out_file

load_json = os.path.exists(args.out_file) and args.job_id is None

res_dict = {
	'theta': [],
	't0': []
}
for m in modes:
	res_dict['mismatch_approx_{}{}'.format(*m)] = []
	res_dict['mismatch_ML_{}{}'.format(*m)] = []

for i in tqdm(range(args.n_points), disable = verbose):
	if load_json:
		with open(out_file, 'r') as f:
			res_dict = json.load(f)
		break

	q = np.random.uniform(1,10)
	s1, s2 = np.random.uniform(0,1,2)
	t1, t2 = np.arccos(np.random.uniform(-1,1, 2))
	phi1, phi2 = np.random.uniform(0, 2*np.pi, 2)
	
	M = 20.
	m1, m2 = q*M/(1+q), M/(1+q)
	s1x, s1y, s1z = s1*np.sin(t1)*np.cos(phi1), s1*np.sin(t1)*np.sin(phi1), s1*np.cos(t1)
	s2x, s2y, s2z = s2*np.sin(t2)*np.cos(phi2), s2*np.sin(t2)*np.sin(phi2), s2*np.cos(t2)
	theta = [m1, m2, s1x, s1y, s1z, s2x, s2y, s2z]

	t0 = np.random.uniform(5, 40)
	t0 = 40
	t_grid = np.linspace(-t0, 0.01, int(t0+0.01)*4096)
	manager = angle_manager(gen, t_grid, 5,5, beta_residuals = not True)
	
	hlm_p, hlm_c, alpha_IMR, beta_IMR, gamma_IMR =  gen.get_twisted_modes(theta, t_grid, modes, f_ref = np.nan, extra_stuff = 'IMR_angles')
	hlm_p_approx, hlm_c_approx, alpha_approx, beta_approx, gamma_approx =  gen.get_twisted_modes(theta, t_grid, modes, f_ref = np.nan, extra_stuff = manager)
	hlm_p_ML, hlm_c_ML, alpha_ML, beta_ML, gamma_ML =  gen.get_twisted_modes(theta, t_grid, modes, f_ref = np.nan)

	if verbose: print("#########\nParams: {}".format(theta))
	res_dict['theta'].append(theta)
	res_dict['t0'].append(t0)
	
	id_mismatch = np.argmin(np.square(t_grid + args.t_mismatch))
	
	for i, mode in enumerate(modes):
		mismatch_approx, _ = compute_optimal_mismatch(
				hlm_p[:id_mismatch,i]+1j*hlm_c[:id_mismatch,i],
				hlm_p_approx[:id_mismatch,i]+1j*hlm_c_approx[:id_mismatch,i],
				optimal = True, return_F = True)
		res_dict['mismatch_approx_{}{}'.format(*mode)].append(float(mismatch_approx))
		mismatch_ML, _ = compute_optimal_mismatch(
				hlm_p[:id_mismatch,i]+1j*hlm_c[:id_mismatch,i],
				hlm_p_ML[:id_mismatch,i]+1j*hlm_c_ML[:id_mismatch,i],
				optimal = True, return_F = True)
		res_dict['mismatch_ML_{}{}'.format(*mode)].append(float(mismatch_ML))
		
		if verbose: print("Mode ({},{}) - {} {}".format(*mode, mismatch_approx, mismatch_ML))
		#print(scalar_prod(hlm_p[:id_mismatch,i], hlm_p_approx[:id_mismatch,i]))

	if show_plots:

		fig, axes = plt.subplots(3,1, figsize = (6.4, 4.8*3/2), sharex = True)
		axes[0].set_title('alpha')
		axes[0].plot(t_grid, alpha_IMR[0], label = 'true')
		axes[0].plot(t_grid, alpha_approx[0], label = 'approx')
		axes[0].plot(t_grid, alpha_ML[0], label = 'ML')
		axes[1].set_title('beta')
		axes[1].plot(t_grid, beta_IMR[0], label = 'true')
		axes[1].plot(t_grid, beta_approx[0], label = 'approx')
		axes[1].plot(t_grid, beta_ML[0], label = 'ML')
		axes[2].set_title('gamma')
		axes[2].plot(t_grid, gamma_IMR[0], label = 'true')
		axes[2].plot(t_grid, gamma_approx[0], label = 'approx')
		axes[2].plot(t_grid, gamma_ML[0], label = 'ML')
		plt.legend()
		plt.tight_layout()

		for i, mode in enumerate(modes):
			plt.figure()
			plt.title('(l,m) = ({},{})'.format(*mode))
			plt.plot(t_grid, hlm_p[:,i], label = 'true')
			plt.plot(t_grid, hlm_p_approx[:,i], label = 'approx')
			plt.plot(t_grid, hlm_p_ML[:,i], label = 'ML')
			plt.legend()
		plt.tight_layout()
		plt.show()

if not load_json:
	with open(out_file, 'w') as f:
		json.dump(res_dict, f)


res_dict['theta'] = np.array(res_dict['theta'])

fig, axes = plt.subplots(len(modes),1, figsize = (6.4, 4.8*len(modes)/2), sharex = True)
plt.suptitle(args.out_file, fontsize = 8)
n_bins = 2*int(np.sqrt(len(res_dict['theta'])))
for mode, ax in zip(modes, axes):
	log_mismatch_ML = np.log10(np.array(res_dict['mismatch_ML_{}{}'.format(*mode)])+1e-10)
	log_mismatch_approx = np.log10(np.array(res_dict['mismatch_approx_{}{}'.format(*mode)])+1e-10)
	ax.set_title('(l,m) = ({},{})'.format(*mode))

		#approx mismatches
	ax.hist(log_mismatch_approx, color = 'orange', alpha = 0.5, bins = n_bins)
	ax.axvline(np.nanmedian(log_mismatch_approx), ls = 'dotted', c= 'k')
	ax.axvline(np.nanpercentile(log_mismatch_approx, 90), ls = 'dotted', c= 'k')

		#ML mismatches
	ax.hist(log_mismatch_ML, color = 'blue', bins = n_bins, histtype = 'step')
	ax.axvline(np.nanmedian(log_mismatch_ML), ls = '--', c= 'k')
	ax.axvline(np.nanpercentile(log_mismatch_ML, 90), ls = '--', c= 'k')
plt.xlabel(r'$log_{10}(1-\mathcal{M})$')
plt.xlim([-8,0.5])
plt.tight_layout()

plt.show()
quit()

xs = [res_dict['theta'][:,0]/res_dict['theta'][:,1],
	np.linalg.norm(res_dict['theta'][:,[2,3,4]], axis = 1),
	np.linalg.norm(res_dict['theta'][:,[5,6,7]], axis = 1),
	np.arccos(res_dict['theta'][:,2]/res_dict['theta'][:,4])]

for x, label in zip(xs, ['q','s1','s2', 't1']):
	plt.figure()
	for mode in modes:
		plt.scatter(x, np.log10(res_dict['mismatch_ML_{}{}'.format(*mode)]), s = 5, label = '{}{}'.format(*mode))
	plt.legend()
	plt.xlabel(label)

plt.show()


