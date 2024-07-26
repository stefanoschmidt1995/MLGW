from mlgw.GW_helper import load_dataset, make_set_split
import matplotlib.pyplot as plt
import mlgw
from mlgw.ML_routines import PCA_model
from mlgw.NN_model import CustomLoss, Schedulers
import numpy as np

from scipy.signal import hilbert

dataset_file = 'tiny_dataset_only_q_fref_eq_fstart_const.dat'
theta, alpha_dataset, cosbeta_dataset, time_grid = load_dataset(dataset_file, N_data=None, N_entries = 2, N_grid = None, shuffle = False, n_params = 9)

angle_name = 'alpha'
dirname = 'pca_model_1d_f_ISCO_fstart_const'

id_ = np.random.randint(0, len(alpha_dataset))

pca = PCA_model()
pca.load_model('{}/pca_{}'.format(dirname, angle_name))
alpha_dataset_first_PC = pca.reconstruct_data(pca.reduce_data(alpha_dataset)[:,:2])


time_grid_linspace = np.linspace(time_grid[0], time_grid[-1], 10000)
alpha_dataset = np.interp(time_grid_linspace, time_grid, alpha_dataset[id_])
alpha_dataset_first_PC = np.interp(time_grid_linspace, time_grid, alpha_dataset_first_PC[id_])

analytic_signal = hilbert(alpha_dataset)

plt.figure()
plt.plot(time_grid_linspace, alpha_dataset, label = 'alpha true')
plt.plot(time_grid_linspace, analytic_signal.real, label = 'alpha hilbert')
plt.plot(time_grid_linspace, alpha_dataset_first_PC, label = 'alpha PCA')
plt.legend()

plt.figure()
plt.title('diff')
plt.plot(time_grid_linspace, alpha_dataset-alpha_dataset_first_PC)

plt.show()

