import numpy as np
import matplotlib.pyplot as plt

from precession_helper import *

import sys
sys.path.insert(0,'../../mlgw_v2')
from GW_helper import *

#alpha, beta = get_alpha_beta(1.4, .4, .14, .75, 1.34, 3., 500., times)

#create_dataset_alpha_beta(50, "angles.dat", 10000, 1000, (1.1,10.))

params, alpha, beta, times = load_dataset("angles.dat", n_params = 6)

print(times)

plt.plot(times, alpha.T)

plt.figure()
plt.plot(times, beta.T)

plt.show()


