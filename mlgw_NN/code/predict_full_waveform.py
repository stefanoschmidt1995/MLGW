import sys
import os
sys.path.append("/home/tim.grimbergen/my_code_v2")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time
from sklearn.metrics import mean_squared_error #not used but need for error resolve
import numpy as np
from GW_generator_NN import GW_generator_NN
from GW_helper import generate_waveform, compute_optimal_mismatch
import matplotlib.pyplot as plt
import tensorflow as tf
tf.compat.v1.disable_eager_execution() #to speed up WF predictions (~4x speed up!)

model_loc = "/home/tim.grimbergen/full_models_test/test_full/"
data_loc = "/home/tim.grimbergen/full_test_wf/test1/"
save_loc = "/home/tim.grimbergen/full_test_wf/test1/" #Save the time it took to generate this dataset!

times = np.linspace(-2, 0.02, 10000)
params = np.genfromtxt(data_loc+"parameters")

batch_size = 1
N = 30

gen_NN = GW_generator_NN(folder = model_loc, frozen=False, batch_size = batch_size)
print('enterting the loop')

t = time.time()
for i in range(N//batch_size): 
    h_p_pred, h_c_pred = gen_NN.get_WF(params[i*batch_size:(i+1)*batch_size],
                                       times,
                                       [(2,2),(3,3),(2,1),(4,4),(5,5)])

print("time taken = " + str(time.time()-t))
    
    
