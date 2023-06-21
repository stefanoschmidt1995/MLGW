import sys
import math

from NNmodel_v2 import NeuralNetwork
from PCAdata_v2 import Schedulers

data_loc = "C:/Users/timgr/Documents/Scriptie/Natuurkunde/Code/Dataset/8_8_HM_spin/res_22_2comp/res1/" #provide location to residual data
save_loc = "C:/Users/timgr/Documents/Scriptie/Natuurkunde/Code/ModelsV2_cluster/spin/HM_spin_1.1/22/res_PH_comp2/res1/" #provide location to save model to

param_dict = {'layer_list' : [[15,15,15,15]],
              'optimizers' : [("Nadam",0.0006)],
              'activation' : ['sigmoid'],
              'batch_size' : [128],
              'schedulers' : [PCAdata_v2.Schedulers('exponential',exp=-0)],
              'loss_functions' : [('custom_mse',[10,2],0,2)]}

NeuralNetwork.HyperParametertesting(data_loc, save_loc, "ph", param_dict, 2, epcs=6000, feat=["chirp","2nd_poly","eff_spin","sym_mas","eff_spin_sym_mas_2nd_poly","eff_spin_sym_mas_3rd_poly","eff_spin_sym_mas_4th_poly"])
