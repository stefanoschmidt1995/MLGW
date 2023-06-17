import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sys.path.append('/home/tim.grimbergen/my_code_v2')
from NNmodel_v2 import NeuralNetwork
import PCAdata_v2



data_loc = "/home/tim.grimbergen/PCA_Data/8_8_HM_spin/"
save_loc = "/home/tim.grimbergen/ModelsV2/HM_spin_3456PH/"

param_dict = {'layer_list' : [[20,20,15,15]],
              'optimizers' : [("Nadam",0.002)],
              'activation' : [],
              'batch_size' : [64],
              'schedulers' : [PCAdata_v2.Schedulers('exponential',exp=-0.0005)],
              'loss_functions' : [['custom_mse', [12,3,1,1],0,2]]}
              
modes = ["21"]
q = ['ph']
c = [[2,3,4,5]]

for mode in modes:
    for q_ in q:
        for c_ in c:
            cur_param_dict = param_dict.copy()
            if isinstance(c_, list):
                c_str = "["
                for x in c_:
                    c_str+=str(x)+","
                c_str = c_str[:-1]+"]"
            else:
                c_str = str(c_)
            cur_save_loc = save_loc+mode+"/"+c_str+"_comp_"+q_.upper()+"/"
            cur_data_loc = data_loc+mode+"/"
            NeuralNetwork.HyperParametertesting(cur_data_loc, cur_save_loc, q_, 
                    cur_param_dict, c_, epcs = 5000, feat=["chirp","log","eff_spin","sym_mas","2nd_poly",
                                                           "eff_spin_sym_mas_2nd_poly", "eff_spin_sym_mas_3rd_poly",
                                                           "eff_spin_sym_mas_4th_poly"])

#["chirp", "log","eff_spin","sym_mas","2nd_poly",
#"eff_spin_sym_mas_2nd_poly","eff_spin_sym_mas_3rd_poly",
#"eff_spin_sym_mas_4th_poly"]
