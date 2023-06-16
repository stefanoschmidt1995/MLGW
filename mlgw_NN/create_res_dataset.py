import sys
import os
sys.path.append('/home/tim.grimbergen/my_code_v2')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from sklearn.metrics import mean_squared_error
import PCAdata_v2
import NNmodel_v2

data_loc = "/home/tim.grimbergen/PCA_Data/8_8_HM_spin/22/"
save_loc_res = "/home/tim.grimbergen/PCA_Data/8_8_HM_spin_res/22/2_comp_PH/res1/"
model_loc = "/home/tim.grimbergen/ModelsV2/HM_spin_6compPH/22/6_comp_PH/ph_[20,20,15,15,15]/model.h5"

D = PCAdata_v2.PcaData(data_loc, 6, "ph", features=["chirp","2nd_poly","eff_spin","sym_mas","eff_spin_sym_mas_2nd_poly","eff_spin_sym_mas_3rd_poly","eff_spin_sym_mas_4th_poly", "log"])
M = NNmodel_v2.NeuralNetwork.load_model(model_loc)
pred_train = M.predict(D.train_theta)
pred_test = M.predict(D.test_theta)
NNmodel_v2.NeuralNetwork.CreateResidualPCAsets(D,(pred_train, pred_test),model_loc,save_loc_res,"ph", components=3)
