import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from sklearn.metrics import mean_squared_error
from PCAdata_v2 import PcaData
from NNmodel_v2 import NeuralNetwork

data_loc = "/home/tim.grimbergen/PCA_Data/8_8_HM_spin/22/" #provide data location on which model is trained
save_loc_res = "/home/tim.grimbergen/PCA_Data/8_8_HM_spin_res/22/2_comp_PH/res1/" #provide location to save residual dataset to
model_loc = "/home/tim.grimbergen/ModelsV2/HM_spin_6compPH/22/6_comp_PH/ph_[20,20,15,15,15]/model.h5" #provide model to create residual dataset of

D = PcaData(data_loc, 6, "ph", features=["chirp","2nd_poly","eff_spin","sym_mas","eff_spin_sym_mas_2nd_poly","eff_spin_sym_mas_3rd_poly","eff_spin_sym_mas_4th_poly", "log"])
M = NeuralNetwork.load_model(model_loc)
pred_train = M.predict(D.train_theta)
pred_test = M.predict(D.test_theta)
NeuralNetwork.CreateResidualPCAsets(D,(pred_train, pred_test),model_loc,save_loc_res,"ph", components=3)
