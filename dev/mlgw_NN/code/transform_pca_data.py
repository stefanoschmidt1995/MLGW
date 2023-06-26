import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from PCAdata_v2 import PcaData

old_data_loc = "/home/tim.grimbergen/PCA_Data/15_30_bigonlyq_highq/22/"
pca_model_loc = "/home/tim.grimbergen/PCA_Data/5_7_bigonlyq_unif/22/"
save_loc = "/home/tim.grimbergen/PCA_Data/bigonlyq_highq_unifPCA/22/"

PcaData.ConvertPcaData(old_data_loc, pca_model_loc, save_loc)

#save_loc_2 = "/home/tim.grimbergen/PCA_Data/Merged_sets/onlyq_lowq_unif_unifPCA/22/"

#PCAdata_v2.PcaData.MergePCAsets(save_loc, pca_model_loc, save_loc_2)
