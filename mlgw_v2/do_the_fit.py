from fit_model import *

lm = "44"

dataset_file = "TD_datasets/{}_dataset.dat".format(lm)
shift_file = "TD_datasets/shift_dataset.dat".format(lm)
PCA_dataset_folder = "TD_datasets/{}".format(lm)
model_folder = "TD_models/model_0/{}".format(lm)
shift_folder = "TD_models/model_0/{}/shifts".format(lm)

fit_PCA = False
fit_MoE_model = True
fit_shifts_ = False

fifth_order = ["00", "11","22", "01", "02", "12","000", "001", "002", "011", "012", "022", "111", "112", "122", "222", #2nd/3rd order
"0000", "0001","0002", "0011", "0022","0012","0111","0112", "0122", "0222","1111", "1112", "1122", "1222", "2222", #4th order
"00000", "00001", "00002", "00011", "00012", "00022", "00111", "00112","00122", "00222", #5th order
"01111", "01112", "01122", "01222", "02222", "11111", "11112", "11122","11222", "12222", "22222"] #5th order

fourth_order = ["00", "11","22", "01", "02", "12","000", "001", "002", "011", "012", "022", "111", "112", "122", "222",
"0000", "0001","0002", "0011", "0022","0012","0111","0112", "0122", "0222","1111", "1112", "1122", "1222", "2222"]

if fit_PCA:
	create_PCA_dataset((4,5), dataset_file, PCA_dataset_folder, train_frac = 0.8)

if fit_MoE_model:
	print("Saving model to: ", model_folder)
	print("Fitting phase")
	fit_MoE("ph", PCA_dataset_folder, model_folder, experts = 4, comp_to_fit = None, features = fifth_order, EM_threshold = 1e-2, args = None, N_train = 6000, verbose = False, test_mismatch = True)
	print("Fitting amplitude")
	fit_MoE("amp", PCA_dataset_folder, model_folder, experts = 4, comp_to_fit = None, features = fifth_order, EM_threshold = 1e-2, args = 	None, N_train = 6000, verbose = False, test_mismatch = True)

if fit_shifts_:
	fit_shifts(shift_file, shift_folder, experts = 4, line_to_fit = 2, train_frac = 0.8, features = fourth_order, EM_threshold = 1e-2, args = None, N_train = None, verbose = True, train_mse = True, test_mse = True)

quit()
