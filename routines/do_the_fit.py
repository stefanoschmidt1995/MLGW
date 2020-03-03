from fit_model import *

dataset_file = "/home/stefano/Desktop/Stefano/scuola/uni/tesi_magistrale/code/datasets/GW_TD_dataset/GW_TD_dataset.dat"
PCA_dataset_folder = "/home/stefano/Desktop/Stefano/scuola/uni/tesi_magistrale/code/datasets/GW_TD_dataset"
model_folder = "/home/stefano/Desktop/Stefano/scuola/uni/tesi_magistrale/code/definitive_code/TD_model"

fit_PCA = False
fit_MoE_model = True

fifth_order = ["00", "11","22", "01", "02", "12","000", "001", "002", "011", "012", "022", "111", "112", "122", "222", #2nd/3rd order
"0000", "0001","0002", "0011", "0022","0012","0111","0112", "0122", "0222","1111", "1112", "1122", "1222", "2222", #4th order
"00000", "00001", "00002", "00011", "00012", "00022", "00111", "00112","00122", "00222", #5th order
"01111", "01112", "01122", "01222", "02222", "11111", "11112", "11122","11222", "12222", "22222"] #5th order

fourth_order = ["00", "11","22", "01", "02", "12","000", "001", "002", "011", "012", "022", "111", "112", "122", "222",
"0000", "0001","0002", "0011", "0022","0012","0111","0112", "0122", "0222","1111", "1112", "1122", "1222", "2222"]

if fit_PCA:
	create_PCA_dataset((4,4), dataset_file, PCA_dataset_folder, train_frac = 0.89)

if fit_MoE_model:
	print("Saving model to: ", model_folder)
	fit_MoE("amp", PCA_dataset_folder, model_folder, experts = 4, comp_to_fit = None, features = fourth_order, EM_threshold = 1e-2, args = None, N_train = 6000, verbose = True, test_mismatch = True)

quit()
