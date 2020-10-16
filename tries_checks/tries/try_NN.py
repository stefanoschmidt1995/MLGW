###################
#	Some tries of fitting GW generation model using a NN
#	Apparently it works quite well
###################
import matplotlib.pyplot as plt 
import numpy as np
import sys
sys.path.insert(1, '../routines') #folder in which every relevant routine is saved

from GW_helper import * 	#routines for dealing with datasets
from ML_routines import *	#PCA model
from EM_MoE import *		#MoE model
import keras


folder = "../datasets/GW_TD_dataset_mtotconst/"

    #loading PCA datasets
N_train = None
train_theta = np.loadtxt(folder+"PCA_train_theta.dat")[:N_train,:]
test_theta = np.loadtxt(folder+"PCA_test_theta.dat")
PCA_train_ph = np.loadtxt(folder+"PCA_train_ph.dat")[:N_train,:]
PCA_test_ph = np.loadtxt(folder+"PCA_test_ph.dat")
K_PCA_to_fit = 7

	#adding extra features for basis function regression
new_features = ["00", "11","22", "01", "02", "12"
,"000", "001", "002", "011", "012", "022", "111", "112", "122", "222"
,"0000", "0001","0002", "0011", "0022","0012","0111","0112", "0122", "0222","1111", "1112", "1122", "1222", "2222"]

outfile = open("./saved_models_ph_NN/ph_feat", "w+")
outfile.write("\n".join(new_features))
outfile.close()

train_theta = add_extra_features(train_theta, new_features)
test_theta = add_extra_features(test_theta, new_features)

print(train_theta.shape, test_theta.shape)

print("Loaded "+ str(train_theta.shape[0]+test_theta.shape[0])+
      " data with ", PCA_train_ph.shape[1]," PCA components")
print("Spins are allowed to vary within domain [-0.8,0.8]x[-0.8,0.8]")


   #setting up an EM model for each component
NN_models = 	[]
load_list =	[]#	[0   ,1   ,2   ,3   ,4   ,5   ,6  ]  #indices of models to be loaded from file

#for 4-th only
N_epochs = 		[200  ,100  ,100  ,50  ,50  ,300  ,200  ]  #number of epocs


D = train_theta.shape[1] #number of independent variables

for k in range(0,K_PCA_to_fit):
	print("### Comp ", k)
		#useless variables for sake of clariness
	y_train = PCA_train_ph[:,k]
	y_test = PCA_test_ph[:,k]

	NN_models.append(keras.models.Sequential())
	NN_models[-1].add(keras.layers.Dense(1, input_shape=(train_theta.shape[1],), activation= "linear"))
	NN_models[-1].summary()

		# compile the model choosing optimizer, loss and metrics objects & fitting
	#opt = keras.optimizers.SGD(lr=.01, momentum=0.01, decay=0.1, nesterov=False)
	opt = 'rmsprop'

	if k in load_list:
		NN_models[-1] = keras.models.load_model("./saved_models_full_ph_NN/NN_"+str(k))
		print("Loaded model for comp: ", k)
	else:
		NN_models[-1].compile(optimizer=opt, loss= 'mse')
		history = NN_models[-1].fit(x=train_theta, y=y_train, batch_size=None, epochs=N_epochs[k], shuffle=True, verbose=1,  validation_data = (test_theta, y_test))
		NN_models[-1].save("./saved_models_ph_NN/NN_"+str(k))

		#doing some test
	y_pred = NN_models[-1].predict(test_theta)[:,0]
	y_pred_train = NN_models[-1].predict(train_theta)[:,0]
	print("Train square loss for comp "+str(k)+": ",np.mean(np.square(y_pred_train-y_train)) )
	print("Test square loss for comp "+str(k)+": ",np.mean(np.square(y_pred-y_test)) )

	for i in range(3):
		plt.figure(i+3*k, figsize=(20,10))
		plt.title("Component #"+str(k)+" vs q/s1/s2 | index "+str(i))
		plt.plot(test_theta[:,i], y_test, 'o', ms = 3,label = 'true')
		plt.plot(test_theta[:,i], y_pred, 'o', ms = 3, label = 'pred')
		plt.legend()
		if i ==0 or i==1 or i ==2:
			#pass
			plt.savefig("../pictures/PCA_comp_full_ph_NN/fit_"+str(k)+"_vs"+str(i)+".jpeg")
	plt.show()

