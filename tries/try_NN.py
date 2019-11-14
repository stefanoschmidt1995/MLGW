###################
#	Some tries of fitting GW generation model using PCA+ logistic regression
#	Apparently it works quite well
###################

import sys
sys.path.insert(1, '/home/stefano/Desktop/Stefano/scuola/uni/tesi_magistrale/code/routines')
from GW_helper import *
import matplotlib.pyplot as plt
from ML_routines import *
import keras

theta_vector, amp_dataset, ph_dataset, frequencies = load_dataset("../datasets/GW_std_dataset.dat", shuffle = False) #loading dataset
PCA_train_ph = np.loadtxt("../datasets/PCA_train.dat")
PCA_test_ph = np.loadtxt("../datasets/PCA_test.dat")

	#adding extra features for non linear regression
#extra_features = np.stack((np.multiply(theta_vector[:,0], theta_vector[:,0]), np.multiply(theta_vector[:,0], theta_vector[:,1]),  np.multiply(theta_vector[:,1], theta_vector[:,2])))
extra_features = np.reshape(np.power(theta_vector[:,0], -1), (theta_vector.shape[0],1))
theta_vector = np.concatenate((theta_vector, extra_features), axis = 1)

train_frac = .8

train_theta, test_theta, train_amp, test_amp = make_set_split(theta_vector, amp_dataset, train_frac, 1e-21)
train_theta, test_theta, train_ph, test_ph   = make_set_split(theta_vector, ph_dataset, train_frac, 1.)

del amp_dataset, ph_dataset, theta_vector

print("Loaded "+ str(train_theta.shape[0]+test_theta.shape[0])+" data with ",PCA_test_ph.shape[1]," features")

		#DOING PCA
print("#####PCA#####")
K_ph = PCA_train_ph.shape[1]
ph_PCA = PCA_model()
ph_PCA.load_model("../datasets/PCA_std_model.dat")

rec_PCA_test_ph = ph_PCA.reconstruct_data(PCA_test_ph) #reconstructed data for phase
error_ph = np.linalg.norm(test_ph - rec_PCA_test_ph, ord= 'fro')/(test_ph.shape[0]*np.std(test_ph))
print("Reconstruction error for phase with PCA: ",error_ph)

F_PCA = compute_mismatch(test_amp, test_ph, test_amp, rec_PCA_test_ph)
print("Mismatch PCA avg: ",np.mean(F_PCA))

	#preprocessing data
max_ph = np.max(np.abs(PCA_train_ph), axis = 0)
PCA_train_ph = np.divide(PCA_train_ph,max_ph)
PCA_test_ph = np.divide(PCA_test_ph,max_ph)



		#FITTING WITH NN
print("#####NN#####")
N_epochs = 80

	#phase

	#doing a keras model to make things better...
model = keras.models.Sequential()
model.add(keras.layers.Dense(8, input_shape=(train_theta.shape[1],), activation= "linear"))
model.add(keras.layers.Dense(32,activation = 'relu'))
model.add(keras.layers.Dense(64,activation = 'relu'))
#model.add(keras.layers.Dense(64,activation = 'relu'))
model.add(keras.layers.Dense(32,activation = 'relu'))
model.add(keras.layers.Dense(1, activation = "linear"))
model.summary()

	# compile the model choosing optimizer, loss and metrics objects & fitting
#	opt = keras.optimizers.SGD(lr=0.01, momentum=0.01, decay=0.1, nesterov=False)
opt = 'rmsprop'
	#pre-fitting with mse
model.compile(optimizer=opt, loss= 'mse')
history = model.fit(x=train_theta, y=PCA_train_ph[:,0], batch_size=64, epochs=50, shuffle=True, verbose=1)

mse_train = (np.mean(np.square(model.predict(train_theta) - PCA_train_ph[:,0])))#/train_theta.shape[0]

print("train model loss ", model.evaluate(train_theta, PCA_train_ph[:,0], verbose=0), mse_train)
print("test model loss ", model.evaluate(test_theta, PCA_test_ph[:,0], verbose =0))

for i in range(3):
	plt.figure(i+3)
	comp = 0
	plt.title("Data lowest components #"+str(comp)+" vs param "+str(i))
	plt.plot(test_theta[:,i], PCA_test_ph[:,comp], 'o',label = 'fitted', ms = 4)
	plt.plot(test_theta[:,i], model.predict(test_theta), 'o',label = 'true', ms = 4)
	#plt.plot(test_theta[:,i], red_fit_ph[:,1]-red_test_ph[:,1], 'o',label = 'difference')
	plt.legend()

plt.show()


quit()

print("Doing test")
		#un_preprocessing data
PCA_fit_ph = np.multiply(model.predict(),max_ph[0])
PCA_test_ph = np.multiply(PCA_test_ph,max_ph)


red_fit_ph = logreg_ph.un_preprocess_data(model.predict(test_theta)) #for single model
red_test_ph = logreg_ph.un_preprocess_data(red_test_ph) #un-preprocessing test labels

#plt.quit()
error_ph = np.linalg.norm(red_test_ph - red_fit_ph, ord= 'fro')/(test_ph.shape[0]*np.std(red_test_ph))
print("Fit reconstruction error for reduced coefficients: ", error_ph)

rec_fit_ph = ph_PCA.reconstruct_data(red_fit_ph)
error_ph = np.linalg.norm(test_ph - rec_fit_ph, ord= 'fro')/(test_ph.shape[0]*np.std(test_ph))
print("Fit reconstruction error for phase: ", error_ph)

plt.figure(2)
plt.title("Phase with FIT")
for i in range(2):
	plt.plot(frequencies, test_ph[i,:], label = 'true |' + str(np.round(test_theta[i,0],2))+","+ str(np.round(test_theta[i,1],2))+","+ str(np.round(test_theta[i,2],2)))
	plt.plot(frequencies, rec_fit_ph[i,:], label = 'fit')
plt.legend()

F = compute_mismatch(train_amp[0,:], test_ph, train_amp[0,:], rec_fit_ph) #ty if it's the same F as test!!!
#F = compute_mismatch(test_amp, test_ph, test_amp, rec_fit_ph)
#F = compute_mismatch(np.ones(test_ph.shape), test_ph, np.ones(test_ph.shape), rec_fit_ph)
print("Mismatch fit avg: ",np.mean(F))

plt.show()
quit()

	#plotting principal components
plt.figure(5)
plt.title("Amplitude principal components")
for i in range(amp_PCA.get_V_matrix().shape[1]):
	plt.plot(frequencies, amp_PCA.get_V_matrix()[:,i], label=str(i))
plt.legend()

plt.figure(6)
plt.title("Phase principal components")
for i in range(ph_PCA.get_V_matrix().shape[1]):
	plt.plot(frequencies, ph_PCA.get_V_matrix()[:,i], label=str(i))
plt.legend()

plt.show()









