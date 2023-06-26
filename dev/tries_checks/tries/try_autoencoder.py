###################
#	Trying to reduce data dimensionality using an DNN autoeconder
#	Does it work??
###################

from GW_helper import *
import matplotlib.pyplot as plt
from ML_routines import *
import keras
import keras.backend as K
from keras.layers import Input, Conv1D, MaxPooling1D, AveragePooling1D, Dense, Reshape, UpSampling1D, Flatten

#theta_vector, amp_dataset, ph_dataset, frequencies = create_dataset(10000, filename = "GW_dataset_low.gz", N_grid=256, q_max =16, spin_mag_max = 0.88, f_step=.01, f_high = 1024, f_min = None, f_max = 50)
theta_vector, amp_dataset, ph_dataset, frequencies = load_dataset("GW_dataset_high.gz")
print("Dataset loaded with "+str(theta_vector.shape[0])+" data")
#quit()
	#splitting into train and test set
	#to make data easier to deal with
train_frac = .85
ph_scale_factor = 1.#np.std(ph_dataset) #phase must be rescaled back before computing mismatch index beacause F strongly depends on an overall phase... (why so strongly????)

min_ph = np.min(ph_dataset)
max_ph = np.max(ph_dataset)
ph_dataset = (ph_dataset -min_ph)/np.abs(max_ph-min_ph)

train_theta, test_theta, train_amp, test_amp = make_set_split(theta_vector, amp_dataset, train_frac, 1e-21)
train_theta, test_theta, train_ph, test_ph = make_set_split(theta_vector, ph_dataset, train_frac, ph_scale_factor)

		#FITTING AN AUTOENCODER
print("#####AutoEncoder#####")
N_epochs = 30

	#phase
	#doing a keras model to make things better...
inputs = Input(shape = (train_ph.shape[1],1)) #(.,256,1)
x = Conv1D(8,3, activation='sigmoid', padding='same', strides=2)(inputs) #(.,128,8)
x1 = MaxPooling1D(2)(x) #(.,64,8)
x2 = Conv1D(4,3, activation='sigmoid', padding='same',dilation_rate=2)(x1) #(.,64,4)
x3 = MaxPooling1D(2)(x2) #(.,32,4)
x4 = AveragePooling1D()(x3) #(.,16,4)
flat = Flatten()(x4) #(.,64)
encoded = Dense(8, activation='sigmoid')(flat) #(.,8)
d1 = Dense(64)(encoded) #(.,64)
d2 = Reshape((16,4))(d1) #(.,16,4)
d3 = Conv1D(4,1,strides=1, activation='sigmoid', padding='same')(d2) #(.,16,4)
d4 = Conv1D(1,1,strides=1, activation='sigmoid', padding='same')(d3) #(.,16,1)
d5 = Dense(256)(d4)
d6 = MaxPooling1D(4)(d5)
d7 = MaxPooling1D(4)(d6)
decoded = Reshape((256,1))(d7)

ph_model= keras.Model(inputs, decoded)
ph_model.summary()
#quit()
	# compile the model choosing optimizer, loss and metrics objects & fitting
#	opt = keras.optimizers.SGD(lr=0.01, momentum=0.01, decay=0.1, nesterov=False)
opt = 'adagrad'
ph_model.compile(optimizer=opt, loss='mean_squared_error')#, metrics=['mse'])
train_data = np.reshape(train_ph, (train_ph.shape[0],train_ph.shape[1],1))
test_data = np.reshape(test_ph, (test_ph.shape[0],test_ph.shape[1],1))
history = ph_model.fit(x=train_data, y=train_data, batch_size=64, epochs=N_epochs, shuffle=True, verbose=1, validation_split=0.1)

ph_loss = ph_model.evaluate(test_data, test_data)
print("Loss for NN: ", ph_loss)

#pred_ph = ph_model.predict(test_data)*ph_scale_factor
#test_data = test_data * ph_scale_factor

test_data = (test_data *np.abs(max_ph-min_ph)) + min_ph
pred_ph = (ph_model.predict(test_data) *np.abs(max_ph-min_ph)) + min_ph

plt.figure(4)
plt.title("Phase with autoencoder")
for i in range(1):
	print(frequencies)
	plt.plot(frequencies, test_data[i,:,0], label = 'true')
	plt.plot(frequencies, pred_ph[i,:,0], label = 'autoencoder')
plt.legend()


#computing mismatch
F = compute_mismatch(test_amp, test_data[:,:,0], test_amp, pred_ph[:,:,0])
#F = compute_mismatch(test_amp, test_ph, rec_fit_amp, test_ph) #to compute amp mismatch
print("Mismatch fit: ",F)
print("Mismatch fit avg: ",np.mean(F))

plt.show()
quit()

	#my version
x = Conv1D(8,3, activation='relu', padding='same', dilation_rate=2)(input_sig) #(1, ..., 8)
x1 = MaxPooling1D(pool_size = 10)(x) # #(1,...,8)
x2 = Conv1D(2,3, activation='relu', padding='same', dilation_rate=2)(x1) #(1, ..., 2)
x3 = Flatten()(x2) #(1,...,1)
encoded = Dense(10)(x2) #(1,...,1)
d1 = Dense(64)(encoded) #(1,...,1)
d2 = Conv1D(4,1,strides=1, activation='relu', padding='same')(d1) #(1,...,4)
d3 = Conv1D(1,1,strides=1, activation='linear', padding='same')(d2) #(1,64,1)
d4 = Dense(10)(d3)

	#online one
x = Conv1D(8,3, activation='relu', padding='same',dilation_rate=2)(input_sig)
x1 = MaxPooling1D(2)(x)
x2 = Conv1D(4,3, activation='relu', padding='same',dilation_rate=2)(x1)
x3 = MaxPooling1D(2)(x2)
x4 = AveragePooling1D()(x3)
flat = Flatten()(x4)
encoded = Dense(5)(flat)
d1 = Dense(50)(encoded)
d2 = Reshape((10,5))(d1)
d3 = Conv1D(4,1,strides=1, activation='relu', padding='same')(d2)
d4 = UpSampling1D(5)(d3)
d5 = Conv1D(8,1,strides=1, activation='relu', padding='same')(d4)
d6 = UpSampling1D(3)(d5)
d7 = UpSampling1D(3)(d6)
decoded = Conv1D(1,1,strides=1, activation='relu', padding='valid')(d7)

	#simple version
inputs = Input(shape=(train_ph.shape[1],1), name='input')
x = Flatten(name = 'flattened_cat')(inputs) #turn everything into a vector
x = Dense(10, activation='relu', name='encoder')(x)
x = Dense(100, activation='relu', name='middle_layer')(x)
x = Dense(train_ph.shape[1], activation='linear', name='decoder')(x)
outputs = Reshape((train_ph.shape[1],1))(x)



