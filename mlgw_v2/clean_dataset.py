import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

y = np.loadtxt("TD_datasets/31/PCA_train_ph.dat")
x = np.loadtxt("TD_datasets/31/PCA_train_theta.dat")

	#for shifts
s_data = np.loadtxt("TD_datasets/shift_dataset.dat")
x = s_data[:,:3]
y = s_data[:,3+0, None]

low_q = np.where(x[:,0]<4.)[0]

q = np.quantile(y[low_q,:], q = [0.25,0.5,0.75], axis = 0)

z_score = np.divide(y[low_q,:]-q[1,:], np.abs(q[0,:]-q[2,:])) #IQR

#z_score = scipy.stats.zscore(y[low_q,:], axis =0) #true z score (worse apparently)

where_bad = np.where(np.abs(z_score[:,:4])>3.)[0]
where_bad = np.unique(where_bad) #indices in the array y[low_q,:]
#print(z_score[:,:4].shape)

where_bad = np.array( [(i in low_q[where_bad]) for i in range(x.shape[0])] ) #indices in the start array

for PC in range(y.shape[1]):
	plt.figure()
	plt.title(str(PC))
	plt.plot(x[:,0],y[:,PC],'o', ms =1)
	plt.plot(x[where_bad,0],y[where_bad,PC],'o', ms =2)
	plt.xlabel("q")
	plt.ylabel("PC proj")
plt.show()
