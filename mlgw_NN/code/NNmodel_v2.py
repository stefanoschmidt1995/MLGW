import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import sys
import time

sys.path.append('/home/tim.grimbergen/my_code_v2')
import PCAdata_v2

class NeuralNetwork:
    def __init__(self, PCA_data, quantity, layers_nodes=[10,10,10,10,10], activation = 'sigmoid',
                 optimizer = ('Adam',0), initializer='glorot_uniform', loss_function = ('mean_squared_error',[],0)):
        
        #utility
        self.time = 0 # time it takes to train the model
        self.quantity = quantity
        self.PCA_data = PCA_data
        
        #architecture
        self.layers_nodes = layers_nodes
        self.activation = activation
        self.optimizer = PCAdata_v2.Optimizers(optimizer[0],lr=optimizer[1])
        self.loss_function = PCAdata_v2.LossFunctions(loss_function[0], weights=loss_function[1], time_evo=loss_function[2], exp=loss_function[3])
        
        '''
        self.weights_list = [0]*len(self.loss_function.weights) #make a list of weights like this so they are changeable in callbacks
        for i,x in enumerate(self.loss_function.weights):
            self.weights_list[i] = tf.keras.backend.variable(x)
        '''
        
        K = len(self.PCA_data.train_var[0]) #number of PCA components as output
        L = len(self.PCA_data.train_theta[0]) #number of parameters as input
        
        #build regression model
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Dense(layers_nodes[0],input_shape=(L,), kernel_initializer=initializer, activation=activation))
        for i in range(1,len(layers_nodes)):
            self.model.add(tf.keras.layers.Dense(layers_nodes[i], kernel_initializer=initializer, activation=activation))
        self.model.add(tf.keras.layers.Dense(K,kernel_initializer=initializer,activation='linear'))
        self.model.compile(loss=self.loss_function.LF, optimizer=self.optimizer.opt)
        

    def fit_model(self, max_epochs=5000, Batch_size=500, LRscheduler=PCAdata_v2.Schedulers('exponential')):
        self.batch_size = Batch_size
        self.scheduler = LRscheduler
        
        start = time.time()
        callback_list = []
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)
        callback_list.append(early_stopping)
        
        LR_scheduler =  tf.keras.callbacks.LearningRateScheduler(LRscheduler.scheduler)
        callback_list.append(LR_scheduler)
        '''
        #don't use for now, does not improve results
        if self.loss_function.time_evo != 0:
            #create callback for changing loss_function
            def call_back_LF(self,epoch):
                if epoch <= 500:
                    #for i in range(1,len(self.loss_function.weights)):
                    #    tf.keras.backend.set_value(self.weights_list[0],tf.keras.backend.get_value(self.weights_list[0])-1/10000)
                    tf.keras.backend.set_value(self.weights_list[0],tf.keras.backend.get_value(self.weights_list[0])*1.011)
            
            callback_list.append(tf.keras.callbacks.LambdaCallback(on_epoch_begin= lambda epoch, log : call_back_LF(self,epoch)))
            print("Succesfully added callback for loss_weights evolution")
        '''      
        
        self.history = self.model.fit(x=self.PCA_data.train_theta,y=self.PCA_data.train_var,batch_size=Batch_size,
                                 validation_data=(self.PCA_data.test_theta,self.PCA_data.test_var),
                                 epochs=max_epochs, verbose=0,callbacks=callback_list)
        
        pred_var = self.model.predict(self.PCA_data.test_theta)
        MSE = mean_squared_error(self.PCA_data.test_var,pred_var)
        MSE_PC1 = mean_squared_error(self.PCA_data.test_var[:,0],pred_var[:,0])
        print("The MSE for all principal components of the fit is: ", MSE)
        print("The MSE of first principal component of the fit is: ", MSE_PC1)
        self.time = time.time()-start
        
    def save_model(self, location, dirname, quantity):
        K = len(self.PCA_data.test_var[0]) #number of PCA components
        self.MSE = np.zeros(K)
        os.mkdir(location+dirname)
        
        pred_var = self.model.predict(self.PCA_data.test_theta)
        for i in range(K):
            self.MSE[i]=mean_squared_error(self.PCA_data.test_var[:,i],pred_var[:,i])
        
        self.model_loc = location+dirname
        f = open(location+dirname+"/Model_fit_info.txt", 'x')
        f.write("Generation 2 model for: " + self.PCA_data.quantity+'\n')
        f.write("Trained on dataset: "+self.PCA_data.data_loc+'\n')
        f.write("Features used: ["+", ".join(self.PCA_data.features)+']\n')
        f.write("created model with params: \n")
        f.write("layer_list : ["+",".join([str(x) for x in self.layers_nodes])+']\n')
        f.write("optimizer : " + self.optimizer.name + " with learning rate " + str(self.optimizer.lr)+'\n')
        f.write("activation : " + self.activation + '\n')
        f.write("batch_size : " + str(self.batch_size) + "\n")
        f.write("schedulers : " + self.scheduler.name + " with decay rate " + str(self.scheduler.exp)+'\n')
        f.write("loss_function : " + self.loss_function.name + " with weights " +"["+",".join([str(x) for x in self.loss_function.weights])+"] and time evolution: "+str(self.loss_function.time_evo)+"\n")
        
        for i in range(K):
            f.write("The MSE of principal component "+ str(i) + " of the fit is: "+ str(self.MSE[i])+"\n")
            
        f.write("Time taken for fit: "+str(self.time))
        f.close()
        
        plt.figure('lossfunction')
        plt.title('Loss function of '+self.PCA_data.quantity)
        plt.plot(self.history.history['loss'], label='train')
        plt.plot(self.history.history['val_loss'], label='test')
        plt.yscale('log')
        plt.legend()
        plt.savefig(location+dirname+'/lossfunction.png')
        plt.close(fig='lossfunction')
        self.model.save(location+dirname+'/model.h5') #changed to h5, because else it does not work on cluster
    
    def load_model2(self,weights_location): #don't use
        self.model_loc = "".join(weights_location.split('/')[-2])
        self.model.load_weights(weights_location)
    
    def load_model(model_location, custom_objects = None, as_h5=False):
        if not as_h5:
            #if custom_objects == None: tf.keras.models.load_model(model_location, compile = False)
            return tf.keras.models.load_model(model_location, custom_objects=custom_objects, compile=False)
        else:
            #if custom_objects == None: tf.keras.models.load_model(model_location+'.h5', compile = False)
            return tf.keras.models.load_model(model_location+'.h5', custom_objects=custom_objects, compile=False)


    
    def create_single_model(data_loc, save_loc, dirname, quantity, param_list):
        D = PCAdata_v2.PcaData(data_loc)
        M = NeuralNetwork(D,quantity)
        M.fit_model()
        M.save_model(save_loc,dirname,quantity)
        
    def HyperParametertesting(data_loc, save_loc, quantity, param_dict, PC_comp, epcs = 5000, feat = []):
        '''
            param_dict is a dictionary that can store several types of hyper_parameters
            param_dict = {'layer_list' : [], 'optimizers' : [], 'activation' : [],
                          'batch_size' : [], 'schedulers' : [], 'loss_functions' : []}
            
            if len(param_dict[HP]) == 0, then it will use default value
            if len(param_dict[HP]) == 1, then it will use that value for all models
            if len(param_dict[HP]) > 2, then it will loop over the specified values
            
            code creates a param_list that is essentially a list of lists of length 6
            that indicate which params to be used
        '''
        
        default = {'layer_list' : [10,10,10,10,10],
                   'optimizers' : ('Adam',0),
                   'activation' : 'sigmoid',
                   'batch_size' : 500,
                   'schedulers' : PCAdata_v2.Schedulers('exponential'),
                   'loss_functions' : ('mean_squared_error',[],0)}
        for key in param_dict:
            if len(param_dict[key]) == 0:
                param_dict[key] = [default[key]]
        
        param_list = []
        for a in param_dict['layer_list']:
            for b in param_dict['optimizers']:
                for c in param_dict['activation']:
                    for d in param_dict['batch_size']:
                        for e in param_dict['schedulers']:
                            for f in param_dict['loss_functions']:
                                param_list.append([a,b,c,d,e,f])
        
        
        
        for (a,b,c,d,e,f) in param_list:
            print("started creating model with params: \n")
            print("layer_list : ["+",".join([str(x) for x in a])+']')
            print("optimizer : " + b[0] + " with learning rate " + str(b[1]))
            print("activation : " + c)
            print("batch_size : " + str(d))
            print("schedulers : " + e.name + " with decay rate " + str(e.exp))
            print("loss_function : " + f[0] + " with weights " +"["+",".join([str(x) for x in f[1]])+"] and time evolution: "+str(f[2])+"\n")
            dirname = quantity+'_'+"["+",".join([str(x) for x in a])+"]"
            D = PCAdata_v2.PcaData(data_loc, PC_comp, quantity, features=feat)
            M = NeuralNetwork(D, quantity ,layers_nodes=a, activation = c, optimizer = b, loss_function = f)
            M.fit_model(Batch_size = d, LRscheduler = e, max_epochs = epcs)
            M.save_model(save_loc, dirname, quantity)
            del D
            del M
    
    def ContinueTraining(data_loc, save_loc, quantity, PC, model_loc, model_param_dict, feat=[], epcs = 5000):
        '''
            model_param_dict is of the form:
                {layer_list : [], activation : str, optimizer : (str,float),
                 loss_function : (str,[]), batch_size : int, scheduler : PCAdataV2.loss_functions('exponential')}
            and should correspond to the model that is continued.
            
            should implement: concatenation of new training set and old data set (what do do with PCA?)
            
        '''
        
        D = PCAdata_v2.PcaData(data_loc, PC, quantity, features=feat)
        M = NeuralNetwork(D,quantity,
                          layers_nodes=model_param_dict['layer_list'],
                          activation=model_param_dict['activation'],
                          optimizer=model_param_dict['optimizer'],
                          loss_function=model_param_dict['loss_function'])
        M.model = NeuralNetwork.load_model(model_loc)
        print("model loaded and initialized, now starting fit")
        M.fit_model(Batch_size = model_param_dict['batch_size'],
                    LRscheduler = model_param_dict['scheduler'],
                    max_epochs=epcs)
        M.save_model(save_loc, model_loc.split('/')[-2]+"_updated",quantity)
        print('model ftted and saved')

    def CreateResidualPCAsets(PCA_data, pred, model_loc, save_loc, q, components=1):
        times = PCA_data.times
        train_theta = PCA_data.train_theta[:,:3]
        test_theta = PCA_data.test_theta[:,:3]
        train_pred = pred[0][:,:components]
        test_pred = pred[1][:,:components]
        
        new_train_var = PCA_data.train_var[:,:components] - train_pred
        new_test_var = PCA_data.test_var[:,:components] - test_pred
        
        #normalize the datasets
        norm_coef = []
        for i in range(components):
            test_max = np.max(abs(new_test_var[:,i]))
            new_train_var[:,i] /= test_max
            new_test_var[:,i] /= test_max
            norm_coef.append(test_max)

        
        if q == 'ph':
            PCA_data.pca.save_model(save_loc+"ph_PCA_model.dat")
            np.savetxt(save_loc+"PCA_train_theta.dat", train_theta)
            np.savetxt(save_loc+"PCA_test_theta.dat", test_theta)
            np.savetxt(save_loc+"PCA_train_ph.dat", new_train_var)
            np.savetxt(save_loc+"PCA_test_ph.dat", new_test_var)
            
        if q == 'amp':
            PCA_data.pca.save_model(save_loc+"amp_PCA_model.dat")
            np.savetxt(save_loc+"PCA_train_theta.dat", train_theta)
            np.savetxt(save_loc+"PCA_test_theta.dat", test_theta)
            np.savetxt(save_loc+"PCA_train_amp.dat", new_train_var)
            np.savetxt(save_loc+"PCA_test_amp.dat", new_test_var)
        
        np.savetxt(save_loc+"times.dat", times)
        
        f = open(save_loc+"/info.txt", 'x')
        f.write("Model location: "+model_loc+"\n")
        f.write("Test scale coefficients: "+", ".join(str(x) for x in norm_coef))
        f.close()
        
        plt.figure('new_pca PC1')
        plt.title('delta pca/pred for test data, PC1')
        plt.scatter(test_theta[:,0], new_test_var[:,0])
        plt.savefig(save_loc+'/delta pca-pred.png')
        plt.close(fig='new_pca') 
