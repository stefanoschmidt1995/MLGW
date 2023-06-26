import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(sys.path[0])),'MLGW-master','mlgw_v2')) #ugly way of adding mlgw_v2

import numpy as np
import warnings

#from sklearn.metrics import mean_squared_error
from PCAdata_v2 import PcaData
from NNmodel_v2 import NeuralNetwork
import ast
from GW_generator import GW_generator, mode_generator
from ML_routines import PCA_model

#tf.compat.v1.disable_eager_execution()

#model_loc = "C:/Users/timgr/Documents/Scriptie/Natuurkunde/Code/own_code/model_test/22/"
#reconstruct wave form with spherical harmonics given the list of modes and parameter data
#esstentially only have to change get_red_coefficients function from mode_generator.

class GW_generator_NN(GW_generator):
    
    def __init__(self, folder = 0, verbose = False, frozen = False, batch_size = 1):
        self.modes = [] #list of modes (classes mode_generator)
        self.mode_dict = {}

        if folder is not None:
            #if type(folder) is int:
            #    int_folder = folder
            #    folder = os.path.dirname(inspect.getfile(GW_generator))+"/TD_models/model_"+str(folder)
            #    if not os.path.isdir(folder):
            #        raise RuntimeError("Given value {0} for pre-fitted model is not valid. Available models are:\n{1}".format(str(int_folder), list_models(False)))
            self.load(folder, verbose, frozen, batch_size)
            return
    
    def load(self, folder, verbose = False, frozen=False, batch_size=1):
        if not os.path.isdir(folder):
            raise RuntimeError("Unable to load folder "+folder+": no such directory!")

        if not folder.endswith('/'):
            folder = folder + "/"
        if verbose: print("Loading model from: ", folder)
        file_list = os.listdir(folder)

        if 'README' in file_list:
            with open(folder+"README") as f:
                contents = f.read()
            self.readme = ast.literal_eval(contents) #dictionary holding some relevant information about the model loaded
            try:
                self.readme = ast.literal_eval(contents) #dictionary holding some relevant information about the model loaded
                assert type(self.readme) == dict
            except:
                warnings.warn("README file is not a valid dictionary: entry ignored")
                self.readme = None
            file_list.remove('README')
        else:
            self.readme = None

        print(file_list)
        #loading modes
        for mode in file_list:
            lm = self.__extract_mode(folder+mode+'/')
            if lm is None:
                continue
            else:
                self.mode_dict[lm] = len(self.modes)
                self.modes.append(mode_generator_NN(lm, folder+mode+'/', frozen=frozen, batch_size=batch_size)) #loads mode_generator

            if verbose: print('    Loaded mode {}'.format(lm))

        return
    
    def __extract_mode(self, folder):
        name = os.path.basename(folder[:-1])
        print(name)
        l = name[0]	
        m = name[1]
        try:
            lm = (int(l), int(m))
            assert l>=m
        except:
            warnings.warn('Folder {}: name not recognized as a valid mode - skipping its content'.format(name))
            return None
        return lm

class mode_generator_NN(mode_generator):
    def __init__(self, mode, folder = None, frozen = False, batch_size = 1):
        self.times = None
        self.mode = mode #(l,m) tuple
        self.readme = None	

        if folder is not None:
            self.load(folder, verbose = False, frozen=frozen, batch_size=batch_size)
        return
    #@profile
    def load(self, folder, verbose = False, frozen=False, batch_size=1):
        '''
            collects all relevant pca models, features and NN models.
        '''
        self.frozen = frozen
        self.batch_size = 1
        #loading PCA
        self.amp_PCA = PCA_model()
        self.amp_PCA.load_model(folder+"amp_PCA_model.dat")
        self.ph_PCA = PCA_model()
        self.ph_PCA.load_model(folder+"ph_PCA_model.dat")
        self.times = np.loadtxt(folder+"times.dat")
        
        with open(folder+'amp_features.txt', 'r') as file:
            self.amp_features = file.readline().split(", ")
            #print(self.amp_features)
        with open(folder+'ph_2_features.txt', 'r') as file:
            self.ph_2_features = file.readline().split(", ")
        with open(folder+'ph_3456_features.txt', 'r') as file:
            self.ph_3456_features = file.readline().split(", ")
        with open(folder+'ph_2res_features.txt', 'r') as file:
            self.ph_2res_features = file.readline().split(", ")
        
        if frozen == False:
            self.model_amp = NeuralNetwork.load_model(folder+"model_amp",as_h5 = True)
            self.model_ph_2 = NeuralNetwork.load_model(folder+"model_ph_2", as_h5 = True)
            self.model_ph_3456 = NeuralNetwork.load_model(folder+"model_ph_3456", as_h5 = True)
            self.model_ph_2res = NeuralNetwork.load_model(folder+"model_ph_2res", as_h5 = True)
        else:
            #frozen graph optimization, of course it is not optimzal to create the frozen models
            #inside of the load function.
            
            model_amp = NeuralNetwork.load_model(folder+"model_amp",as_h5 = True)
            model_ph_2 = NeuralNetwork.load_model(folder+"model_ph_2", as_h5 = True)
            model_ph_3456 = NeuralNetwork.load_model(folder+"model_ph_3456", as_h5 = True)
            model_ph_2res = NeuralNetwork.load_model(folder+"model_ph_2res", as_h5 = True)
            
            print("amp_model input shape = ", model_amp.inputs[0].shape)
            full_model_amp = tf.function(lambda x : model_amp(x))
            full_model_amp = full_model_amp.get_concrete_function(
                tf.TensorSpec(shape=[None,18], dtype=model_amp.inputs[0].dtype))
            self.model_amp = convert_variables_to_constants_v2(full_model_amp)
            #self.model_amp.graph.as_graph_def()
            
            full_model_ph_2 = tf.function(lambda x : model_ph_2(x))
            full_model_ph_2 = full_model_ph_2.get_concrete_function(
                tf.TensorSpec(model_ph_2.inputs[0].shape,model_ph_2.inputs[0].dtype))
            self.model_ph_2 = convert_variables_to_constants_v2(full_model_ph_2)
            #self.model_ph_2.graph.as_graph_def()
            
            full_model_ph_3456 = tf.function(lambda x : model_ph_3456(x))
            full_model_ph_3456 = full_model_ph_3456.get_concrete_function(
                tf.TensorSpec(model_ph_3456.inputs[0].shape, model_ph_3456.inputs[0].dtype))
            self.model_ph_3456 = convert_variables_to_constants_v2(full_model_ph_3456)
            #self.model_ph_3456.graph.as_graph_def()
            
            full_model_ph_2res = tf.function(lambda x : model_ph_2res(x))
            full_model_ph_2res = full_model_ph_2res.get_concrete_function(
                tf.TensorSpec(model_ph_2res.inputs[0].shape, model_ph_2res.inputs[0].dtype))
            self.model_ph_2res = convert_variables_to_constants_v2(full_model_ph_2res)
            #self.model_ph_2res.graph.as_graph_def()
            
        self.res_coefficients = np.genfromtxt(folder+"res_coefficients.txt")
        print("mode generator loaded succesfully")
        
    @profile
    def get_red_coefficients(self, theta):
        #Utilize bigger batch_size by loading in multiple theta_vectors. But... architecture of mode_generator
        #has to be changed.
        
        amp_theta = PcaData.augment_features_2(theta, self.amp_features)
        ph_2_theta = PcaData.augment_features_2(theta, self.ph_2_features)
        ph_3456_theta = PcaData.augment_features_2(theta, self.ph_3456_features)
        ph_2res_theta = PcaData.augment_features_2(theta, self.ph_2res_features)
        
        amp_pred = np.zeros((amp_theta.shape[0], self.amp_PCA.get_dimensions()[1]))
        ph_pred = np.zeros((ph_2_theta.shape[0], self.ph_PCA.get_dimensions()[1]))
        
        if not self.frozen:
            amp_pred[:,:4] = self.model_amp.predict(amp_theta, batch_size=self.batch_size, verbose=0)
            ph_2_pred = self.model_ph_2.predict(ph_2_theta, batch_size=self.batch_size, verbose=0)
            ph_3456_pred = self.model_ph_3456.predict(ph_3456_theta, batch_size=self.batch_size, verbose=0)
            ph_2res_pred = self.model_ph_2res.predict(ph_2res_theta, batch_size=self.batch_size, verbose=0)
        else: #does not work 
            amp_theta = np.moveaxis(amp_theta,0,-1)
            amp_pred[:,:4] = self.model_amp(tf.convert_to_tensor(amp_theta[0],dtype=tf.float32))
            ph_2_pred = self.model_ph_2(tf.convert_to_tensor(ph_2_theta[0],dtype=tf.float32))
            ph_3456_pred = self.model_ph_3456(tf.convert_to_tensor(ph_3456_theta[0],dtype=tf.float32))
            ph_2res_pred = self.model_ph_2res(tf.convert_to_tensor(ph_2res_theta[0],dtype=tf.float32))
        
        ph_2res_pred[:,0]*=self.res_coefficients[0]
        ph_2res_pred[:,1]*=self.res_coefficients[1]
        
        ph_2_pred += ph_2res_pred
        ph_pred[:,:6] = np.concatenate((ph_2_pred,ph_3456_pred), axis=1)
        return amp_pred, ph_pred
