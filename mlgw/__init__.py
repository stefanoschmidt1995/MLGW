"""
`mlgw` uses Machine Learning (ML) to generate a Gravitational Wave (GW) signal from a Binary Black Hole (BBH) coalescence.
The model makes use of a PCA model for reducing wave dimensionality. An ensemble of Neural Networks (or a Mixture of Expert model, in the older versions) makes a regression from orbital parameters to the reduced representation of a wave.

The packacke has 4 modules:

GW_generator.py
	It holds an implementation of the full ML model used to reconstruct a WF. It is a lal style method for creating a waveform. It is ready to use. Loads authomatically EM_MoE and ML_routines modules.
EM_MoE.py
	It holds a Mixture of Experts model as well as a softmax regression model required for it.
ML_routines.py
	It holds some useful ML routines such as PCA model (required by the MLGW_generator) and a routine for performing basis function expansion.
GW_helper.py
	It holds some routines useful for generating a GW dataset and a computing mismatch between waveforms. This is not strictly required by the model but it is useful for training the model. Used by module fit_model.py
fit_model.py
	It holds some routines to effectively fit the model.
		
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from .GW_generator import GW_generator, list_models
