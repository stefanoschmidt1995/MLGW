"""
mlgw
====
Package for a ML model to generate a GW signal by a BBH coalescence. The model makes use of a PCA model for reducing wave dimensionality. A Mixture of Expert model makes a regression from orbital parameters to the reduced representation of a wave.
The packacke has 4 modules:
	GW_generator.py
		It holds an implementation of the full ML model used to reconstruct a WF. It is a lal style method for creating a waveform. It is ready to use. Loads authomatically EM_MoE and ML_routines modules.
	EM_MoE.py
		It holds a Mixture of Experts model as well as a softmax regression model required for it.
	ML_routines.py
		It holds some useful ML routines such as PCA model (required by the MLGW_generator) and a routine for enlarging a dataset dimensionality.
	GW_helper.py
		It holds some routines useful for generating a GW dataset and a computing mismatch between waveforms. This is not strictly required by the model but it is useful for training the model. It is used by shared code for fitting and created dataset.
		
"""
