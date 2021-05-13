# MLGW
MLGW is a Machine Learning model to compute the gravitational waves generated by a Binary Black Hole coalescence. It is part of a thesis project at Università di Pisa under the supervision of prof. Walter Del Pozzo.
The model is released as a Python package ``mlgw`` in the PyPI repository: <https://pypi.org/project/mlgw/>.
You can install the package with
``pip install mlgw``
The model outputs the waveform when given the two BHs masses and spins. It implements also the dependence of the waveform on the spherical harmonics.
Version 1 outputs only the 22 dominant mode. The model is presented in this [paper](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.103.043020) (also online as [arXiv:2011.01958](https://arxiv.org/abs/2011.01958) ), where we present its details, we assess its accuracy and we employ it for analysing the whole GWTC-1, the first catalog of GW transients.
Version 2 is suited to deal with an arbitrary numbers of modes, while at the same time it keeps full compatibility with the previous version interface.
To generate a WF:
```Python
import mlgw.GW_generator as generator
generator = generator.GW_generator() #creating an istance of the generator
theta = np.array([20,10,0.5,-0.3, 1.43, 1.3, 2.3]) #physical parameters [m1,m2,s1,s2, d_L, iota, phi]
times = np.linspace(-8,0.02, 100000) #time grid at which waves shall be evaluated
h_p, h_c = generator.get_WF(theta, times) #returns amplitude and phase of the wave
```
You can read much more details about the model in the [thesis](https://raw.githubusercontent.com/stefanoschmidt1995/MLGW/master/MLGW_package/docs/schmidt_thesis.pdf "Thesis").

### How to use the model
The waveforms are generated by the module ``mlgw.GW_generator``. 
For version 2 a number of tuturials are available:
- [generate_WF](https://raw.githubusercontent.com/stefanoschmidt1995/MLGW/master/mlgw_v2/generate_WF.py): generates and plots a WF. It shows basic usage of the model
- [test_HM](https://raw.githubusercontent.com/stefanoschmidt1995/MLGW/master/mlgw_v2/test_HM.py): builds an histogram of the accuracy of the WFs reconstruced by the model, by comparing them to the train model.
- [play_WF](https://raw.githubusercontent.com/stefanoschmidt1995/MLGW/master/mlgw_v2/play_WF.py): interactive WF generator, to display WF dependence on the relevant parameters.

A number of prefitted models are realesed toghether with the package. However, the interested user can build their own model. A model can be built in two steps following the tutorials:
1. [generating dataset](https://raw.githubusercontent.com/stefanoschmidt1995/MLGW/master/mlgw_v2/generate_dataset.py): a dataset of WFs is built, using TEOBResumS model. User should build a dataset for each mode to be included.
2. [fitting the model](https://raw.githubusercontent.com/stefanoschmidt1995/MLGW/master/mlgw_v2/do_the_fit.py): a PCA model should be fitted on the WFs dataset and a MoE model should learn the regression from the orbital parameter to the reduced order representation of the WF. Module ``fit_model`` takes care of that. A WF model must be saved in a single folder, which must contain a subfolder for each mode included.

### Content of the repository
The repository is organised in the following folders:
- **mlgw_v1**: first version of the model (only 22 mode is fitted)
- **mlgw_v2**: second version of the model (higher modes are included) - not released yet
- **tries_checks**: it holds some tests and checks performed to develop the code - the cose here is not guarenteed to work
- **mlgw_package**: it holds code relevant to the package [``mlgw``](https://pypi.org/project/mlgw/ "mlgw package at PyPI").
- **precession**: code for the developments of ML methods for including precession in the model.
- **paper**: it holds the documents for the [paper](https://arxiv.org/abs/2011.01958 "mlgw").

For more information, you can contact me at [stefanoschmidt1995@gmail.com](mailto:stefanoschmidt1995@gmail.com)

