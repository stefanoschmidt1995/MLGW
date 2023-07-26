"""
Given a PCA dataset and a trained model, it creates a dataset with the residual of the given model.

Typical usage:
	
	python create_res_dataset.py --pca-dataset ../pca_datasets/IMRPhenomTPHM/22/ --save-location ../pca_datasets/IMRPhenomTPHM/22_residual/ --model-location ../models_NN/model_IMRPhenomTPHM/ph/22/comp01/ --components 0 1 --quantity ph

"""
from mlgw.NN_model import create_residual_PCA
import argparse

parser = argparse.ArgumentParser(__doc__)

parser.add_argument(
	"--pca-dataset", type = str, required = True,
	help="Folder for the PCA dataset")

parser.add_argument(
	"--model-location", type = str, required = True,
	help="Location for the model")

parser.add_argument(
	"--save-location", type = str, required = True,
	help="Location to save the residual dataset")

parser.add_argument(
	"--quantity", type = str, required = False, choices = ['amp', 'ph'], default = 'ph',
	help="Wheter to create the dataset for amplitude of phase")

parser.add_argument(
	"--components", type = int, required = False, nargs = '+', default = 2,
	help="Wheter to create the dataset for amplitude of phase")

args = parser.parse_args()

create_residual_PCA(args.pca_dataset, args.model_location, args.save_location, args.quantity, args.components)
