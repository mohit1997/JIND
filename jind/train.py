import numpy as np
import torch, sys, os, pdb
import pandas as pd
from torch import optim
from torch.autograd import Variable
from utils import DataLoaderCustom, ConfusionMatrixPlot, compute_ap, normalize
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from models import Classifier, Discriminator, ClassifierBig
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from jind import JindLib
plt.rc('font', family='serif')

def main():
	import pickle
	# data = pd.read_pickle('data/pancreas_integrated.pkl')
	data = pd.read_pickle('data/pancreas_annotatedbatched.pkl')
	cell_ids = np.arange(len(data))
	np.random.seed(0)
	# np.random.shuffle(cell_ids)
	# l = int(0.5*len(cell_ids))

	batches = list(set(data['batch']))
	batches.sort()
	l = int(0.5*len(batches))
	train_data = data[data['batch'].isin(batches[0:1])].copy()
	test_data = data[data['batch'].isin(batches[1:2])].copy()

	train_labels = train_data['labels']
	# train_gene_mat =  train_data.drop(['labels', 'batch'], 1)

	test_labels = test_data['labels']
	# test_gene_mat =  test_data.drop(['labels', 'batch'], 1)

	common_labels = list(set(train_labels) & set(test_labels))
	common_labels.sort()

	common_labels = ['alpha', 'beta', 'delta', 'gamma', 'ductal', 'endothelial']

	train_data = train_data[train_data['labels'].isin(common_labels[:-2])].copy()
	test_data = data[data['batch'].isin(batches[2:3])].copy()
	test_data = test_data[test_data['labels'].isin(common_labels[:-1])].copy()
	# test_data = test_data[test_data['labels'].isin(common_labels)].copy()

	train_labels = train_data['labels']
	train_gene_mat =  train_data.drop(['labels', 'batch'], 1)

	test_labels = test_data['labels']
	test_gene_mat =  test_data.drop(['labels', 'batch'], 1)

	# assert (set(train_labels)) == (set(test_labels))
	common_labels.sort()
	testing_set = list(set(test_labels))
	testing_set.sort()
	print("Selected Common Labels:".rjust(25), common_labels)
	print("Test Labels:".rjust(25), testing_set)


	obj = JindLib(train_gene_mat, train_labels, path="pancreas_results")
	# obj.preprocess()
	obj.dim_reduction(5000, 'Var')

	train_config = {'val_frac': 0.2, 'seed': 0, 'batch_size': 128, 'cuda': False,
					'epochs': 15}
	
	obj.train_classifier(True, train_config, cmat=True)
	
	obj.to_pickle("JindLib_obj.pkl")
	
	predicted_label = obj.evaluate(test_gene_mat, test_labels, frac=0.05, name="testcfmt.pdf")


if __name__ == "__main__":
	# MODEL_WIDTH = 3000
	main()


