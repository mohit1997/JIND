import numpy as np
import torch, sys, os, pdb
from torch import optim
from utils import DataLoaderCustom, ConfusionMatrixPlot, compute_ap
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from models import Classifier
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from .jindlib import JindLib

def main():
	import pickle
	with open('data/blood_annotated.pkl', 'rb') as f:
		data = pickle.load(f)
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

	train_data = train_data[train_data['labels'].isin(common_labels)].copy()
	test_data = data[data['batch'].isin(batches[1:2])].copy()
	# test_data = test_data[test_data['labels'].isin(common_labels)].copy()
	# test_data = test_data[test_data['labels'].isin(common_labels)].copy()

	train_labels = train_data['labels']
	train_gene_mat =  train_data.drop(['labels', 'batch'], 1)

	test_labels = test_data['labels']
	test_gene_mat =  test_data.drop(['labels', 'batch'], 1)

	# assert (set(train_labels)) == (set(test_labels))
	common_labels.sort()
	testing_set = list(set(test_labels))
	testing_set.sort()
	print("Selected Common Labels", common_labels)
	print("Test Labels", testing_set)


	with open('dendrites_results/scRNALib_obj.pkl', 'rb') as f:
		obj = pickle.load(f)

	train_config = {'seed': 0, 'batch_size': 64, 'cuda': False,
					'epochs': 15}

	torch.set_num_threads(25)
	obj.remove_effect(train_gene_mat, test_gene_mat, train_config, test_labels)
	predicted_label  = obj.evaluate(test_gene_mat, test_labels, frac=0.05, name="testcfmtbr.pdf", test=True)
	train_config = {'val_frac': 0.1, 'seed': 0, 'batch_size': 32, 'cuda': False,
					'epochs': 10}
	obj.ftune(test_gene_mat, train_config)
	predicted_label  = obj.evaluate(test_gene_mat, test_labels, frac=0.05, name="testcfmtbrftune.pdf", test=True)
	obj.raw_features = None
	obj.reduced_features = None
	with open('blood_results/scRNALib_objbr.pkl', 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)



if __name__ == "__main__":
	MODEL_WIDTH = 3000
	main()


