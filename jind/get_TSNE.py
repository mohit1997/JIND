import numpy as np
import torch, sys, os, pdb
import pandas as pd
from torch import optim
from utils import DataLoaderCustom, ConfusionMatrixPlot, compute_ap
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from models import Classifier
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from jind import JindLib
import seaborn as sns
from sklearn.cluster import DBSCAN, OPTICS, SpectralClustering, AgglomerativeClustering

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
	test_data = data[data['batch'].isin(batches[2:3])].copy()

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

	# test_gene_mat = train_gene_mat * (1 + 5 * np.random.randn(train_gene_mat.shape[1]))
	# test_labels = train_labels

	with open('pancreas_results/JindLib_objbr.pkl', 'rb') as f:
		obj = pickle.load(f)

	predicted_label  = obj.predict(test_gene_mat, test=True, return_names=True)
	assert len(predicted_label) == len(test_labels)
	filtered_label = obj.detect_novel(train_gene_mat, train_labels, test_gene_mat, predicted_label, test_labels=test_labels)

	test_gene_mat_filtered = test_gene_mat[filtered_label['Novel'] == False]
	test_labels_filtered = test_labels[filtered_label['Novel']==False]

	train_config = {'seed': 0, 'batch_size': 512, 'cuda': False,
					'epochs': 20}

	torch.set_num_threads(25)
	obj.remove_effect(train_gene_mat, test_gene_mat_filtered, train_config, test_labels_filtered)
	train_config = {'val_frac': 0.1, 'seed': 0, 'batch_size': 32, 'cuda': False,
					'epochs': 10}
	obj.ftune(test_gene_mat_filtered, train_config)
	obj.to_pickle("JindLib_objbrnovel.pkl")

	obj.evaluate(test_gene_mat, test_labels, frac=0.05, name="testcfmtbrnovel.pdf", test=True)



if __name__ == "__main__":
	main()


