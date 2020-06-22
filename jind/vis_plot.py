import numpy as np
import pandas as pd
import sys, os, pdb
from .scRNALib import scRNALib
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt
import multiprocessing
from functools import partial
from .scRNAvis import scRNAVis


def main():
	import pickle
	data = pd.read_pickle('data/blood_annotated.pkl')

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
	test_data = test_data[test_data['labels'].isin(common_labels)].copy()
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


	with open('blood_results/scRNAvis_obj.pkl', 'rb') as f:
		visobj = pickle.load(f)

	visobj.mat = test_gene_mat
	visobj.plot_2d('tsne', test=True)
	visobj.plot_2d('umap', test=True)


if __name__ == "__main__":
	main()

