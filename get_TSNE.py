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
from scRNALib import scRNALib

def main():
	import pickle
	with open('data/pancreas_annotatedbatched.pkl', 'rb') as f:
		data = pickle.load(f)
	cell_ids = np.arange(len(data))
	np.random.seed(0)
	# np.random.shuffle(cell_ids)
	# l = int(0.5*len(cell_ids))

	batches = list(set(data['batch']))
	batches.sort()
	l = int(0.5*len(batches))
	train_data = data[data['batch'].isin(batches[0:1])].copy()
	test_data = data[data['batch'].isin(batches[1:4])].copy()

	train_labels = train_data['labels']
	# train_gene_mat =  train_data.drop(['labels', 'batch'], 1)

	test_labels = test_data['labels']
	# test_gene_mat =  test_data.drop(['labels', 'batch'], 1)

	common_labels = list(set(train_labels) & set(test_labels))

	train_data = train_data[train_data['labels'].isin(common_labels)].copy()
	test_data = data[data['batch'].isin(batches[3:4])].copy()
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


	with open('pancreas_results/scRNALib_objbr.pkl', 'rb') as f:
		obj = pickle.load(f)

	encoding1 = obj.get_encoding(train_gene_mat)

	encoding2 = obj.get_encoding(test_gene_mat, test=True)
	embedding = obj.get_TSNE(np.concatenate([encoding1, encoding2], axis=0))

	embedding1 = embedding[:len(encoding1)]
	embedding2 = embedding[len(encoding1):]

	plt.figure()
	plt.scatter(embedding1[:, 0], embedding1[:, 1], label="Batch Train")
	plt.legend()
	plt.savefig("Comparison_TSNEb1.pdf")

	plt.figure()
	plt.scatter(embedding1[:, 0], embedding1[:, 1], label="Batch Train")
	plt.scatter(embedding2[:, 0], embedding2[:, 1], label="Batch Test")
	plt.legend()
	plt.savefig("Comparison_TSNE.pdf")



if __name__ == "__main__":
	main()


