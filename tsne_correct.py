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
import seaborn as sns
import pandas as pd

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

	train_data = train_data[train_data['labels'].isin(common_labels[:-4])].copy()
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

	for lab in testing_set:
		ind = test_labels == lab
		print("Proportion {}: {:.3f}".format(lab, np.mean(ind)))


	with open('pancreas_results/scRNALib_objbr.pkl', 'rb') as f:
		obj = pickle.load(f)

	torch.set_num_threads(25)
	obj.evaluate(test_gene_mat, test_labels, frac=0.05, name="testcfmt1.pdf", test=False)
	predicted_label  = obj.evaluate(test_gene_mat, test_labels, frac=0.05, name="testcfmtbr1.pdf", test=True)

	# encoding = obj.get_encoding(test_gene_mat, test=True)
	embedding = obj.get_TSNE(test_gene_mat.values)

	from sklearn.cluster import DBSCAN, OPTICS, SpectralClustering, AgglomerativeClustering
	from sklearn.neighbors import kneighbors_graph

	db = DBSCAN(eps=3., min_samples=2).fit(embedding)

	knn_graph = kneighbors_graph(embedding, 30, include_self=False)
	agg = AgglomerativeClustering(n_clusters=(obj.n_classes+1), connectivity=knn_graph).fit(embedding)
	opt = OPTICS(min_samples=20, max_eps=6.).fit(embedding)

	df = pd.DataFrame({"ex": embedding[:, 0],
						"ey": embedding[:, 1],
						# "cellname":test_gene_mat.index,
						"pred": predicted_label,
						"labels": test_labels,
						"DBSCAN": db.labels_,
						"AGG": agg.labels_,
						"OPTICS": opt.labels_,
						})
	ind = df['pred'] == df['labels']
	data_filt = df[~df['pred'].isin(["Unassigned"])]
	ind_filt = data_filt['pred'] == data_filt['labels']
	ind_filt = ind_filt.values
	print("Accuracy Pre {:.4f} Post {:.4f}".format(np.mean(ind), np.mean(ind_filt)))
	
	df.to_pickle("cluster_res.pkl")
	





if __name__ == "__main__":
	main()


