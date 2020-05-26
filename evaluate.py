import numpy as np
import torch, sys, os, pdb
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
	print("Selected Common Labels", common_labels)

	train_data = train_data[train_data['labels'].isin(common_labels)].copy()
	test_data = test_data[test_data['labels'].isin(common_labels)].copy()

	train_labels = train_data['labels']
	train_gene_mat =  train_data.drop(['labels', 'batch'], 1)

	test_labels = test_data['labels']
	test_gene_mat =  test_data.drop(['labels', 'batch'], 1)

	# assert (set(train_labels)) == (set(test_labels))


	with open('pancreas_results/scRNALib_objbr.pkl', 'rb') as f:
		obj = pickle.load(f)

	torch.set_num_threads(25)
	obj.evaluate(test_gene_mat, test_labels, frac=0.05, name="testcfmtbr.pdf", test=True)
	# pdb.set_trace()


if __name__ == "__main__":
	main()
