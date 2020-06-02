import numpy as np
import pandas as pd
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

	for lab in testing_set:
		ind = test_labels == lab
		print("Proporation {}: {:.3f}".format(lab, np.mean(ind)))


	with open('pancreas_results/scRNALib_objbr.pkl', 'rb') as f:
		obj = pickle.load(f)

	torch.set_num_threads(25)
	obj.evaluate(test_gene_mat, test_labels, frac=0.05, name="testcfmt1.pdf", test=False)
	predicted_label  = obj.evaluate(test_gene_mat, test_labels, frac=0.05, name="testcfmtbr1.pdf", test=True)
	predicted_label = pd.DataFrame({"cellname":test_gene_mat.index, "pred":predicted_label, "labels":test_labels})
	predicted_label.to_csv("predicted_label3.txt", sep="\t", index=False)
	# pdb.set_trace()


if __name__ == "__main__":
	main()
