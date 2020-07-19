import numpy as np
import sys, os, pdb
import pandas as pd
import argparse
from jind import SVMReject
from matplotlib import pyplot as plt
import argparse

np.random.seed(0)

parser = argparse.ArgumentParser(description='RUN SVM Reject')
parser.add_argument('--train_path', default="datasets/human_blood_integrated_01/train.pkl", type=str,
					help='path to train data frame with labels')
parser.add_argument('--test_path', default="datasets/human_blood_integrated_01/test.pkl", type=str,
					help='path to test data frame with labels')
parser.add_argument('--column', type=str, default='labels',
					help='column name for cell types')

def main():
	args = parser.parse_args()
	train_batch = pd.read_pickle(args.train_path)
	test_batch = pd.read_pickle(args.test_path)
	lname = args.column

	train_mat = train_batch.drop(lname, axis=1)
	train_labels = train_batch[lname]

	test_mat = test_batch.drop(lname, axis=1)
	test_labels = test_batch[lname]

	path = os.path.dirname(args.train_path) + "/SVMReject"

	obj = SVMReject(train_mat, train_labels, path=path)
	# obj.preprocess()
	obj.dim_reduction(5000, 'Var')

	train_config = {'val_frac': 0.2, 'seed': 0, 'batch_size': 128, 'cuda': False,
					'epochs': 15}
	
	obj.train_classifier(True, train_config, cmat=True)
	
	obj.to_pickle("SVMReject_obj.pkl")
	
	predicted_label, log = obj.evaluate(test_mat, test_labels, frac=0.05, name="testcfmt.pdf", return_log=True)
	
	with open("{}/test.log".format(path), "w") as text_file:
		print("{}".format(log), file=text_file)
	predicted_label.to_pickle("{}/SVMReject_assignment.pkl".format(path))


if __name__ == "__main__":
	main()
