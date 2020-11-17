import numpy as np
import sys, os, pdb
import pandas as pd
import argparse
import torch
from jind import JindLib
from matplotlib import pyplot as plt
import argparse
from datetime import datetime

np.random.seed(0)

parser = argparse.ArgumentParser(description='RUN JIND')
parser.add_argument('--train_path', default="datasets/human_blood_01/train.pkl", type=str,
					help='path to train data frame with labels')
parser.add_argument('--test_path', default="datasets/human_blood_01/test.pkl", type=str,
					help='path to test data frame with labels')
parser.add_argument('--column', type=str, default='labels',
					help='column name for cell types')

def main():
	torch.set_num_threads(40)
	startTime = datetime.now()
	args = parser.parse_args()
	train_batch = pd.read_pickle(args.train_path)
	test_batch = pd.read_pickle(args.test_path)
	lname = args.column

	train_mat = train_batch.drop(lname, axis=1)
	train_labels = train_batch[lname]

	test_mat = test_batch.drop(lname, axis=1)
	test_labels = test_batch[lname]

	path = os.path.dirname(args.train_path) + "/JIND"

	obj = JindLib(train_mat, train_labels, path=path)
	mat = train_mat.values
	mat_round = np.rint(mat)
	error = np.mean(np.abs(mat - mat_round))
	if error == 0:
		print("Data is int")
		obj.preprocess()

	obj.dim_reduction(5000, 'Var')

	train_config = {'val_frac': 0.2, 'seed': 0, 'batch_size': 128, 'cuda': False,
					'epochs': 15}
	
	obj.train_classifier(config=train_config, cmat=True)
	
	
	predicted_label1, log1 = obj.evaluate(test_mat, test_labels, frac=0.05, name="testcfmt.pdf", return_log=True)
	train_config = {'seed': 0, 'batch_size': 512, 'cuda': False,
					'epochs': 20}

	temp = datetime.now()
	obj.remove_effect(train_mat, test_mat, train_config, test_labels)
	print(datetime.now()  - temp)
	predicted_label2, log2  = obj.evaluate(test_mat, test_labels, frac=0.05, name="testcfmtbr.pdf", test=True, return_log=True)

	train_config = {'val_frac': 0.1, 'seed': 0, 'batch_size': 32, 'cuda': False,
					'epochs': 10}
	obj.ftune(test_mat, train_config)
	predicted_label3, log3  = obj.evaluate(test_mat, test_labels, frac=0.05, name="testcfmtbrftune.pdf", test=True, return_log=True)
	
	obj.to_pickle("JindLib_obj.pkl")

	with open("{}/test.log".format(path), "w") as text_file:
		print("{}".format(log1), file=text_file)
		print("BR {}".format(log2), file=text_file)
		print("ftune {}".format(log3), file=text_file)
		print("Runtime {}".format(datetime.now() - startTime), file=text_file)
	predicted_label1.to_pickle("{}/JIND_assignment.pkl".format(path))
	predicted_label2.to_pickle("{}/JIND_assignmentbr.pkl".format(path))
	predicted_label3.to_pickle("{}/JIND_assignmentbrftune.pkl".format(path))


if __name__ == "__main__":
	main()
