import numpy as np
import sys, os, pdb
import pandas as pd
import argparse
import torch
from jind import JindLib
from matplotlib import pyplot as plt
from datetime import datetime

np.random.seed(0)

parser = argparse.ArgumentParser(description='RUN JIND')
parser.add_argument('--train_path', default="datasets/human_blood_01/train.pkl", type=str,
					help='path to train data frame with labels')
parser.add_argument('--test_path', default="datasets/human_blood_01/test.pkl", type=str,
					help='path to test data frame with labels')
parser.add_argument('--column', type=str, default='labels',
					help='column name for cell types')
parser.add_argument('--seed', type=int, default=0,
					help='Random Seed')

def main():
	torch.set_num_threads(40)
	startTime = datetime.now()
	args = parser.parse_args()
	train_batch = pd.read_pickle(args.train_path)
	test_batch = pd.read_pickle(args.test_path)
	lname = args.column

	common_genes = list(set(train_batch.columns).intersection(set(test_batch.columns)))
	common_genes.sort()
	train_batch = train_batch[list(common_genes)]
	test_batch = test_batch[list(common_genes)]

	train_mat = train_batch.drop(lname, axis=1).fillna(0)
	train_labels = train_batch[lname]

	test_mat = test_batch.drop(lname, axis=1).fillna(0)
	test_labels = test_batch[lname]

	path = os.path.dirname(args.train_path) + f"/JIND_rawtop_{args.seed}"

	obj = JindLib(train_mat, train_labels, path=path)
	mat = train_mat.values
	mat_round = np.rint(mat)
	error = np.mean(np.abs(mat - mat_round))
	if error == 0:
		if "human_dataset_random" in args.train_path:
			obj.preprocess(count_normalize=True, logt=False)	
		else:
			obj.preprocess(count_normalize=True, logt=True)
	# Uncomment if the data is non integer but contains counts
	obj.preprocess(count_normalize=True, logt=True)

	obj.dim_reduction(5000, 'Var')
	# obj.normalize()

	train_config = {'val_frac': 0.2, 'seed': args.seed, 'batch_size': 128, 'cuda': False,
					'epochs': 15}
	
	obj.train_classifier(config=train_config, cmat=True)
	
	
	predicted_label1, log1 = obj.evaluate(test_mat, test_labels, frac=0.05, name="testcfmt.pdf", return_log=True)


	# train_config = {'val_frac': 0.1, 'seed': args.seed, 'batch_size': 128, 'cuda': False,
	# 				'epochs': 20}
	
	# obj.ftune_encoder(test_mat, train_config)
	# predicted_label1_, log1_ = obj.evaluate(test_mat, test_labels, frac=0.05, name="testcfmtftuneencoder.pdf", test="modelftuned", return_log=True)

	train_config = {'seed': args.seed, 'batch_size': 128, 'cuda': False,
					'epochs': 15, 'gdecay': 1e-2, 'ddecay': 1e-1, 'maxcount': 7, 'sigma': 0.0}

	temp = datetime.now()
	obj.remove_effect(train_mat, test_mat, train_config, test_labels)
	print(datetime.now()  - temp)
	predicted_label2, log2  = obj.evaluate(test_mat, test_labels, frac=0.05, name="testcfmtbr.pdf", test=True, return_log=True)

	# obj.detect_novel(train_mat, train_labels, test_mat, predicted_label2, test_labels=test_labels, test=True)
	train_config = {'val_frac': 0.1, 'seed': args.seed, 'batch_size': 128, 'cuda': False,
					'epochs': 10}
	obj.ftune_top(test_mat, train_config)
	predicted_label3, log3  = obj.evaluate(test_mat, test_labels, frac=0.05, name="testcfmtbrftune.pdf", test=True, return_log=True)
	
	obj.to_pickle("JindLib_obj.pkl")

	with open("{}/test.log".format(path), "w") as text_file:
		print("{}".format(log1), file=text_file)
		# print("ftune encoder {}".format(log1_), file=text_file)
		print("BR {}".format(log2), file=text_file)
		print("ftune {}".format(log3), file=text_file)
		print("Runtime {}".format(datetime.now() - startTime), file=text_file)
	predicted_label1.to_pickle("{}/JIND_assignment.pkl".format(path))
	predicted_label2.to_pickle("{}/JIND_assignmentbr.pkl".format(path))
	predicted_label3.to_pickle("{}/JIND_assignmentbrftune.pkl".format(path))


if __name__ == "__main__":
	main()
