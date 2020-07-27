import numpy as np
import sys, os, pdb
import pandas as pd
import argparse
from jind import JindLib
from matplotlib import pyplot as plt
import argparse
from datetime import datetime

np.random.seed(0)

parser = argparse.ArgumentParser(description='RUN JIND')
parser.add_argument('--train_path', default="datasets/mouse_dataset_random/train.pkl", type=str,
					help='path to train data frame with labels')
parser.add_argument('--test_path', default="datasets/mouse_dataset_random/test.pkl", type=str,
					help='path to test data frame with labels')
parser.add_argument('--column', type=str, default='labels',
					help='column name for cell types')

def main():
	startTime = datetime.now()
	args = parser.parse_args()
	train_batch = pd.read_pickle(args.train_path)
	test_batch = pd.read_pickle(args.test_path)
	lname = args.column

	train_mat = train_batch.drop(lname, axis=1)
	train_labels = train_batch[lname]

	path = os.path.dirname(args.train_path) + "/Silhoutte_tSNE"

	obj = JindLib(train_mat, train_labels, path=path)
	mat = train_mat.values
	mat_round = np.rint(mat)
	error = np.mean(np.abs(mat - mat_round))
	if error == 0:
		print("Data is int")
		obj.preprocess()

	obj.dim_reduction(5000, 'Var')

	df, log = obj.get_complexity()
	
	with open("{}/test.log".format(path), "w") as text_file:
		print("{}".format(log), file=text_file)

	df.to_pickle("{}/tSNE_embeddings.pkl".format(path))


if __name__ == "__main__":
	main()
