import numpy as np
import sys, os, pdb
import pandas as pd
import argparse

np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--data', action='store', default=None,
                    dest='data',
                    help='choose pandas dataframe')
parser.add_argument('--traintestbatch', type=str, action='store', default="01",
                    dest='traintestbatch',
                    help='choose training testing batch ids such as "01", put "random" for creating batch randomly')
parser.add_argument('--log', type=str, default='logs.txt',
                        help='Name for the log file')
parser.add_argument('--name', action='store', default="model1",
                    dest='name',
                    help='Experiment Name')

def main():
	arguments = parser.parse_args()


	data = pd.read_pickle(arguments.data)

	if arguments.traintestbatch.isdigit():
		path = "data/{}_{}".format(arguments.name, arguments.traintestbatch)
		os.system("mkdir -p {}".format(path))

		batches = list(set(data['batch']))
		batches.sort()
		train_batch = int(arguments.traintestbatch[0])
		test_batch = int(arguments.traintestbatch[1])
		train_data = data[data['batch'] == batches[train_batch]].copy()
		test_data = data[data['batch'] == batches[test_batch]].copy()

		train_labels = train_data['labels']

		test_labels = test_data['labels']

		common_labels = list(set(train_labels) & set(test_labels))
		common_labels.sort()

		print(common_labels)
		
		train_data = train_data[train_data['labels'].isin(common_labels)].copy()
		test_data = test_data[test_data['labels'].isin(common_labels)].copy()

		data = pd.concat([train_data, test_data], axis=0)

		data.to_pickle("{}/frame_batched.pkl".format(path))


if __name__ == "__main__":
	main()
