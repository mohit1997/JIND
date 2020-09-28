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
parser.add_argument('--name', action='store', default="model1",
                    dest='name',
                    help='Experiment Name')

def main():
	arguments = parser.parse_args()


	data = pd.read_pickle(arguments.data)
	print(set(list(data['labels'])))
	
	source_types = ['alpha', 'beta', 'gamma', 'delta']
	target_types = ['alpha', 'beta', 'gamma', 'delta', 'acinar']

	data = data[data['labels'].isin(target_types)]
	# sys.exit()
	if arguments.traintestbatch == "random":
		path = "datasets/{}_{}".format(arguments.name, arguments.traintestbatch)
		os.system("mkdir -p {}".format(path))
		cell_ids = np.arange(len(data))
		np.random.seed(0)
		np.random.shuffle(cell_ids)
		l = int(0.7*len(cell_ids))

		train_data = data.iloc[cell_ids[:l]]

		test_data = data.iloc[cell_ids[l:]]

		train_gene_mat =  train_data.drop(['batch'], 1, errors='ignore')
		test_gene_mat =  test_data.drop(['batch'], 1, errors='ignore')

	elif arguments.traintestbatch.isdigit():
		path = "datasets/{}_{}".format(arguments.name, arguments.traintestbatch)
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
		
		if source_types is not None:
			train_data = train_data[train_data['labels'].isin(source_types)].copy()
			test_data = test_data[test_data['labels'].isin(target_types)].copy()
		else:
			train_data = train_data[train_data['labels'].isin(common_labels)].copy()
			test_data = test_data[test_data['labels'].isin(common_labels)].copy()

		train_gene_mat =  train_data.drop(['batch'], 1, errors='ignore')
		test_gene_mat =  test_data.drop(['batch'], 1, errors='ignore')

	train_gene_mat.to_pickle("{}/train.pkl".format(path))
	test_gene_mat.to_pickle("{}/test.pkl".format(path))


if __name__ == "__main__":
	main()
