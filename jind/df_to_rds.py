import pandas as pd
import rpy2
from rpy2 import robjects
from rpy2.robjects import pandas2ri
import numpy as np
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('-d', action='store', default=None,
					dest='data',
					help='choose dataframe pickle file')
parser.add_argument('-output', action='store', default=None,
					dest='output',
					help='name of the rds file')


def convert_to_rds(df, filename):
	pandas2ri.activate()
	r_frame = robjects.conversion.py2rpy(df)
	robjects.r.assign("my_df", r_frame)
	robjects.r("saveRDS(my_df, file='{}')".format(filename))

if __name__ == "__main__":
	args = parser.parse_args()
	print(args)
	with open(args.data, 'rb') as f:
		data = pickle.load(f)

	convert_to_rds(data, args.output)


	