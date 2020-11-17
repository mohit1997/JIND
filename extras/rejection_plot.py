import numpy as np
import sys, os, pdb
import pandas as pd
import argparse
from matplotlib import pyplot as plt
import argparse
from datetime import datetime
import plotly.graph_objects as go
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns

np.random.seed(0)

parser = argparse.ArgumentParser(description='Plot Rejection Figure')
parser.add_argument('--file', default="datasets/results.csv", type=str,
					help='path to train data frame with labels')

def main():
	args = parser.parse_args()

	data = pd.read_csv(args.file, )
	df = data.melt('Dataset', var_name='Method',  value_name='% Rejected')
	print(df)
	plt.figure(figsize=(20,5))
	ax = sns.catplot(x="% Rejected", y="Method", hue='Dataset', data=df, s=5, height=3, aspect=1.5)
	ax._legend.remove()
	plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),fancybox=True, shadow=True, ncol=3)
	plt.tight_layout()


	path = os.path.dirname(args.file)
	plt.savefig("{}/{}_Rejection.pdf".format(path, os.path.splitext(os.path.basename(args.file))[0]))


	plt.figure(figsize=(8,5))
	ax = sns.barplot(x="% Rejected", y="Method", hue='Dataset', data=df)
	# ax._legend.remove()
	plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),fancybox=True, shadow=True, ncol=3)
	plt.tight_layout()


	path = os.path.dirname(args.file)
	plt.savefig("{}/{}_RejectionBar.pdf".format(path, os.path.splitext(os.path.basename(args.file))[0]))

	plt.figure(figsize=(6,8))
	ax = sns.barplot(x="Method", y="% Rejected", hue='Dataset', data=df)
	ax.set_xticklabels(ax.get_xticklabels(), rotation=0, horizontalalignment='center')
	# ax._legend.remove()
	plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),fancybox=True, shadow=True, ncol=2)
	plt.tight_layout()


	path = os.path.dirname(args.file)
	plt.savefig("{}/{}_RejectionBarVertical.pdf".format(path, os.path.splitext(os.path.basename(args.file))[0]))


	plt.figure(figsize=(10, 6))
	ax = sns.lineplot(x="Dataset", y="% Rejected", hue='Method', data=df, markers=True, style="Method")
	# ax._legend.remove()
	# ax.set_xticklabels(list(data['Dataset']), rotation=0, horizontalalignment='center')
	plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),fancybox=True, shadow=True, ncol=3)
	plt.tight_layout()

	plt.savefig("{}/{}_RejectionLine.pdf".format(path, os.path.splitext(os.path.basename(args.file))[0]))





if __name__ == "__main__":
	main()
