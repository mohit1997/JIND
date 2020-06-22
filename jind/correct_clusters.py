import numpy as np
import torch, sys, os, pdb
from torch import optim
from utils import DataLoaderCustom, ConfusionMatrixPlot, compute_ap
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from models import Classifier
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scRNALib import scRNALib
import seaborn as sns
import pandas as pd

def main():
	import pickle
	with open('pancreas_results/scRNALib_objbr.pkl', 'rb') as f:
		obj = pickle.load(f)
	df = pd.read_pickle("cluster_res.pkl")
	
	ind = df['pred'] == "Unassigned"
	max_id = df[ind]['DBSCAN'].value_counts().idxmax()
	
	temp = df[ind]['DBSCAN'].values
	un, counts = np.unique(temp, return_counts=True)

	temp_ = np.sort(df['DBSCAN'].values)
	un_, counts_ = np.unique(temp_, return_counts=True)
	print(un_)
	if -1 in un_:
		counts_match = counts_[un+1]
	else:
		counts_match = counts_[un]

	args = np.argsort(-counts)

	sorted_counts = counts[args]
	rep_list = un[args]
	sorted_counts_ = counts_match[args]
	sorted_frac = sorted_counts/sorted_counts_

	df['pred_correct'] = df['pred'].copy()
	print(sorted_counts, sorted_frac)
	if sorted_frac[0] > 0.3:
		ind = df['DBSCAN'] == max_id
		df.loc[ind, 'pred_correct'] = "Unassigned"

	plt.figure()
	order = list(set(df['pred']))
	order.sort()

	g = sns.scatterplot(x="ex", y="ey", hue="pred", data=df, hue_order=order)
	plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
	plt.tight_layout()
	plt.savefig("TSNE_pred.pdf")

	plt.figure()
	order = list(set(df['labels']))
	order.sort()

	g = sns.scatterplot(x="ex", y="ey", hue="labels", data=df, hue_order=order)
	plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
	plt.tight_layout()
	plt.savefig("TSNE_true.pdf")

	plt.figure()

	g = sns.scatterplot(x="ex", y="ey", hue="DBSCAN", data=df, legend="full", palette="viridis")
	# plt.scatter(embedding[:, 0], embedding[:, 1], c=clustering.labels_)
	plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
	plt.tight_layout()
	plt.savefig("TSNE_DBSCAN.pdf")

	plt.figure()
	order = list(set(df['pred_correct']))
	order.sort()

	g = sns.scatterplot(x="ex", y="ey", hue="pred_correct", data=df, hue_order=order)
	# plt.scatter(embedding[:, 0], embedding[:, 1], c=clustering.labels_)
	plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
	plt.tight_layout()
	plt.savefig("TSNE_corrected.pdf")

	ind = df['pred'] == df['labels']
	data_filt = df[~df['pred'].isin(["Unassigned"])]
	ind_filt = data_filt['pred'] == data_filt['labels']
	ind_filt = ind_filt.values
	print("Accuracy Pre {:.4f} Post {:.4f}".format(np.mean(ind), np.mean(ind_filt)))

	ind = df['pred_correct'] == df['labels']
	data_filt = df[~df['pred_correct'].isin(["Unassigned"])]
	ind_filt = data_filt['pred_correct'] == data_filt['labels']
	ind_filt = ind_filt.values
	print("Accuracy Pre {:.4f} Post {:.4f}".format(np.mean(ind), np.mean(ind_filt)))

	obj.generate_cfmt(df['pred_correct'], df['labels'], name="testcfmtcorrected.pdf")





if __name__ == "__main__":
	main()


