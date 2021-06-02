import os
import numpy as np
import pandas as pd
from sklearn import metrics
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="dark")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

DASH_STYLE_LIST = ['-', '--', ':', '-.', '.', 'o']

# linestyles = ["-", (0,(5,2,5,5,1,4))]
# plt.rcParams["axes.prop_cycle"] += plt.cycler("linestyle", linestyles)

np.random.seed(0)

parser = argparse.ArgumentParser(description='RUN SVM Reject')
# parser.add_argument('--dataset', default="mouse_dataset_random", type=str,
# 					help='dataset name')
# parser.add_argument('--SVM_dataset', default="mouse_dataset_random", type=str,
# 					help='dataset name')
# parser.add_argument('--ACTINN_dataset', default="mouse_dataset_random", type=str,
# 					help='dataset name')
# parser.add_argument('--scPred_dataset', default="mouse_dataset_random", type=str,
# 					help='dataset name')

parser.add_argument('--dataset', default="human_blood_01", type=str,
					help='dataset name')
parser.add_argument('--SVM_dataset', default="human_blood_integrated_01", type=str,
					help='dataset name')
parser.add_argument('--ACTINN_dataset', default="human_blood_integrated_01", type=str,
					help='dataset name')
parser.add_argument('--scPred_dataset', default="human_blood_integrated_01", type=str,
					help='dataset name')

# parser.add_argument('--dataset', default="pancreas_raw_01", type=str,
# 					help='dataset name')
# parser.add_argument('--SVM_dataset', default="pancreas_raw_hintegrated_01_01", type=str,
# 					help='dataset name')
# parser.add_argument('--ACTINN_dataset', default="pancreas_raw_hintegrated_01_01", type=str,
# 					help='dataset name')
# parser.add_argument('--scPred_dataset', default="pancreas_raw_sintegrated_01_01", type=str,
# 					help='dataset name')

# parser.add_argument('--dataset', default="pancreas_raw_02", type=str,
# 					help='dataset name')
# parser.add_argument('--SVM_dataset', default="pancreas_raw_hintegrated_02_01", type=str,
# 					help='dataset name')
# parser.add_argument('--ACTINN_dataset', default="pancreas_raw_hintegrated_02_01", type=str,
# 					help='dataset name')
# parser.add_argument('--scPred_dataset', default="pancreas_raw_sintegrated_02_01", type=str,
# 					help='dataset name')

def main():
	args = parser.parse_args()

	JINDp_path = f"/home/mohit/mohit/seq-rna/Comparison/datasets/{args.dataset}/JIND_rawtop_0/JIND_assignmentbrftune.pkl"
	JIND_path = f"/home/mohit/mohit/seq-rna/Comparison/datasets/{args.dataset}/JIND_rawtop_0/JIND_assignmentbr.pkl"
	Seurat_path = f"/home/mohit/mohit/seq-rna/Comparison/datasets/{args.dataset}/seurat/seurat_assignment.pkl"
	ITClust_path = f"/home/mohit/mohit/seq-rna/Comparison/datasets/{args.dataset}/ItClusterfiltercells_0/ItCluster_assignment.pkl"
	SVM_path = f"/home/mohit/mohit/seq-rna/Comparison/datasets/{args.SVM_dataset}/SVMReject/SVMReject_assignment.pkl"
	ACTINN_path = f"/home/mohit/mohit/seq-rna/Comparison/datasets/{args.ACTINN_dataset}/ACTINN/ACTINN_assignment.pkl"
	scPred_path = f"/home/mohit/mohit/seq-rna/Comparison/datasets/{args.scPred_dataset}/scPred/scPred_assignment.pkl"

	data_jindp = pd.read_pickle(JINDp_path)
	data_jind = pd.read_pickle(JIND_path)
	JIND_outpath = f"/home/mohit/mohit/seq-rna/Comparison/PR_plots/{args.dataset}"
	data_seurat = pd.read_pickle(Seurat_path)
	data_itclust = pd.read_pickle(ITClust_path)
	data_svm = pd.read_pickle(SVM_path)
	data_actinn = pd.read_pickle(ACTINN_path)
	data_scpred = pd.read_pickle(scPred_path)
	
	os.makedirs(JIND_outpath, exist_ok=True)

	num_ctypes = len(list(set(data_jind['labels'])))

	gridsize = int(np.ceil(np.sqrt(num_ctypes)))

	fig = plt.figure(figsize=(6*gridsize, 6*gridsize))

	AUPR_dic = {'JIND+': [], 'JIND': [], 'Seurat': [], "ItClust": [], 'SVM_Rej': [], 'ACTINN': [], 'scPred': []}
	AUPRw_dic = {'JIND+': [], 'JIND': [], 'Seurat': [], "ItClust": [], 'SVM_Rej': [], 'ACTINN': [], 'scPred': []}

	for ind, i in enumerate(sorted(list(set(data_jind['labels'])))):
		# ax = sns.lineplot(x=recall, y=precision)
		# ax = sns.lineplot(x=recall, y=precision)

		# ax = axes[0, 0]
		# fig, ax = plt.subplots(1, 1, figsize=(6, 7))

		R_i = i.replace(" ", ".").replace("-", ".").replace("_", ".")

		ax = plt.subplot(gridsize, gridsize, ind + 1)

		probs = list(data_jindp[i])
		labels = np.array(list(data_jindp['labels'] == i)) * 1
		precision, recall, thresholds = metrics.precision_recall_curve(labels, probs)
		AUPR_dic['JIND+'].append(metrics.average_precision_score(labels, probs))
		AUPRw_dic['JIND+'].append(np.sum(labels) + 1e-5)
		ax.plot(recall, precision, label="JIND+")
		

		probs = list(data_jind[i])
		labels = np.array(list(data_jind['labels'] == i)) * 1
		precision, recall, thresholds = metrics.precision_recall_curve(labels, probs)
		AUPR_dic['JIND'].append(metrics.average_precision_score(labels, probs))
		AUPRw_dic['JIND'].append(np.sum(labels) + 1e-5)
		ax.plot(recall, precision, label="JIND")

		probs = list(data_seurat[R_i])
		labels = np.array(list(data_seurat['labels'] == R_i)) * 1
		precision, recall, thresholds = metrics.precision_recall_curve(labels, probs)
		AUPR_dic['Seurat'].append(metrics.average_precision_score(labels, probs))
		AUPRw_dic['Seurat'].append(np.sum(labels) + 1e-5)
		ax.plot(recall, precision, label="Seurat")

		probs = list(data_itclust[i])
		labels = np.array(list(data_itclust['labels'] == i)) * 1
		precision, recall, thresholds = metrics.precision_recall_curve(labels, probs)
		AUPR_dic['ItClust'].append(metrics.average_precision_score(labels, probs))
		AUPRw_dic['ItClust'].append(np.sum(labels) + 1e-5)
		ax.plot(recall, precision, label="ItClust")

		probs = list(data_svm[i])
		labels = np.array(list(data_svm['labels'] == i)) * 1
		precision, recall, thresholds = metrics.precision_recall_curve(labels, probs)
		AUPR_dic['SVM_Rej'].append(metrics.average_precision_score(labels, probs))
		AUPRw_dic['SVM_Rej'].append(np.sum(labels) + 1e-5)
		ax.plot(recall, precision, label="SVM_Rej")

		probs = list(data_actinn[i])
		labels = np.array(list(data_actinn['labels'] == i)) * 1
		precision, recall, thresholds = metrics.precision_recall_curve(labels, probs)
		AUPR_dic['ACTINN'].append(metrics.average_precision_score(labels, probs))
		AUPRw_dic['ACTINN'].append(np.sum(labels) + 1e-5)
		ax.plot(recall, precision, label="ACTINN")

		probs = list(data_scpred[R_i])
		labels = np.array(list(data_scpred['labels'] == R_i)) * 1
		precision, recall, thresholds = metrics.precision_recall_curve(labels, probs)
		AUPR_dic['scPred'].append(metrics.average_precision_score(labels, probs))
		AUPRw_dic['scPred'].append(np.sum(labels) + 1e-5)
		ax.plot(recall, precision, label="scPred")

		ax.set_xlabel("Recall")
		ax.set_ylabel("Precision")
		ax.set_title(f"Cell-type {i}", fontsize=14)

		# plt.grid(b=True, which='major', color='w', linewidth=1.0)
		# plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),fancybox=True, shadow=True, ncol=3, prop={"size":12})
		# plt.tight_layout()
		# # plt.legend()
		# plt.savefig(f"{JIND_outpath}/{i}.pdf")

		# ax.grid(b=True, which='major', color='w', linewidth=1.0)
		# ax.grid(b=True, which='minor', color='w', linewidth=0.5)
		ax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
		ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
		ax.grid(b=True, which='major', color='w', linewidth=1.0)
		# ax.grid(b=True, which='minor', color='w', linewidth=0.1)
	# plt.legend(handles=handles, labels=labels, loc='upper center', bbox_to_anchor=(0.5, -0.1),fancybox=True, shadow=True, ncol=3, prop={"size":12})
	handles, labels = plt.gca().get_legend_handles_labels()
	fig.legend(handles=handles, labels=labels, loc='upper center', ncol=3, fancybox=True, prop={"size":12}, borderaxespad=1.)
	plt.tight_layout()
	plt.subplots_adjust(top=0.92)
	# plt.legend()
	plt.savefig(f"{JIND_outpath}/types.pdf")

	print(sorted(list(set(data_jind['labels']))))
	for i, j in AUPR_dic.items():
		weighted_AUPR = np.sum(np.array(AUPRw_dic[i])/ np.sum(np.array(AUPRw_dic[i])) * np.array(j))
		print(f"{i} mAUPR {np.mean(j):.4f} wAUPR {np.mean(weighted_AUPR):.4f}")






if __name__ == "__main__":
	main()

