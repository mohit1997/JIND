import numpy as np
import pandas as pd


def main():
	lis = ['rawpredicted_label123.txt']#, 'predicted_label2.txt', 'predicted_label3.txt']
	lab_lis = []
	for path in lis:
		lab = pd.read_csv(path, sep='\t', index_col=0)
		lab_lis.append(lab)

	data = pd.concat(lab_lis, axis=0)
	ind = data['pred'] == data['labels']
	ind = ind.values

	data_filt = data[~data['pred'].isin(["Unassigned"])]
	ind_filt = data_filt['pred'] == data_filt['labels']
	ind_filt = ind_filt.values
	print("Accuracy Post {:.4f} Eff {:.4f}".format(np.mean(ind), np.mean(ind_filt)))
	# pdb.set_trace()


if __name__ == "__main__":
	main()
