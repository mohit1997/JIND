from torch.utils.data import DataLoader, Dataset
import numpy as np
from itertools import product

class DataLoaderCustom(Dataset):
	def __init__(self, features, labels=None, weights=None, transform=None):
		"""
			Args:
				features (string): np array of features.
				transform (callable, optional): Optional transform to be applied
					on a sample.
		"""
		self.features = features
		self.labels = labels
		self.weights = weights
		self.transform = transform

	def __len__(self):
		return len(self.features)

	def __getitem__(self, idx):
		sample = {}
		sample['x'] = self.features[idx].astype('float32')
		if self.labels is not None:
			sample['y'] = self.labels[idx]
			if self.weights is not None:
				sample['w'] = self.weights[self.labels[idx]].astype('float32')
		return sample


def _check_targets(y_true, y_pred):
	check_consistent_length(y_true, y_pred)
	type_true = type_of_target(y_true)
	type_pred = type_of_target(y_pred)

	y_type = {type_true, type_pred}
	if y_type == {"binary", "multiclass"}:
		y_type = {"multiclass"}

	if len(y_type) > 1:
		raise ValueError("Classification metrics can't handle a mix of {0} "
						 "and {1} targets".format(type_true, type_pred))

	# We can't have more than one value on y_type => The set is no more needed
	y_type = y_type.pop()

	# No metrics support "multiclass-multioutput" format
	if (y_type not in ["binary", "multiclass", "multilabel-indicator"]):
		raise ValueError("{0} is not supported".format(y_type))

	if y_type in ["binary", "multiclass"]:
		y_true = column_or_1d(y_true)
		y_pred = column_or_1d(y_pred)
		if y_type == "binary":
			unique_values = np.union1d(y_true, y_pred)
			if len(unique_values) > 2:
				y_type = "multiclass"

	if y_type.startswith('multilabel'):
		y_true = csr_matrix(y_true)
		y_pred = csr_matrix(y_pred)
		y_type = 'multilabel-indicator'

	return y_type, y_true, y_pred


def normalize(cm, normalize=None, epsilon=1e-8):
	with np.errstate(all='ignore'):
		if normalize == 'true':
			cm = cm / (cm.sum(axis=1, keepdims=True) + epsilon)
		elif normalize == 'pred':
			cm = cm / (cm.sum(axis=0, keepdims=True) + epsilon)
		elif normalize == 'all':
			cm = cm / (cm.sum() + epsilon)
		cm = np.nan_to_num(cm)
	return cm


class ConfusionMatrixPlot:

	def __init__(self, confusion_matrix, display_labels):
		self.confusion_matrix = confusion_matrix
		self.display_labels = display_labels.copy()
		self.displabelsx = display_labels.copy()
		if "Novel" in display_labels:
			self.displabelsx.remove("Novel")
		self.displabelsy = display_labels.copy()
		self.displabelsy.remove("Unassigned")

	def plot(self, include_values=True, cmap='viridis',
			 xticks_rotation='vertical', values_format=None, ax=None, fontsize=13):

		import matplotlib.pyplot as plt

		if ax is None:
			fig, ax = plt.subplots()
		else:
			fig = ax.figure

		cm = self.confusion_matrix
		n_classes = cm.shape[0]
		self.im_ = ax.imshow(cm, interpolation='nearest', cmap=cmap)
		self.text_ = None

		cmap_min, cmap_max = self.im_.cmap(0), self.im_.cmap(256)

		if include_values:
			self.text_ = np.empty_like(cm, dtype=object)
			if values_format is None:
				values_format = '.2g'

			# print text with appropriate color depending on background
			thresh = (cm.max() + cm.min()) / 2.0
			for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
				color = cmap_max if cm[i, j] < thresh else cmap_min
				self.text_[i, j] = ax.text(j, i,
										   format(cm[i, j], values_format),
										   ha="center", va="center",
										   color=color, fontsize=fontsize)

		fig.colorbar(self.im_, ax=ax)
		ax.set(xticks=np.arange(cm.shape[1]),
			   yticks=np.arange(cm.shape[0]),
			   )
		ax.set_xticklabels(self.displabelsx[:cm.shape[1]], fontsize=fontsize)
		ax.set_yticklabels(self.displabelsy[:cm.shape[0]], fontsize=fontsize)
		ax.set_xlabel(xlabel="Predicted label", fontsize=fontsize+2)
		ax.set_ylabel(ylabel="True label", fontsize=fontsize+2)

		ax.set_ylim((n_classes - 0.5, -0.5))
		plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)

		self.figure_ = fig
		self.ax_ = ax
		return self

def compute_ap(gts, preds):
	aps = []
	for i in range(preds.shape[1]):
		ap, prec, rec = calc_pr(gts == i, preds[:,i:i+1])
		aps.append(ap)
	aps = np.array(aps)
	return np.nan_to_num(aps)

def calc_pr(gt, out, wt=None):
	gt = gt.astype(np.float64).reshape((-1,1))
	out = out.astype(np.float64).reshape((-1,1))

	tog = np.concatenate([gt, out], axis=1)*1.
	ind = np.argsort(tog[:,1], axis=0)[::-1]
	tog = tog[ind,:]
	cumsumsortgt = np.cumsum(tog[:,0])
	cumsumsortwt = np.cumsum(tog[:,0]-tog[:,0]+1)
	prec = cumsumsortgt / (cumsumsortwt + 1e-8)
	rec = cumsumsortgt / (np.sum(tog[:,0]) + 1e-8)
	ap = voc_ap(rec, prec)
	return ap, rec, prec

def voc_ap(rec, prec):
	rec = rec.reshape((-1,1))
	prec = prec.reshape((-1,1))
	z = np.zeros((1,1)) 
	o = np.ones((1,1))
	mrec = np.vstack((z, rec, o))
	mpre = np.vstack((z, prec, z))

	mpre = np.maximum.accumulate(mpre[::-1])[::-1]
	I = np.where(mrec[1:] != mrec[0:-1])[0]+1;
	ap = np.sum((mrec[I] - mrec[I-1])*mpre[I])
	return ap