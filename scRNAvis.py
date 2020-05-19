import numpy as np
import sys, os, pdb
from scRNALib import scRNALib
import matplotlib.pyplot as plt
import multiprocessing
from functools import partial

class scRNAVis:
	def __init__(self, mat, labels, libobj, direc):
		self.class2num = libobj.class2num
		self.num2class = libobj.num2class
		self.reduce_method = libobj.reduce_method
		self.model = None
		self.preprocess = False
		self.dir = direc
		self.obj = libobj
		os.system('mkdir -p {}'.format(self.dir))

		self.gene_names = libobj.gene_names

		self.classes = libobj.classes
		self.shortclasses = [i[:3] + "-" + i[-3:]  for i in self.classes]
		self.n_classes = libobj.n_classes
		libobj.path = self.dir
		
		if libobj.val_stats is None:
			print("The library object doesn't have validation stats")
			sys.exit()
		self.val_stats = libobj.val_stats
		self.mat = mat
		self.labels = labels


	def evaluate(self, mat, labels):
		predictions = self.obj.evaluate(mat, labels)
		return predictions

	def setup(self):
		self.y_pred = self.obj.predict(self.mat)
		self.y_true = np.array([self.class2num[i] for i in self.labels])

		# Freeze the predictions and labels
		self.y_pred.flags.writeable = False
		self.y_true.flags.writeable = False
		print("Setup Complete")

	def display_mean_prob(self):
		probs_train = self.val_stats['pred']
		y_train = self.val_stats['true']

		probs_test = self.y_pred
		y_test = self.y_true

		# Using for Loop
		for klass in range(self.n_classes):
			self.plot_prob(probs_train, y_train, probs_test, y_test, klass)

		# Using multiple cores
		# pool = multiprocessing.Pool(processes=6)
		# func = partial(self.plot_prob, probs_train, y_train, probs_test, y_test)
		# pool.map(func, (i for i in range(self.n_classes)))

	def plot_prob(self, probs_train, y_train, probs_test, y_test, klass):
		factor = 1 + self.n_classes//10
		fig = plt.figure(figsize=(6*factor, 6))
		indices = np.argmax(probs_train, axis=1)==klass
		probs = probs_train[indices]
		y_klass = y_train[indices]
		class_name = self.classes[klass]

		plt.subplot(2, 2, 1)
		if len(indices) != 0:
			probs_TP = probs[y_klass==klass]
			if len(probs_TP) != 0:
				mean = np.mean(probs_TP, axis=0)
				std = np.std(probs_TP, axis=0)
				plt.bar(np.arange(0, len(mean)), mean, yerr=std)
			plt.xticks(np.arange(0, len(self.shortclasses), 1.0), labels=self.shortclasses, rotation=60, ha='right')
			plt.xlabel("Class")
			plt.ylabel("Probability")
			plt.title("Val TP Frac {:.4f}".format(len(probs_TP)/(len(probs)+1e-8)))

			plt.subplot(2, 2, 2)
			probs_FP = probs[y_klass!=klass]
			if len(probs_FP) != 0:
				mean = np.mean(probs_FP, axis=0)
				std = np.std(probs_FP, axis=0)
				plt.bar(np.arange(0, len(mean)), mean, yerr=std)
			plt.xticks(np.arange(0, len(self.shortclasses), 1.0), labels=self.shortclasses, rotation=60, ha='right')
			plt.xlabel("Class")
			plt.ylabel("Probability")
			plt.title("Val FP Frac {:.4f}".format(len(probs_FP)/(len(probs)+1e-8)))

		indices = np.argmax(probs_test, axis=1)==klass
		plt.subplot(2, 2, 3)
		if len(indices) != 0:
			probs = probs_test[indices]
			y_klass = y_test[indices]

			probs_TP = probs[y_klass==klass]
			if len(probs_TP) != 0:
				mean = np.mean(probs_TP, axis=0)
				std = np.std(probs_TP, axis=0)
				plt.bar(np.arange(0, len(mean)), mean, yerr=std)
			plt.xticks(np.arange(0, len(self.shortclasses), 1.0), labels=self.shortclasses, rotation=60, ha='right')
			plt.xlabel("Class")
			plt.ylabel("Probability")
			plt.title("Test TP Frac {:.4f}".format(len(probs_TP)/(len(probs)+1e-8)))

			plt.subplot(2, 2, 4)
			probs_FP = probs[y_klass!=klass]
			if len(probs_FP) != 0:
				mean = np.mean(probs_FP, axis=0)
				std = np.std(probs_FP, axis=0)
				plt.bar(np.arange(0, len(mean)), mean, yerr=std)
			plt.xticks(np.arange(0, len(self.shortclasses), 1.0), labels=self.shortclasses, rotation=60, ha='right')
			plt.xlabel("Class")
			plt.ylabel("Probability")
			plt.title("Test FP Frac {:.4f}".format(len(probs_FP)/(len(probs)+1e-8)))
		fig.subplots_adjust(bottom=0.15, top=0.9, left=0.05, right=0.95, wspace=0.3, hspace=0.7)
		plt.suptitle("Class {}".format(class_name), y=0.98)
		# plt.tight_layout()
		# textstr = '\n'.join([ "{}: {}".format(i, cl) for i, cl in enumerate(self.classes)])
		# props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
		# plt.text(1.3, 1.8, textstr, {'color': 'k', 'fontsize': 10}, bbox=props)
		plt.savefig('{}/probhist_class{}.pdf'.format(self.dir, klass))
		plt.close(fig)

	def display_entropy_plot(self):
		probs_train = self.val_stats['pred']
		y_train = self.val_stats['true']

		probs_test = self.y_pred
		y_test = self.y_true

		# Using for Loop
		for klass in range(self.n_classes):
			self.plot_entropy(probs_train, y_train, probs_test, y_test, klass)

		# Using multiple cores
		# pool = multiprocessing.Pool(processes=6)
		# func = partial(self.plot_entropy, probs_train, y_train, probs_test, y_test)
		# pool.map(func, (i for i in range(self.n_classes)))

	def display_rentropy_plot(self, alpha=2):
		probs_train = self.val_stats['pred']
		y_train = self.val_stats['true']

		probs_test = self.y_pred
		y_test = self.y_true

		# Using for Loop
		for klass in range(self.n_classes):
			self.plot_rentropy(probs_train, y_train, probs_test, y_test, alpha, klass)

		# Using multiple cores
		# pool = multiprocessing.Pool(processes=6)
		# # func = partial(self.plot_entropy, probs_train, y_train, probs_test, y_test, alpha
		# func = partial(self.plot_rentropy, probs_train, y_train, probs_test, y_test, alpha)
		# pool.map(func, (i for i in range(self.n_classes)))

	def plot_entropy(self, probs_train, y_train, probs_test, y_test, klass):
		
		fig = plt.figure()
		indices = np.argmax(probs_train, axis=1)==klass
		probs = probs_train[indices]
		y_klass = y_train[indices]
		class_name = self.classes[klass]
		
		plt.subplot(2, 2, 1)
		if len(indices) != 0:
			probs_TP = probs[y_klass==klass]
			if len(probs_TP) != 0:
				H = entropy(probs_TP.transpose(), base=2)
				plt.hist(H, density=False, bins=30)
			plt.xlabel('Entropy')
			plt.ylabel('Counts')
			plt.title("Train TP Frac {:.4f}".format(len(probs_TP)/(len(probs)+1e-8)))

			plt.subplot(2, 2, 2)
			probs_FP = probs[y_klass!=klass]
			if len(probs_FP) != 0:
				H = entropy(probs_FP.transpose(), base=2)
				plt.hist(H, density=False, bins=30)
			plt.xlabel('Entropy')
			plt.ylabel('Counts')
			plt.title("Train FP Frac {:.4f}".format(len(probs_FP)/(len(probs)+1e-8)))

		indices = np.argmax(probs_test, axis=1)==klass
		probs = probs_test[indices]
		y_klass = y_test[indices]

		plt.subplot(2, 2, 3)
		if len(indices) != 0:
			probs_TP = probs[y_klass==klass]
			if len(probs_TP) != 0:
				H = entropy(probs_TP.transpose(), base=2)
				plt.hist(H, density=False, bins=30)
			plt.xlabel('Entropy')
			plt.ylabel('Counts')
			plt.title("Test TP Frac {:.4f}".format(len(probs_TP)/(len(probs)+1e-8)))

			plt.subplot(2, 2, 4)
			probs_FP = probs[y_klass!=klass]
			if len(probs_FP) != 0:
				H = entropy(probs_FP.transpose(), base=2)
				plt.hist(H, density=False, bins=30)
			plt.xlabel('Entropy')
			plt.ylabel('Counts')
			plt.title("Test FP Frac {:.4f}".format(len(probs_FP)/(len(probs)+1e-8)))
		fig.subplots_adjust(bottom=0.1, top=0.9, wspace=0.3, hspace=0.6)
		plt.suptitle("Class {}".format(class_name), y=0.98)
		plt.savefig('{}/entropyhist_class{}.pdf'.format(self.dir, klass))
		plt.close(fig)

	def plot_rentropy(self, probs_train, y_train, probs_test, y_test, alpha, klass):
		
		fig = plt.figure()
		indices = np.argmax(probs_train, axis=1)==klass
		probs = probs_train[indices]
		y_klass = y_train[indices]
		class_name = self.classes[klass]
		
		plt.subplot(2, 2, 1)
		if len(indices) != 0:
			probs_TP = probs[y_klass==klass]
			if len(probs_TP) != 0:
				H = renyi_entropy(probs_TP.transpose(), base=2, alpha=alpha)
				plt.hist(H, density=False, bins=30)
			plt.xlabel('Renyi Entropy $ \\alpha={} $'.format(alpha))
			plt.ylabel('Counts')
			plt.title("Train TP Frac {:.4f}".format(len(probs_TP)/(len(probs)+1e-8)))

			plt.subplot(2, 2, 2)
			probs_FP = probs[y_klass!=klass]
			if len(probs_FP) != 0:
				H = renyi_entropy(probs_FP.transpose(), base=2, alpha=alpha)
				plt.hist(H, density=False, bins=30)
			plt.xlabel('Renyi Entropy $ \\alpha={} $'.format(alpha))
			plt.ylabel('Counts')
			plt.title("Train FP Frac {:.4f}".format(len(probs_FP)/(len(probs)+1e-8)))

		indices = np.argmax(probs_test, axis=1)==klass
		probs = probs_test[indices]
		y_klass = y_test[indices]

		plt.subplot(2, 2, 3)
		if len(indices) != 0:
			probs_TP = probs[y_klass==klass]
			if len(probs_TP) != 0:
				H = renyi_entropy(probs_TP.transpose(), base=2, alpha=alpha)
				plt.hist(H, density=False, bins=30)
			plt.xlabel('Renyi Entropy $ \\alpha={} $'.format(alpha))
			plt.ylabel('Counts')
			plt.title("Test TP Frac {:.4f}".format(len(probs_TP)/(len(probs)+1e-8)))

			plt.subplot(2, 2, 4)
			probs_FP = probs[y_klass!=klass]
			if len(probs_FP) != 0:
				H = renyi_entropy(probs_FP.transpose(), base=2, alpha=alpha)
				plt.hist(H, density=False, bins=30)
			plt.xlabel('Renyi Entropy $ \\alpha={} $'.format(alpha))
			plt.ylabel('Counts')
			plt.title("Test FP Frac {:.4f}".format(len(probs_FP)/(len(probs)+1e-8)))
		fig.subplots_adjust(bottom=0.1, top=0.9, wspace=0.3, hspace=0.6)
		plt.suptitle("Class {}".format(class_name), y=0.98)
		plt.savefig('{}/rentropyhist_class{}.pdf'.format(self.dir, klass))
		plt.close(fig)

	def display_KLdiv(self):
		probs_train = self.val_stats['pred']
		y_train = self.val_stats['true']

		probs_test = self.y_pred
		y_test = self.y_true

		for klass in range(self.n_classes):
			self.plot_KLdiv(probs_train, y_train, probs_test, y_test, klass)

		# Using multiple cores
		# pool = multiprocessing.Pool(processes=6)
		# # func = partial(self.plot_entropy, probs_train, y_train, probs_test, y_test, alpha
		# func = partial(self.plot_KLdiv, probs_train, y_train, probs_test, y_test)
		# pool.map(func, (i for i in range(self.n_classes)))

	def plot_KLdiv(self, probs_train, y_train, probs_test, y_test, klass):
		fig = plt.figure()
		indices = np.argmax(probs_train, axis=1)==klass
		class_name = self.classes[klass]
		if len(indices) != 0:
			probs = probs_train[indices]
			y_klass = y_train[indices]
			
			probs_TP_mean = np.mean(probs[y_klass==klass], axis=0, keepdims=True)

			indices_t = np.argmax(probs_test, axis=1)==klass
			if len(indices_t) != 0:
				probs = probs_test[indices_t]
				y_klass = y_test[indices_t]

				plt.subplot(2, 1, 1)
				probs_TP = probs[y_klass==klass]
				H = KLDiv(probs_TP, probs_TP_mean)
				plt.hist(H, density=False, bins=30)
				plt.xlabel('KL Div')
				plt.ylabel('Counts')
				plt.title("Test TP Frac {:.4f}".format(len(probs_TP)/(len(probs)+1e-8)))

				plt.subplot(2, 1, 2)
				probs_FP = probs[y_klass!=klass]
				H = KLDiv(probs_FP, probs_TP_mean)
				plt.hist(H, density=False, bins=30)
				plt.xlabel('KL Div')
				plt.ylabel('Counts')
				plt.title("Test FP Frac {:.4f}".format(len(probs_FP)/(len(probs)+1e-8)))
				fig.subplots_adjust(bottom=0.1, top=0.9, wspace=0.3, hspace=0.6)
				plt.suptitle("Class {}".format(class_name), y=0.98)
				plt.savefig('{}/KLDiv_class{}.pdf'.format(self.dir, klass))
				plt.close(fig)


def entropy(inp, base=2):
	probs = inp/(np.sum(inp, axis=0, keepdims=True)+1e-5) + 1e-5
	H = - np.sum(probs * np.log2(probs), axis=0)/np.log2(base)
	return H

def renyi_entropy(inp, base=2, alpha=2):
	probs = inp/(np.sum(inp, axis=0, keepdims=True)+1e-5) + 1e-5
	H = 1./(1-alpha) * np.log2(np.sum(probs**alpha, axis=0))/np.log2(base)
	return H

def KLDiv(probs, reference, base=2):
	probs = probs/(np.sum(probs, axis=1, keepdims=True)+1e-5) + 1e-5
	reference = reference/(np.sum(reference, axis=1, keepdims=True)+1e-5)

	H = np.sum(probs*np.log2(probs/(reference+1e-5)), axis=1)
	return H


def main():
	import pickle
	with open('Human_annotated', 'rb') as f:
		data = pickle.load(f)
	cell_ids = np.arange(len(data))
	np.random.seed(0)
	np.random.shuffle(cell_ids)
	l = int(0.7*len(cell_ids))

	train_data = data.iloc[cell_ids[:l]]
	train_labels = train_data['labels']
	train_gene_mat =  train_data.drop('labels', 1)

	test_data = data.iloc[cell_ids[l:]]
	test_labels = test_data['labels']
	test_gene_mat =  test_data.drop('labels', 1)


	with open('human_results/scRNALib_obj.pkl', 'rb') as f:
		obj = pickle.load(f)

	visobj = scRNAVis(test_gene_mat, test_labels, obj, direc="human_vis")
	visobj.setup()
	# visobj.display_mean_prob()
	# visobj.display_entropy_plot()
	visobj.display_rentropy_plot(alpha=2)
	# visobj.display_KLdiv()
	# pdb.set_trace()


if __name__ == "__main__":
	main()

