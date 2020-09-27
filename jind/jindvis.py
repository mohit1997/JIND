import numpy as np
import pandas as pd
import sys, os, pdb
from .jindlib import JindLib
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt
import multiprocessing
from functools import partial
import pickle
import seaborn as sns

class JindVis:
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
		self.embeddings = {}


	def evaluate(self, mat, labels):
		predictions = self.obj.evaluate(mat, labels)
		return predictions

	def setup(self, test=False):
		self.y_pred = self.obj.predict(self.mat, test=test)
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

	def reduce(self, method="tsne"):
		pca = PCA(n_components=50)
		dim_size = 5000
		feats = self.mat.values
		self.variances = np.argsort(-np.var(feats, axis=0))[:dim_size]
		self.reduced_features = feats[:, self.variances]
		pca_feats = pca.fit_transform(self.reduced_features)
		if method == "tsne":
			self.embeddings[method] = TSNE(n_components=2, verbose=1, n_jobs=-1, perplexity=50).fit_transform(pca_feats)
		elif method == "umap":
			fit = umap.UMAP()
			self.embeddings[method] = fit.fit_transform(pca_feats)
		else:
			print("choose tsne/umap")

		return

	def plot_2d(self, method="tsne", test=False):
		out = self.obj.evaluate(self.mat, self.labels, frac=0.05, name=None, test=test)
		raw_preds = list([self.num2class[i] for i in np.argmax(self.y_pred, axis=1)])
		preds = list(out['predictions'])
		df = pd.DataFrame({'Predictions': preds,
						'Raw Predictions': raw_preds,
						'Labels': self.labels,
						"preds_labels": ["{}_{}".format(i, j) for i,j in zip(preds, list(self.labels))]
						})
		if method in self.embeddings.keys():
			df["{}_x".format(method)] = self.embeddings[method][:, 0]
			df["{}_y".format(method)] = self.embeddings[method][:, 1]

			check = list(df['Predictions'] == df['Labels'])
			marker_list = ['correct' if i else 'miss' for i in check]
			marker_list = ['Unassigned' if j == "Unassigned" else "Assigned" for i, j in zip(marker_list, preds)]
			df['|Assignment|'] = marker_list

			check = list(df['Raw Predictions'] == df['Labels'])
			size_list = ["Correct" if i else "Miss" for i in check]
			df['|Evaluation|'] = size_list

			df = df.sort_values('|Assignment|')

			color_list=['r' if i else 'b' for i in check]

			plt.figure()
			order = list(set(df['Predictions']))
			order = sorted(order, key=str.casefold)

			g = sns.scatterplot(x="{}_x".format(method), y="{}_y".format(method), hue='Predictions', data=df, hue_order=order)
			plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
			plt.tight_layout()
			plt.savefig("{}/{}_pred.pdf".format(self.dir, method))

			plt.figure()
			order = list(set(df['Labels']))
			order = sorted(order, key=str.casefold)

			g = sns.scatterplot(x="{}_x".format(method), y="{}_y".format(method), hue='Labels', data=df, hue_order=order)
			plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
			plt.tight_layout()
			plt.savefig("{}/{}_true.pdf".format(self.dir, method))

			plt.figure(figsize=(16, 8))
			plt.subplot(1, 2, 1)			
			order = list(set(df['Raw Predictions']).union(set(["Unassigned"])))
			order = sorted(order, key=str.casefold)

			g = sns.scatterplot(x="{}_x".format(method), y="{}_y".format(method), hue='Predictions', data=df, hue_order=order, size='|Assignment|', size_order=['Unassigned', 'Assigned'], sizes=(20, 80))
			# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
			handles, labels = g.get_legend_handles_labels()
			g.get_legend().remove()
			plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),fancybox=True, shadow=True, ncol=3, markerscale=2., fontsize=10)
			plt.title("Predictions", fontsize= 30)
			
			plt.subplot(1, 2, 2)
			order = list(set(df['Raw Predictions']))
			order = sorted(order, key=str.casefold)

			g = sns.scatterplot(x="{}_x".format(method), y="{}_y".format(method), hue='Raw Predictions', data=df, hue_order=order, markers=["o", "X"], style='|Evaluation|', style_order=["Correct", "Miss"], size='|Assignment|', size_order=['Unassigned', 'Assigned'], sizes=(20, 80))
			# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
			plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),fancybox=True, shadow=True, ncol=3, markerscale=2., fontsize=10)
			plt.title("Raw Predictions", fontsize= 30)
			plt.tight_layout()

			plt.savefig("{}/{}_pred_true.pdf".format(self.dir, method))

	def to_pickle(self, name):
		self.mat = None
		self.reduced_features = None
		with open('{}/{}'.format(self.dir, name), 'wb') as f:
			pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)


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
	data = pd.read_pickle('data/blood_annotated.pkl')

	cell_ids = np.arange(len(data))
	np.random.seed(0)
	# np.random.shuffle(cell_ids)
	# l = int(0.5*len(cell_ids))

	batches = list(set(data['batch']))
	batches.sort()
	l = int(0.5*len(batches))
	train_data = data[data['batch'].isin(batches[0:1])].copy()
	test_data = data[data['batch'].isin(batches[1:2])].copy()

	train_labels = train_data['labels']
	# train_gene_mat =  train_data.drop(['labels', 'batch'], 1)

	test_labels = test_data['labels']
	# test_gene_mat =  test_data.drop(['labels', 'batch'], 1)

	common_labels = list(set(train_labels) & set(test_labels))

	train_data = train_data[train_data['labels'].isin(common_labels)].copy()
	test_data = data[data['batch'].isin(batches[1:2])].copy()
	test_data = test_data[test_data['labels'].isin(common_labels)].copy()
	# test_data = test_data[test_data['labels'].isin(common_labels)].copy()

	train_labels = train_data['labels']
	train_gene_mat =  train_data.drop(['labels', 'batch'], 1)

	test_labels = test_data['labels']
	test_gene_mat =  test_data.drop(['labels', 'batch'], 1)

	# assert (set(train_labels)) == (set(test_labels))
	common_labels.sort()
	testing_set = list(set(test_labels))
	testing_set.sort()
	print("Selected Common Labels", common_labels)
	print("Test Labels", testing_set)


	with open('blood_results/scRNALib_objbr.pkl', 'rb') as f:
		obj = pickle.load(f)

	visobj = JindVis(test_gene_mat, test_labels, obj, direc="blood_vis")
	visobj.setup(test=True)
	visobj.reduce("tsne")
	visobj.reduce("umap")
	# visobj.display_mean_prob()
	# visobj.display_entropy_plot()
	# visobj.display_rentropy_plot(alpha=2)
	# visobj.display_KLdiv()
	# pdb.set_trace()

	visobj.mat = None
	visobj.reduced_features = None

	with open('blood_results/scRNAvis_obj.pkl', 'wb') as f:
		pickle.dump(visobj, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
	main()

