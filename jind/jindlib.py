import numpy as np
import torch, sys, os, pdb
import pandas as pd
from torch import optim
from torch.autograd import Variable
from .utils import DataLoaderCustom, ConfusionMatrixPlot, compute_ap, normalize
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from .models import Classifier, Discriminator, ClassifierBig
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import DBSCAN, OPTICS, SpectralClustering, AgglomerativeClustering
import seaborn as sns
import pickle
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from scipy import optimize
import umap

class JindLib:
	global MODEL_WIDTH, LDIM, GLDIM
	MODEL_WIDTH = 1500
	LDIM = 256
	GLDIM = 512

	def __init__(self, gene_mat, cell_labels, path):
		self.class2num = None
		self.num2class = None
		self.reduced_features = None
		self.reduce_method = None
		self.model = None
		self.modelftuned = None
		self.count_normalize = False
		self.logt = False
		self.path = path
		os.system('mkdir -p {}'.format(self.path))

		self.raw_features = gene_mat.values
		self.cell_ids = list(gene_mat.index)
		self.gene_names = list(gene_mat.columns)
		

		classes = list(set(cell_labels))
		classes.sort()
		self.classes = classes
		self.n_classes = len(classes)

		self.class2num = class2num = {c: i for (i, c) in enumerate(classes)}
		self.class2num['Unassigned'] = self.n_classes

		self.num2class = num2class = {i: c for (i, c) in enumerate(classes)}
		self.num2class[self.n_classes] = 'Unassigned'

		self.labels = np.array([class2num[i] for i in cell_labels])
		self.val_stats = None
		self.scaler = None

	def preprocess(self, count_normalize=False, target_sum=1e4, logt=True):
		self.logt = logt
		self.count_normalize = count_normalize
		if count_normalize:
			print('Normalizing counts ...')
			self.raw_features = self.raw_features / (np.sum(self.raw_features, axis=1, keepdims=True) + 1e-5) * target_sum
		if logt:
			print('Applying log transformation ...')
			self.raw_features = np.log(1 + self.raw_features)


	def dim_reduction(self, num_features=5000, method='var', save_as=None):
		dim_size = num_features
		self.reduce_method = method

		if method.lower() == 'pca':
			print('Performing PCA ...')
			self.pca = PCA(n_components=dim_size)
			self.reduced_features = self.pca.fit_transform(self.raw_features)
			if save_as is not None:
				np.save('{}_{}'.format(save_as, method), self.reduced_features)

		elif method.lower() == 'var':
			print('Variance based reduction ...')
			self.variances = np.argsort(-np.var(self.raw_features, axis=0))[:dim_size]
			self.reduced_features = self.raw_features[:, self.variances]
			self.selected_genes = [self.gene_names[i] for i in self.variances]
			if save_as is not None:
				np.save('{}_{}'.format(save_as, method), self.reduced_features)

	def normalize(self):
		scaler = StandardScaler()
		print(self.reduced_features.shape)
		self.reduced_features = scaler.fit_transform(self.reduced_features)
		self.scaler = scaler


	def train_classifier(self, config=None, cmat=True):
		if config is None:
			config = {'val_frac': 0.2, 'seed': 0, 'batch_size': 128, 'cuda': False,
					'epochs': 15}
		
		if self.reduced_features is not None:
			features = self.reduced_features
		else:
			features = self.raw_features

		labels = self.labels

		values, counts = np.unique(labels, return_counts=True)

		

		torch.manual_seed(config['seed'])
		torch.cuda.manual_seed(config['seed'])
		np.random.seed(config['seed'])
		torch.backends.cudnn.deterministic = True

		if np.min(counts) > 1:
			X_train, X_val, y_train, y_val = train_test_split(
				features, labels, test_size=config['val_frac'], stratify=labels, shuffle=True, random_state=config['seed'])
		else:
			X_train, X_val, y_train, y_val = train_test_split(
				features, labels, test_size=config['val_frac'], shuffle=True, random_state=config['seed'])

		X_train, X_val, y_train, y_val = train_test_split(
			features, labels, test_size=config['val_frac'], stratify=labels, shuffle=True, random_state=config['seed'])

		train_dataset = DataLoaderCustom(X_train, y_train)
		val_dataset = DataLoaderCustom(X_val, y_val)


		use_cuda = config['cuda']
		use_cuda = use_cuda and torch.cuda.is_available()

		device = torch.device("cuda" if use_cuda else "cpu")
		kwargs = {'num_workers': 4, 'pin_memory': False} if use_cuda else {}

		train_loader = torch.utils.data.DataLoader(train_dataset,
										   batch_size=config['batch_size'],
										   shuffle=True, **kwargs)

		val_loader = torch.utils.data.DataLoader(val_dataset,
										   batch_size=config['batch_size'],
										   shuffle=False, **kwargs)

		weights, n_classes = self.get_class_weights()
		class_weights = torch.FloatTensor(weights).to(device)

		criterion = torch.nn.NLLLoss(weight=class_weights)

		model = Classifier(X_train.shape[1], LDIM, MODEL_WIDTH, n_classes).to(device)
		optimizer = optim.Adam(model.parameters(), lr=1e-3)
		sch = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, threshold=0.05, verbose=True)

		logger = {'tloss': [], 'val_acc': []}
		best_val_acc = 0.
		for epoch in range(config['epochs']):
			c, s = 0, 0
			pBar = tqdm(train_loader)
			model.train()
			for sample in pBar:
				x = sample['x'].to(device)
				y = sample['y'].to(device)
				
				optimizer.zero_grad()
				p = model.predict(x)
				loss = criterion(p, y)
				# print(loss)
				s = ((s*c)+(float(loss.item())*len(p)))/(c+len(p))
				c += len(p)
				pBar.set_description('Epoch {} Train: '.format(epoch) +str(round(float(s),4)))
				loss.backward()
				optimizer.step()
			logger['tloss'].append(s)
			sch.step(s)

			model.eval()
			y_pred, y_true = [], []
			with torch.no_grad():
				for sample in val_loader:
					x = sample['x'].to(device)
					y = sample['y'].to(device)
					
					p = model.predict_proba(x)
					y_pred.append(p.cpu().detach().numpy())
					y_true.append(y.cpu().detach().numpy())
			y_pred = np.concatenate(y_pred)
			y_true = np.concatenate(y_true)

			val_acc = (y_true == y_pred.argmax(axis=1)).mean()
			logger['val_acc'].append(val_acc)
			print("Validation Accuracy {:.4f}".format(val_acc))
			if val_acc > best_val_acc:
				# print('Model improved')
				best_val_acc = val_acc
				torch.save(model.state_dict(), self.path+"/best.pth")
				self.val_stats = {'pred': y_pred, 'true': y_true}

		if cmat:
			# Plot validation confusion matrix
			self.plot_cfmt(self.val_stats['pred'], self.val_stats['true'], 0.05, 'val_cfmt.pdf')

		# Finally keep the best model
		model.load_state_dict(torch.load(self.path+"/best.pth"))
		self.model = model
		self.model.eval()

			# sys.exit()

	def get_class_weights(self):
		unique, counts = np.unique(self.labels, return_counts=True)
		counts = counts/np.sum(counts)
		weights = 2./(0.01+counts) / len(unique)
		return weights, len(unique)

	def load_model(self, path):
		_, n_classes = self.get_class_weights()
		model = Classifier(self.reduced_features.shape[1], 256, MODEL_WIDTH, n_classes)
		model.load_state_dict(torch.load(path, map_location='cpu'))
		self.model = model

	def get_features(self, gene_mat):
		features = gene_mat.values
		if self.count_normalize:
			features = features / (np.sum(features, axis=1, keepdims=True) + 1e-5) * 1e4
		if self.logt:
			features = np.log(1+features)
		if self.reduce_method is not None:
			if self.reduce_method == "Var":
				selected_genes = [gene_mat.columns[i] for i in self.variances]
				if selected_genes != self.selected_genes:
					print("Reorder the genes for the target batch in the same order as the source batch")
					sys.exit()
				features = features[:, self.variances]
			elif self.reduce_method == "PCA":
				features = self.pca.transform(features)
		if self.scaler is not None:
			self.test_scaler = StandardScaler()
			features = self.test_scaler.fit_transform(features)

		return features

	def predict(self, test_gene_mat, test=False, return_names=False):
		
		features = self.get_features(test_gene_mat)

		test_dataset = DataLoaderCustom(features)

		use_cuda = torch.cuda.is_available()
		device = torch.device("cuda" if use_cuda else "cpu")
		kwargs = {'num_workers': 4, 'pin_memory': False} if use_cuda else {}

		test_loader = torch.utils.data.DataLoader(test_dataset,
										   batch_size=512,
										   shuffle=False, **kwargs)

		if test == "modelftuned":
			model = self.modelftuned.to(device)
		elif test:
			model = self.test_model.to(device)
		else:
			model = self.model.to(device)

		model.eval()

		y_pred, y_true = [], []
		with torch.no_grad():
			for sample in test_loader:
				x = sample['x'].to(device)
				
				p = model.predict_proba(x)
				y_pred.append(p.cpu().detach().numpy())
		y_pred = np.concatenate(y_pred)
		if return_names:
			preds = np.argmax(y_pred, axis=1)
			predictions = [self.num2class[i] for i in preds]
			predicted_label = pd.DataFrame({"cellname":test_gene_mat.index, "predictions":predictions})
			return predicted_label
		return y_pred

	def get_encoding(self, test_gene_mat, test=False):
		
		features = self.get_features(test_gene_mat)

		test_dataset = DataLoaderCustom(features)

		use_cuda = torch.cuda.is_available()
		device = torch.device("cuda" if use_cuda else "cpu")
		kwargs = {'num_workers': 4, 'pin_memory': False} if use_cuda else {}

		test_loader = torch.utils.data.DataLoader(test_dataset,
										   batch_size=512,
										   shuffle=False, **kwargs)

		if test:
			model = self.test_model.to(device)
		else:
			model = self.model.to(device)

		model.eval()

		y_pred, y_true = [], []
		with torch.no_grad():
			for sample in test_loader:
				x = sample['x'].to(device)
				if test:
					p, _ = model.get_repr(x)
				else:
					p = model.get_repr(x)
				y_pred.append(p.cpu().detach().numpy())
		y_pred = np.concatenate(y_pred, axis=0)

		return y_pred

	def get_filtered_prediction(self, test_gene_mat, frac=0.05, test=False):
		y_pred = self.predict(test_gene_mat, test=test)

		if frac != 0:
			preds = self.filter_pred(y_pred, frac)
		else:
			preds = np.argmax(y_pred, axis=1)

		predictions = [self.num2class[i] for i in preds]
		raw_predictions = [self.num2class[i] for i in np.argmax(y_pred, axis=1)]
		
		dic1 = {"cellname": test_gene_mat.index,
				"raw_predictions": raw_predictions,
				"predictions": predictions}

		dic2 = {self.num2class[i]: list(y_pred[:, i]) for i in range(self.n_classes)}

		dic = {**dic1, **dic2}

		predicted_label = pd.DataFrame(dic)

		predicted_label = predicted_label.set_index("cellname")

		return predicted_label



	def evaluate(self, test_gene_mat, test_labels, frac=0.05, name=None, test=False, return_log=False):
		y_pred = self.predict(test_gene_mat, test=test)
		y_true = np.array([self.class2num[i] if (i in self.class2num.keys()) else (self.n_classes + 1) for i in test_labels])
		if frac != 0:
			preds = self.filter_pred(y_pred, frac)
		else:
			preds = np.argmax(y_pred, axis=1)
		pretest_acc = (y_true == np.argmax(y_pred, axis=1)).mean() 
		test_acc = (y_true == preds).mean()
		ind = preds != self.n_classes
		pred_acc = (y_true[ind] == preds[ind]).mean()
		filtered = 1 - np.mean(ind)
		print('Test Acc Raw {:.4f} Eff {:.4f} Rej {:.4f}'.format(pretest_acc, pred_acc, filtered))

		if name is not None:
			cm = normalize(confusion_matrix(y_true,
							preds,
							labels=np.arange(0, max(np.max(y_true)+1, np.max(preds)+1, self.n_classes+1))
							),
							normalize='true')
			cm = np.delete(cm, (self.n_classes), axis=0)
			if cm.shape[1] > (self.n_classes+1):
				cm = np.delete(cm, (self.n_classes+1), axis=1)
			aps = np.zeros((len(cm), 1))
			aps[:self.n_classes] = np.array(compute_ap(y_true, y_pred)).reshape(-1, 1)
			cm = np.concatenate([cm, aps], axis=1)

			class_labels = list(self.class2num.keys()) +['Novel'] + ['AP']
			cm_ob = ConfusionMatrixPlot(cm, class_labels)
			factor = max(1, len(cm) // 10)
			fig = plt.figure(figsize=(10*factor,8*factor))
			cm_ob.plot(values_format='0.2f', ax=fig.gca())

			APs = aps[:self.n_classes]
			mAP = np.true_divide(aps.sum(), (aps!=0).sum())
			plt.title('Accuracy Raw {:.3f} Eff {:.3f} Rej {:.3f} mAP {:.3f}'.format(pretest_acc, pred_acc, filtered, mAP), fontsize=16)
			plt.tight_layout()
			plt.savefig('{}/{}'.format(self.path, name))

		predictions = [self.num2class[i] for i in preds]
		raw_predictions = [self.num2class[i] for i in np.argmax(y_pred, axis=1)]
		
		dic1 = {"cellname": test_gene_mat.index,
				"raw_predictions": raw_predictions,
				"predictions": predictions,
				"labels": test_labels}

		dic2 = {self.num2class[i]: list(y_pred[:, i]) for i in range(self.n_classes)}

		dic = {**dic1, **dic2}

		predicted_label = pd.DataFrame(dic)

		predicted_label = predicted_label.set_index("cellname")

		if return_log:
			return predicted_label, 'Test Acc Raw {:.4f} Eff {:.4f} Rej {:.4f}'.format(pretest_acc, pred_acc, filtered)
		return predicted_label

	def plot_cfmt(self, y_pred, y_true, frac=0.05, name=None):
		if frac != 0:
			preds = self.filter_pred(y_pred, frac)
		else:
			preds = np.argmax(y_pred, axis=1)
		pretest_acc = (y_true == np.argmax(y_pred, axis=1)).mean() 
		test_acc = (y_true == preds).mean()
		ind = preds != self.n_classes
		pred_acc = (y_true[ind] == preds[ind]).mean()
		filtered = 1 - np.mean(ind)

		if name is not None:
			cm = normalize(confusion_matrix(y_true,
							preds,
							labels=np.arange(0, max(np.max(y_true)+1, np.max(preds)+1, self.n_classes+1))
							),
							normalize='true')
			cm = np.delete(cm, (self.n_classes), axis=0)
			if cm.shape[1] > (self.n_classes+1):
				cm = np.delete(cm, (self.n_classes+1), axis=1)
			aps = np.zeros((len(cm), 1))
			aps[:self.n_classes] = np.array(compute_ap(y_true, y_pred)).reshape(-1, 1)
			cm = np.concatenate([cm, aps], axis=1)

			class_labels = list(self.class2num.keys()) +['Novel'] + ['AP']
			cm_ob = ConfusionMatrixPlot(cm, class_labels)
			factor = max(1, len(cm) // 10)
			fig = plt.figure(figsize=(10*factor,8*factor))
			cm_ob.plot(values_format='0.2f', ax=fig.gca())

			APs = aps[:self.n_classes]
			mAP = np.true_divide(aps.sum(), (aps!=0).sum())
			plt.title('Accuracy Raw {:.3f} Eff {:.3f} Rej {:.3f} mAP {:.3f}'.format(pretest_acc, pred_acc, filtered, mAP), fontsize=16)

			plt.tight_layout()
			plt.savefig('{}/{}'.format(self.path, name))

	def generate_cfmt(self, pred_labels, test_labels, name=None):
		preds = np.array([self.class2num[i] for i in pred_labels])
		y_true = np.array([self.class2num[i] if (i in self.class2num.keys()) else (self.n_classes + 1) for i in test_labels])

		pretest_acc = (y_true == preds).mean() 
		test_acc = (y_true == preds).mean()
		ind = preds != self.n_classes
		pred_acc = (y_true[ind] == preds[ind]).mean()
		filtered = 1 - np.mean(ind)
		print('Test Acc Raw {:.4f} Eff {:.4f} Rej {:.4f}'.format(pretest_acc, pred_acc, filtered))

		if name is not None:
			cm = normalize(confusion_matrix(y_true,
							preds,
							labels=np.arange(0, max(np.max(y_true)+1, np.max(preds)+1, self.n_classes+1))
							),
							normalize='true')
			cm = np.delete(cm, (self.n_classes), axis=0)
			if cm.shape[1] > (self.n_classes+1):
				cm = np.delete(cm, (self.n_classes+1), axis=1)
			# aps = np.zeros((len(cm), 1))
			# aps[:self.n_classes] = np.array(compute_ap(y_true, y_pred)).reshape(-1, 1)
			# cm = np.concatenate([cm, aps], axis=1)

			class_labels = list(self.class2num.keys()) +['Novel']
			cm_ob = ConfusionMatrixPlot(cm, class_labels)
			factor = max(1, len(cm) // 10)
			fig = plt.figure(figsize=(10*factor,8*factor))
			cm_ob.plot(values_format='0.2f', ax=fig.gca())

			plt.title('Accuracy Raw {:.3f} Eff {:.3f} Rej {:.3f}'.format(pretest_acc, pred_acc, filtered), fontsize=16)
			plt.tight_layout()
			plt.savefig('{}/{}'.format(self.path, name))


	def get_thresholds(self, outlier_frac):

		thresholds = 0.9*np.ones((self.n_classes))
		probs_train = self.val_stats['pred']
		y_train = self.val_stats['true']
		for top_klass in range(self.n_classes):
			ind = (np.argmax(probs_train, axis=1) == top_klass) #& (y_train == top_klass)

			if np.sum(ind) != 0:
				best_prob = np.max(probs_train[ind], axis=1)
				best_prob = np.sort(best_prob)
				l = int(outlier_frac * len(best_prob)) + 1
				# print(len(best_prob))
				if l < (len(best_prob)): 
					thresholds[top_klass] = best_prob[l]

		return thresholds


	def filter_pred(self, pred, outlier_frac):
		thresholds = self.get_thresholds(outlier_frac)

		pred_class = np.argmax(pred, axis=1)
		prob_max = np.max(pred, axis=1)

		ind = prob_max < thresholds[pred_class]
		pred_class[ind] = self.n_classes # assign unassigned class
		return pred_class

	def get_TSNE(self, features):
		pca = PCA(n_components=50)
		reduced_feats = pca.fit_transform(features)
		embeddings = TSNE(n_components=2, verbose=1, n_jobs=-1, perplexity=50, random_state=43).fit_transform(reduced_feats)
		return embeddings

	def get_correl_score(self, features, labels):
		class_types = np.sort(np.unique(labels))
		num_features = features.shape[1]

		means = []
		for label in class_types:
			cluster_mean = np.mean(features[labels == label].reshape(-1, num_features), axis=0, keepdims=True)
			means.append(cluster_mean)

		means = np.concatenate(means, axis=0)

		cof = np.corrcoef(means) - np.eye(len(class_types))

		maxes = np.max(cof, axis=0)

		return np.mean(maxes)

	def get_KNN_score(self, features, labels):
		neigh = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)

		# pca = PCA(n_components=10)
		# reduced_feats = pca.fit_transform(features)
		
		fit = umap.UMAP(
			n_components=20,
		)
		
		reduced_feats = fit.fit_transform(features)

		neigh.fit(reduced_feats, labels)

		acc = neigh.score(reduced_feats, labels)
		probs_ = np.mean(neigh.predict_proba(reduced_feats)[np.arange(len(features)), labels])

		return acc, probs_



	def get_complexity(self):
		if self.reduced_features is not None:
			features = self.reduced_features
		else:
			features = self.raw_features

		labels = self.labels

		score_corr = self.get_correl_score(features, labels)

		embedding = self.get_TSNE(features)

		score_tsne_sil = silhouette_score(embedding, labels)

		# score_raw_sil = silhouette_score(features, labels)

		score_knn_acc, score_knn_prob = self.get_KNN_score(features, labels)

		log = "Correl Score {:.4f} tSNE_silhoutte {:.4f} KNN Score Hard {:.4f} Soft {:.4f}".format(score_corr, score_tsne_sil, score_knn_acc, score_knn_prob)

		print(log)

		df = pd.DataFrame({'tSNE_x': embedding[:, 0],
							'tSNE_y': embedding[:, 1],
							'Labels': list(labels),
							})

		return df, log

		



	def clustercorrect_TSNE(self, test_gene_mat, predictions, labels=None, vis=True):
		# embedding = self.get_TSNE(test_gene_mat.values)
		encoding = self.get_encoding(test_gene_mat, test=True)
		embedding = self.get_TSNE(encoding)
		db = DBSCAN(eps=3., min_samples=2).fit(embedding)

		out = predictions
		predicted_label = out['predictions']
		df = pd.DataFrame({'tSNE_x': embedding[:, 0],
						'tSNE_y': embedding[:, 1],
						"cellname":test_gene_mat.index,
						"predictions": predicted_label,
						"DBSCAN": db.labels_,
						})
		if labels is not None:
			df['labels'] = labels

		ind = df['predictions'] == "Unassigned"
		max_id = df[ind]['DBSCAN'].value_counts().idxmax()
		
		temp = df[ind]['DBSCAN'].values
		un, counts = np.unique(temp, return_counts=True)

		temp_ = np.sort(df['DBSCAN'].values)
		un_, counts_ = np.unique(temp_, return_counts=True)
		if -1 in un_:
			counts_match = counts_[un+1]
		else:
			counts_match = counts_[un]

		args = np.argsort(-counts)

		sorted_counts = counts[args]
		rep_list = un[args]
		sorted_counts_ = counts_match[args]
		sorted_frac = sorted_counts/sorted_counts_

		df['pred_correct'] = df['predictions'].copy()
		print(sorted_frac)
		if sorted_frac[0] > 0.:
			ind = df['DBSCAN'] == max_id
			df.loc[ind, 'pred_correct'] = "Unassigned"

		if vis:
			plt.figure()
			order = list(set(df['predictions']))
			order = sorted(order, key=str.casefold)

			g = sns.scatterplot(x='tSNE_x', y='tSNE_y', hue="predictions", data=df, hue_order=order)
			plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
			plt.tight_layout()
			plt.savefig("{}/TSNE_pred.pdf".format(self.path))

			if labels is not None:
				plt.figure()
				order = list(set(df['labels']))
				order = sorted(order, key=str.casefold)

				g = sns.scatterplot(x='tSNE_x', y='tSNE_y', hue="labels", data=df, hue_order=order)
				plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
				plt.tight_layout()
				plt.savefig("{}/TSNE_true.pdf".format(self.path))

			plt.figure()

			g = sns.scatterplot(x='tSNE_x', y='tSNE_y', hue="DBSCAN", data=df, legend="full", palette="viridis")
			# plt.scatter(embedding[:, 0], embedding[:, 1], c=clustering.labels_)
			plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
			plt.tight_layout()
			plt.savefig("{}/TSNE_DBSCAN.pdf".format(self.path))

			plt.figure()
			order = list(set(df['pred_correct']))
			order = sorted(order, key=str.casefold)

			g = sns.scatterplot(x='tSNE_x', y='tSNE_y', hue="pred_correct", data=df, hue_order=order)
			# plt.scatter(embedding[:, 0], embedding[:, 1], c=clustering.labels_)
			plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
			plt.tight_layout()
			plt.savefig("{}/TSNE_corrected.pdf".format(self.path))

		if labels is not None:
			ind = df['predictions'] == df['labels']
			data_filt = df[~df['predictions'].isin(["Unassigned"])]
			ind_filt = data_filt['predictions'] == data_filt['labels']
			ind_filt = ind_filt.values
			print("Before correction Accuracy Post {:.4f} Eff {:.4f}".format(np.mean(ind), np.mean(ind_filt)))

			ind = df['pred_correct'] == df['labels']
			data_filt = df[~df['pred_correct'].isin(["Unassigned"])]
			ind_filt = data_filt['pred_correct'] == data_filt['labels']
			ind_filt = ind_filt.values
			print("After Correction Accuracy Post {:.4f} Eff {:.4f}".format(np.mean(ind), np.mean(ind_filt)))

			self.generate_cfmt(df['pred_correct'], df['labels'], name="testcfmtcorrected.pdf")

		return df

	def detect_outlier(self, train_gene_mat, train_labels, test_gene_mat, predictions, test_labels=None):
		# encoding1 = self.get_encoding(train_gene_mat)

		# encoding2 = self.get_encoding(test_gene_mat, test=False)
		encoding1 = train_gene_mat.values
		encoding2 = test_gene_mat.values
		
		embedding = self.get_TSNE(np.concatenate([encoding1, encoding2], axis=0))

		embedding1 = embedding[:len(encoding1)]
		embedding2 = embedding[len(encoding1):]

		db = DBSCAN(eps=5., min_samples=15).fit(embedding2)

		clf = IsolationForest(n_estimators=100, contamination=0.02).fit(embedding1)
		detections = clf.predict(embedding2)

		out = predictions
		predicted_label = out['predictions']
		df = pd.DataFrame({'tSNE_x': embedding[:, 0],
							'tSNE_y': embedding[:, 1],
							'DBSCAN': list(train_labels) + [str(a) for a in list(db.labels_)],
							'Batch': ['Source']*(len(encoding1)) + ['Target']*len(encoding2),
							'Detector': [1]*(len(encoding1)) + list(detections),
							'predictions': list(train_labels) + list(predicted_label)
							})

		if test_labels is not None:
			df = pd.DataFrame({'tSNE_x': embedding[:, 0],
							'tSNE_y': embedding[:, 1],
							'Labels': list(train_labels) + list(test_labels),
							'DBSCAN': list(train_labels) + [str(a) for a in list(db.labels_)],
							'Detector': [1]*(len(encoding1)) + list(detections),
							'Batch': ['Source']*(len(encoding1)) + ['Target']*len(encoding2),
							'predictions': list(train_labels) + list(predicted_label)
							})
			plt.figure(figsize=(6, 6))		
			order = list(set(df['Labels']))
			order = sorted(order, key=str.casefold)

			g = sns.scatterplot(x='tSNE_x', y='tSNE_y', hue='Labels', data=df, hue_order=order, style='Batch', style_order=["Source", "Target"]) #, size='|Match|', size_order=['miss', 'correct'])
			plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
			plt.title("Labels")

			plt.tight_layout()

			plt.savefig("{}/Comparison_TSNE.pdf".format(self.path))

		df = pd.DataFrame({'tSNE_x': embedding[:, 0],
							'tSNE_y': embedding[:, 1],
							# 'Labels': list(train_labels) + list(test_labels),
							'DBSCAN': list(train_labels) + [str(a) for a in list(db.labels_)],
							'Detector': [1]*(len(encoding1)) + list(detections),
							'Batch': ['Source']*(len(encoding1)) + ['Target']*len(encoding2),
							'predictions': list(train_labels) + list(predicted_label)
							})
		if test_labels is not None:
			df = pd.DataFrame({'tSNE_x': embedding[:, 0],
							'tSNE_y': embedding[:, 1],
							'Labels': list(train_labels) + list(test_labels),
							'DBSCAN': list(train_labels) + [str(a) for a in list(db.labels_)],
							'Detector': [1]*(len(encoding1)) + list(detections),
							'Batch': ['Source']*(len(encoding1)) + ['Target']*len(encoding2),
							'predictions': list(train_labels) + list(predicted_label)
							})
		
		df['Novel'] = False
		df['pred_correct'] = df['predictions']

		# selected = df[((df['batch']=='Target')) & (df['DBSCAN'].isin(list(novel_clusters)))][]
		df.loc[((df['Batch']=='Target')) & (df['Detector'].isin([-1])), 'Novel'] = True
		df.loc[((df['Batch']=='Target')) & (df['Detector'].isin([-1])), 'pred_correct'] = "Unassigned"
		print(np.sum(df['Novel']==True), "detected as Novel cells")


		if test_labels is not None:
			plt.figure(figsize=(6, 6))		
			order = list(set(df['Labels']))
			order = sorted(order, key=str.casefold)

			g = sns.scatterplot(x='tSNE_x', y='tSNE_y', hue='Labels', data=df, hue_order=order, style='Batch', style_order=["Source", "Target"])#, size='|Match|', size_order=['miss', 'correct'])
			plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
			plt.title("Labels")

			plt.tight_layout()

			plt.savefig("{}/Comparison_TSNE_postremoval.pdf".format(self.path))

		plt.figure(figsize=(6, 6))		
		order = list(set(df['Detector']))
		order = sorted(order)

		g = sns.scatterplot(x='tSNE_x', y='tSNE_y', hue='Detector', data=df, hue_order=order, style='Batch', style_order=["Source", "Target"], legend='full')#, size='|Match|', size_order=['miss', 'correct'])
		plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
		plt.title("Labels")

		plt.tight_layout()
		plt.savefig("{}/Comparison_TSNE_Detector.pdf".format(self.path))

		dftest = df.loc[(df['Batch'] == 'Target')]
		if test_labels is not None:
			ind = dftest['predictions'] == dftest['Labels']
			data_filt = dftest[~dftest['predictions'].isin(["Unassigned"])]
			ind_filt = data_filt['predictions'] == data_filt['Labels']
			ind_filt = ind_filt.values
			print("Before correction Accuracy Post {:.4f} Eff {:.4f}".format(np.mean(ind), np.mean(ind_filt)))

			ind = dftest['pred_correct'] == dftest['Labels']
			data_filt = dftest[~dftest['pred_correct'].isin(["Unassigned"])]
			ind_filt = data_filt['pred_correct'] == data_filt['Labels']
			ind_filt = ind_filt.values
			print("After Correction Accuracy Post {:.4f} Eff {:.4f}".format(np.mean(ind), np.mean(ind_filt)))

			self.generate_cfmt(dftest['pred_correct'], dftest['Labels'], name="testcfmtcorrected.pdf")

		frame = df.loc[(df['Batch'] == 'Target')]
		frame.index = test_gene_mat.index
		return frame

	def detect_novel(self, train_gene_mat, train_labels, test_gene_mat, predictions, test_labels=None, test=False):
		encoding1 = self.get_encoding(train_gene_mat)

		encoding2 = self.get_encoding(test_gene_mat, test=False)
		# encoding1 = train_gene_mat.values
		# encoding2 = test_gene_mat.values

		embedding = self.get_TSNE(np.concatenate([encoding1, encoding2], axis=0))

		embedding1 = embedding[:len(encoding1)]
		embedding2 = embedding[len(encoding1):]

		db = DBSCAN(eps=5., min_samples=15).fit(embedding2)

		out = predictions
		predicted_label = out['predictions']
		df = pd.DataFrame({'tSNE_x': embedding[:, 0],
							'tSNE_y': embedding[:, 1],
							'Batch': ['Source']*(len(encoding1)) + ['Target']*len(encoding2),
							'predictions': list(train_labels) + list(predicted_label)
							})

		if test_labels is not None:
			df = pd.DataFrame({'tSNE_x': embedding[:, 0],
							'tSNE_y': embedding[:, 1],
							'Labels': list(train_labels) + list(test_labels),
							'DBSCAN': list(train_labels) + [str(a) for a in list(db.labels_)],
							'Batch': ['Source']*(len(encoding1)) + ['Target']*len(encoding2),
							'predictions': list(train_labels) + list(predicted_label)
							})
			plt.figure(figsize=(6, 6))		
			order = list(set(df['Labels']))
			order = sorted(order, key=str.casefold)

			g = sns.scatterplot(x='tSNE_x', y='tSNE_y', hue='Labels', data=df, hue_order=order, style='Batch', style_order=["Source", "Target"]) #, size='|Match|', size_order=['miss', 'correct'])
			plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
			plt.title("Labels")

			plt.tight_layout()

			plt.savefig("{}/Comparison_TSNE_labelled.pdf".format(self.path))

		plt.figure(figsize=(4, 4))		
		order = list(set(df['Batch']))
		order = sorted(order, key=str.casefold)

		g = sns.scatterplot(x='tSNE_x', y='tSNE_y', hue='Batch', data=df, hue_order=order) #, size='|Match|', size_order=['miss', 'correct'])
		plt.legend()
		plt.title("Batch Plot")

		plt.tight_layout()

		plt.savefig("{}/Comparison_TSNE.pdf".format(self.path))

		if test:
			encoding2 = self.get_encoding(test_gene_mat, test=test)
			embedding = self.get_TSNE(np.concatenate([encoding1, encoding2], axis=0))

		embedding1 = embedding[:len(encoding1)]
		embedding2 = embedding[len(encoding1):]
		
		stacked_encoding = np.concatenate([encoding1, encoding2], axis=0)

		df = pd.DataFrame({'tSNE_x': embedding[:, 0],
							'tSNE_y': embedding[:, 1],
							# 'Labels': list(train_labels) + list(test_labels),
							'DBSCAN': list(train_labels) + [str(a) for a in list(db.labels_)],
							'Batch': ['Source']*(len(encoding1)) + ['Target']*len(encoding2),
							'predictions': list(train_labels) + list(predicted_label)
							})
		if test_labels is not None:
			df = pd.DataFrame({'tSNE_x': embedding[:, 0],
							'tSNE_y': embedding[:, 1],
							'Labels': list(train_labels) + list(test_labels),
							'DBSCAN': list(train_labels) + [str(a) for a in list(db.labels_)],
							'Batch': ['Source']*(len(encoding1)) + ['Target']*len(encoding2),
							'predictions': list(train_labels) + list(predicted_label)
							})

		test_clusters = [str(a) for a in list(db.labels_)]
		test_clusters.sort()

		sourcec_means = {}
		source = df[df['Batch'] == "Source"]
		stacked_encoding_source = stacked_encoding[df['Batch'] == "Source"]
		for label in list(set(train_labels)):
			k_source = source[source['DBSCAN'] == label]
			# k_mean = np.array([np.mean(k_source['tSNE_x']), np.mean(k_source['tSNE_y'])])
			k_mean = np.mean(stacked_encoding_source[source['DBSCAN'] == label], axis=0)
			sourcec_means[label] = k_mean

		targetc_means = {}
		target_mean_list = []
		target = df[df['Batch'] == "Target"]
		stacked_encoding_target = stacked_encoding[df['Batch'] == "Target"]
		test_clusters = list(set(test_clusters) - {'-1'})
		test_clusters.sort()
		for label in test_clusters:
			# Remove noise cluster labelled as -1
			if label != '-1':
				k_target = target[target['DBSCAN'] == label]
				# k_mean = np.array([np.mean(k_target['tSNE_x']), np.mean(k_target['tSNE_y'])])
				k_mean = np.mean(stacked_encoding_target[target['DBSCAN'] == label], axis=0)
				target_mean_list.append(k_mean)
				targetc_means[label] = k_mean

		target_means = np.stack(target_mean_list, axis=0)
		# source2target = {}
		train_clusters = list(set(train_labels))
		train_clusters.sort()
		dist_frame = {}
		for label in train_clusters:
			k_mean = sourcec_means[label]
			dist = np.mean((target_means - k_mean)**2, axis=1)
			dist_frame[label] = dist
			# ind = np.argmin(dist)
			# print("{}->{}".format(label, ind).rjust(20), dist)
			# source2target[label] = str(ind)

		# print(source2target)

		dist_matrix = pd.DataFrame(dist_frame, index=test_clusters)
		print(dist_matrix)

		mat = dist_matrix.values
		rows, cols = optimize.linear_sum_assignment(mat)
		mapped_clusters = list(dist_matrix.index[rows])
		
		print(mapped_clusters, "->", list(dist_matrix.columns[cols]))
		novel_clusters = set(test_clusters) - {'-1'} - set(mapped_clusters).intersection(set(test_clusters))
		
		print(novel_clusters)
		
		df['Novel'] = False
		df['pred_correct'] = df['predictions']

		# selected = df[((df['batch']=='Target')) & (df['DBSCAN'].isin(list(novel_clusters)))][]
		df.loc[((df['Batch']=='Target')) & (df['DBSCAN'].isin(novel_clusters)), 'Novel'] = True
		df.loc[((df['Batch']=='Target')) & (df['DBSCAN'].isin(novel_clusters)), 'pred_correct'] = "Unassigned"
		print(np.sum(df['Novel']==True), "detected as Novel cells")


		if test_labels is not None:
			plt.figure(figsize=(6, 6))		
			order = list(set(df['Labels']))
			order = sorted(order, key=str.casefold)

			g = sns.scatterplot(x='tSNE_x', y='tSNE_y', hue='Labels', data=df, hue_order=order, style='Batch', style_order=["Source", "Target"])#, size='|Match|', size_order=['miss', 'correct'])
			plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
			plt.title("Labels")

			plt.tight_layout()

			plt.savefig("{}/Comparison_TSNE_postremoval_labelled.pdf".format(self.path))

		plt.figure(figsize=(4, 4))		
		order = list(set(df['Batch']))
		order = sorted(order, key=str.casefold)

		g = sns.scatterplot(x='tSNE_x', y='tSNE_y', hue='Batch', data=df, hue_order=order)#, size='|Match|', size_order=['miss', 'correct'])
		plt.legend()
		plt.title("Batch Plot")

		plt.tight_layout()

		plt.savefig("{}/Comparison_TSNE_postremoval.pdf".format(self.path))

		plt.figure(figsize=(6, 6))		
		order = list(set(df['DBSCAN']))
		order = sorted(order, key=str.casefold)

		g = sns.scatterplot(x='tSNE_x', y='tSNE_y', hue='DBSCAN', data=df, hue_order=order, style='Batch', style_order=["Source", "Target"])#, size='|Match|', size_order=['miss', 'correct'])
		plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
		plt.title("Labels")

		plt.tight_layout()
		plt.savefig("{}/Comparison_TSNE_DBSCAN.pdf".format(self.path))

		dftest = df.loc[(df['Batch'] == 'Target')]
		if test_labels is not None:
			ind = dftest['predictions'] == dftest['Labels']
			data_filt = dftest[~dftest['predictions'].isin(["Unassigned"])]
			ind_filt = data_filt['predictions'] == data_filt['Labels']
			ind_filt = ind_filt.values
			print("Before correction Accuracy Post {:.4f} Eff {:.4f}".format(np.mean(ind), np.mean(ind_filt)))

			ind = dftest['pred_correct'] == dftest['Labels']
			data_filt = dftest[~dftest['pred_correct'].isin(["Unassigned"])]
			ind_filt = data_filt['pred_correct'] == data_filt['Labels']
			ind_filt = ind_filt.values
			print("After Correction Accuracy Post {:.4f} Eff {:.4f}".format(np.mean(ind), np.mean(ind_filt)))

			self.generate_cfmt(dftest['pred_correct'], dftest['Labels'], name="testcfmtcorrected.pdf")

		frame = df.loc[(df['Batch'] == 'Target')]
		frame.index = test_gene_mat.index
		return frame


	def vis_latent(self, train_gene_mat, train_labels, test_gene_mat, test_labels, test=False):
		encoding1 = self.get_encoding(train_gene_mat)

		encoding2 = self.get_encoding(test_gene_mat, test=test)
		# encoding1 = train_gene_mat.values
		# encoding2 = test_gene_mat.values

		embedding = self.get_TSNE(np.concatenate([encoding1, encoding2], axis=0))

		embedding1 = embedding[:len(encoding1)]
		embedding2 = embedding[len(encoding1):]

		df = pd.DataFrame({'tSNE_x': embedding[:, 0],
						'tSNE_y': embedding[:, 1],
						'Labels': list(train_labels) + list(test_labels),
						'Batch': ['Source']*(len(encoding1)) + ['Target']*len(encoding2),
						})

		plt.figure(figsize=(8, 6))		
		order = list(set(df['Labels']))
		order = list(df[df['Batch'] == "Target"]['Labels'].value_counts().index)

		g = sns.scatterplot(x='tSNE_x', y='tSNE_y', hue='Labels', data=df, hue_order=order, style='Batch', style_order=["Source", "Target"], s=80) #, size='|Match|', size_order=['miss', 'correct'])
		plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=15, markerscale=2.)
		plt.title("Labels")

		plt.tight_layout()

		plt.savefig("{}/Comparison_TSNE_labelled_{}.pdf".format(self.path, test))

		plt.figure(figsize=(6, 6))		
		order = list(set(df['Batch']))
		order = sorted(order, key=str.casefold)

		g = sns.scatterplot(x='tSNE_x', y='tSNE_y', hue='Batch', data=df, hue_order=order, s=80) #, size='|Match|', size_order=['miss', 'correct'])
		plt.legend(fontsize=15, markerscale=2.)
		plt.title("Batch Plot")

		plt.tight_layout()

		plt.savefig("{}/Comparison_TSNE_{}.pdf".format(self.path, test))
		return


	def remove_effect(self, train_gene_mat, test_gene_mat, config, test_labels=None):
		
		features_batch1 = self.get_features(train_gene_mat)
		features_batch2 = self.get_features(test_gene_mat)
		
		torch.manual_seed(config['seed'])
		torch.cuda.manual_seed(config['seed'])
		np.random.seed(config['seed'])
		torch.backends.cudnn.deterministic = True

		batch1_dataset = DataLoaderCustom(features_batch1)
		batch2_dataset = DataLoaderCustom(features_batch2)

		use_cuda = config['cuda']
		use_cuda = use_cuda and torch.cuda.is_available()

		device = torch.device("cuda" if use_cuda else "cpu")
		kwargs = {'num_workers': 4, 'pin_memory': False} if use_cuda else {}

		batch1_loader = torch.utils.data.DataLoader(batch1_dataset,
										   batch_size=config['batch_size'],
										   shuffle=True, **kwargs)

		batch2_loader = torch.utils.data.DataLoader(batch2_dataset,
										   batch_size=config['batch_size'],
										   shuffle=False, **kwargs)


		model1 = self.model.to(device)
		for param in model1.parameters():
			param.requires_grad = False
		# Define new model
		model_copy = Classifier(features_batch1.shape[1], LDIM, MODEL_WIDTH, self.n_classes).to(device)
		# Intialize it with the same parameter values as trained model
		model_copy.load_state_dict(model1.state_dict())
		for param in model_copy.parameters():
			param.requires_grad = False
		model2 = ClassifierBig(model_copy,features_batch1.shape[1], LDIM, GLDIM).to(device)

		disc = Discriminator(LDIM).to(device)

		G_decay = config.get("gdecay", 1e-2)
		D_decay = config.get("ddecay", 1e-6)
		max_count = config.get("maxcount", 3)

		# optimizer_G = torch.optim.Adam(model2.parameters(), lr=3e-4, betas=(0.5, 0.999))
		# optimizer_D = torch.optim.Adam(disc.parameters(), lr=1e-4, betas=(0.5, 0.999))
		optimizer_G = torch.optim.RMSprop(model2.parameters(), lr=1e-4, weight_decay=G_decay)
		optimizer_D = torch.optim.RMSprop(disc.parameters(), lr=1e-4, weight_decay=D_decay)
		adversarial_weight = torch.nn.BCELoss(reduction='none')
		adversarial_loss = torch.nn.BCELoss()
		sample_loss = torch.nn.BCELoss()


		Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

		bs = min(config['batch_size'], len(features_batch2), len(features_batch1))
		count = 0
		dry_epochs = 0
		best_rej_frac = 1.0
		for epoch in range(config['epochs']):
			if len(batch2_loader) < 50:
				pBar = tqdm(range(40))
			else:
				pBar = tqdm(batch2_loader)
			model1.eval()
			model2.eval()
			disc.train()
			c1, s1 = 0, 0.7
			c2, s2 = 0, 0.7
			for sample in pBar:

				valid = Variable(Tensor(bs, 1).fill_(1.0), requires_grad=False)
				fake = Variable(Tensor(bs, 1).fill_(0.0), requires_grad=False)

				sample_loss = torch.nn.BCELoss()
				disc.eval()
				
				for i in range(1):
					ind = np.random.randint(0, (len(features_batch2)), bs)
					batch2_inps = Variable(torch.from_numpy(features_batch2[ind])).to(device).type(Tensor)
					optimizer_G.zero_grad()

					batch2_code, penalty = model2.get_repr(batch2_inps)
					# g_loss = adversarial_weight(disc(batch2_code), valid)
					# print(np.mean(weights.numpy()))
					# weights = torch.exp(g_loss.detach() - 0.8).clamp(0.9, 1.5)
					# sample_loss = torch.nn.BCELoss(weight=weights.detach())
					g_loss = sample_loss(disc(batch2_code), valid) #+ 0.001 * penalty
					# g_loss = -torch.mean(disc(batch2_code))
					if s2 > 0.4:
						g_loss.backward()
						optimizer_G.step()
					s2 = ((s2*c2)+(float(g_loss.item())*len(batch2_code)))/(c2+len(batch2_code))
					c2 += len(batch2_code)

					if s2 == 0 or g_loss.item() == 0:
						model2.reinitialize()
						# reset count as well
						count = 0
						dry_epochs = 0

				
				sample_loss = torch.nn.BCELoss()
				model2.eval()
				disc.train()
				for i in range(2):
					if i != 0:
						ind = np.random.randint(0, (len(features_batch2)), bs)
						batch2_inps = Variable(torch.from_numpy(features_batch2[ind])).to(device).type(Tensor)
						batch2_code, _ = model2.get_repr(batch2_inps)
					optimizer_D.zero_grad()
					ind = np.random.randint(0, (len(features_batch1)), bs)
					batch1_inps = Variable(torch.from_numpy(features_batch1[ind])).to(device).type(Tensor)
					batch1_code = model1.get_repr(batch1_inps)
					
					# real_loss = adversarial_weight(disc(batch1_code), valid[:batch1_code.size()[0]])
					# weights = torch.exp(real_loss.detach() - 0.8).clamp(1., 1.2)
					# sample_loss = torch.nn.BCELoss(weight=weights.detach())
					real_loss = sample_loss(disc(batch1_code), valid[:batch1_code.size()[0]])

					# fake_loss = adversarial_weight(disc(batch2_code.detach()), fake)
					# weights = torch.exp(fake_loss.detach() - 0.8).clamp(1., 1.2)
					# sample_loss = torch.nn.BCELoss(weight=weights.detach())
					fake_loss = sample_loss(disc(batch2_code.detach()), fake)
					# real_loss = -torch.mean(disc(batch1_code))
					# fake_loss = torch.mean(disc(batch2_code.detach()))
					d_loss = 0.5 * (real_loss + fake_loss)

					if s2 < 0.8 or s1 > 1.0:
						d_loss.backward()
						optimizer_D.step()
					# for p in disc.parameters():
					# 	p.data.clamp_(-0.01, 0.01)
					s1 = ((s1*c1)+(float(d_loss.item())*len(batch1_code)))/(c1+len(batch1_code))
					c1 += len(batch1_code)

					if s1 == 0 or d_loss.item() == 0:
						model2.reinitialize()
						# reset count as well
						count = 0
						dry_epochs = 0

				pBar.set_description('Epoch {} G Loss: {:.3f} D Loss: {:.3f}'.format(epoch, s2, s1))
			if (s2 < 0.78) and (s2 > 0.5) and (s1 < 0.78) and (s1 > 0.5):
				count += 1
				self.test_model = model2
				if test_labels is not None:
					print("Evaluating....")
					predictions = self.evaluate(test_gene_mat, test_labels, frac=0.05, name=None, test=True)
				
				predictions = self.get_filtered_prediction(test_gene_mat, frac=0.05, test=True)
				rej_frac = np.mean(predictions["predictions"] == "Unassigned")
				if rej_frac < best_rej_frac:
					print(f"Updated Rejected cells from {best_rej_frac:.3f} to {rej_frac:.3f}")
					best_rej_frac = rej_frac
					torch.save(model2.state_dict(), self.path+"/best_br.pth")
				
				dry_epochs = 0
				if count >= max_count:
					break
			else:
				dry_epochs += 1
				if dry_epochs == 3:
					print("Loss not improving, stopping alignment")
					break

		if not os.path.isfile(self.path+"/best_br.pth"):
			print("Warning: Alignment did not succeed properly, try changing the gdecay or ddecay!")
			torch.save(model2.state_dict(), self.path+"/best_br.pth")
			
		model2.load_state_dict(torch.load(self.path+"/best_br.pth"))
		self.test_model = model2


	def remove_effectv2(self, train_gene_mat, test_gene_mat, config, test_labels=None):
		
		features_batch1 = self.get_features(train_gene_mat)
		features_batch2 = self.get_features(test_gene_mat)
		
		torch.manual_seed(config['seed'])
		torch.cuda.manual_seed(config['seed'])
		np.random.seed(config['seed'])
		torch.backends.cudnn.deterministic = True

		batch1_dataset = DataLoaderCustom(features_batch1)
		batch2_dataset = DataLoaderCustom(features_batch2)

		use_cuda = config['cuda']
		use_cuda = use_cuda and torch.cuda.is_available()

		device = torch.device("cuda" if use_cuda else "cpu")
		kwargs = {'num_workers': 4, 'pin_memory': False} if use_cuda else {}

		batch1_loader = torch.utils.data.DataLoader(batch1_dataset,
										   batch_size=config['batch_size'],
										   shuffle=True, **kwargs)

		batch2_loader = torch.utils.data.DataLoader(batch2_dataset,
										   batch_size=config['batch_size'],
										   shuffle=False, **kwargs)


		model1 = self.model.to(device)
		for param in model1.parameters():
			param.requires_grad = False
		# Define new model
		model_copy = Classifier(features_batch1.shape[1], LDIM, MODEL_WIDTH, self.n_classes).to(device)
		# Intialize it with the same parameter values as trained model
		model_copy.load_state_dict(self.model.state_dict())
		
		for param in model_copy.fc.parameters():
			param.requires_grad = False

		for param in model_copy.fc1.parameters():
			param.requires_grad = False


		model2 = ClassifierBig(model_copy,features_batch1.shape[1], LDIM, GLDIM).to(device)

		disc = Discriminator(LDIM).to(device)

		G_decay = config.get("gdecay", 1e-2)
		D_decay = config.get("ddecay", 1e-6)
		max_count = config.get("maxcount", 3)

		# optimizer_G = torch.optim.Adam(model2.parameters(), lr=3e-4, betas=(0.5, 0.999))
		# optimizer_D = torch.optim.Adam(disc.parameters(), lr=1e-4, betas=(0.5, 0.999))
		optimizer_G = torch.optim.RMSprop(model2.parameters(), lr=1e-4, weight_decay=G_decay)
		optimizer_D = torch.optim.RMSprop(disc.parameters(), lr=1e-4, weight_decay=D_decay)
		adversarial_weight = torch.nn.BCELoss(reduction='none')
		adversarial_loss = torch.nn.BCELoss()
		sample_loss = torch.nn.BCELoss()


		Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

		bs = min(config['batch_size'], len(features_batch2), len(features_batch1))
		count = 0
		best_rej_frac = 1.0
		for epoch in range(config['epochs']):
			if len(batch2_loader) < 50:
				pBar = tqdm(range(40))
			else:
				pBar = tqdm(batch2_loader)
			model1.eval()
			model2.eval()
			disc.train()
			c1, s1 = 0, 0.7
			c2, s2 = 0, 0.7
			for sample in pBar:

				valid = Variable(Tensor(bs, 1).fill_(1.0), requires_grad=False)
				fake = Variable(Tensor(bs, 1).fill_(0.0), requires_grad=False)

				sample_loss = torch.nn.BCELoss()
				disc.eval()
				
				for i in range(1):
					ind = np.random.randint(0, (len(features_batch2)), bs)
					batch2_inps = Variable(torch.from_numpy(features_batch2[ind])).to(device).type(Tensor)
					optimizer_G.zero_grad()

					batch2_code, penalty = model2.get_repr(batch2_inps)
					# g_loss = adversarial_weight(disc(batch2_code), valid)
					# print(np.mean(weights.numpy()))
					# weights = torch.exp(g_loss.detach() - 0.8).clamp(0.9, 1.5)
					# sample_loss = torch.nn.BCELoss(weight=weights.detach())
					g_loss = sample_loss(disc(batch2_code), valid) #+ 0.001 * penalty
					# g_loss = -torch.mean(disc(batch2_code))
					if s2 > 0.4:
						g_loss.backward()
						optimizer_G.step()
					s2 = ((s2*c2)+(float(g_loss.item())*len(batch2_code)))/(c2+len(batch2_code))
					c2 += len(batch2_code)

					if s2 == 0 or g_loss.item() == 0:
						model2.reinitialize()
						# reset count as well
						count = 0

				
				sample_loss = torch.nn.BCELoss()
				model2.eval()
				disc.train()
				for i in range(2):
					if i != 0:
						ind = np.random.randint(0, (len(features_batch2)), bs)
						batch2_inps = Variable(torch.from_numpy(features_batch2[ind])).to(device).type(Tensor)
						batch2_code, _ = model2.get_repr(batch2_inps)
					optimizer_D.zero_grad()
					ind = np.random.randint(0, (len(features_batch1)), bs)
					batch1_inps = Variable(torch.from_numpy(features_batch1[ind])).to(device).type(Tensor)
					batch1_code = model1.get_repr(batch1_inps)
					
					# real_loss = adversarial_weight(disc(batch1_code), valid[:batch1_code.size()[0]])
					# weights = torch.exp(real_loss.detach() - 0.8).clamp(1., 1.2)
					# sample_loss = torch.nn.BCELoss(weight=weights.detach())
					real_loss = sample_loss(disc(batch1_code), valid[:batch1_code.size()[0]])

					# fake_loss = adversarial_weight(disc(batch2_code.detach()), fake)
					# weights = torch.exp(fake_loss.detach() - 0.8).clamp(1., 1.2)
					# sample_loss = torch.nn.BCELoss(weight=weights.detach())
					fake_loss = sample_loss(disc(batch2_code.detach()), fake)
					# real_loss = -torch.mean(disc(batch1_code))
					# fake_loss = torch.mean(disc(batch2_code.detach()))
					d_loss = 0.5 * (real_loss + fake_loss)

					if s2 < 0.8 or s1 > 1.0:
						d_loss.backward()
						optimizer_D.step()
					# for p in disc.parameters():
					# 	p.data.clamp_(-0.01, 0.01)
					s1 = ((s1*c1)+(float(d_loss.item())*len(batch1_code)))/(c1+len(batch1_code))
					c1 += len(batch1_code)

					if s1 == 0 or d_loss.item() == 0:
						model2.reinitialize()
						# reset count as well
						count = 0

				pBar.set_description('Epoch {} G Loss: {:.3f} D Loss: {:.3f}'.format(epoch, s2, s1))
			if (s2 < 0.78) and (s2 > 0.5) and (s1 < 0.78) and (s1 > 0.5):
				count += 1
				self.test_model = model2
				if test_labels is not None:
					print("Evaluating....")
					predictions = self.evaluate(test_gene_mat, test_labels, frac=0.05, name=None, test=True)
				
				predictions = self.get_filtered_prediction(test_gene_mat, frac=0.05, test=True)
				rej_frac = np.mean(predictions["predictions"] == "Unassigned")
				if rej_frac < best_rej_frac:
					print(f"Updated Rejected cells from {best_rej_frac:.3f} to {rej_frac:.3f}")
					best_rej_frac = rej_frac
					torch.save(model2.state_dict(), self.path+"/best_br.pth")
				

				if count >= max_count:
					break

		if not os.path.isfile(self.path+"/best_br.pth"):
			print("Warning: Alignment did not succeed properly, try changing the gdecay or ddecay!")
			torch.save(model2.state_dict(), self.path+"/best_br.pth")
			
		model2.load_state_dict(torch.load(self.path+"/best_br.pth"))
		self.test_model = model2


	def weighted_remove_effect(self, train_gene_mat, test_gene_mat, config, test_labels=None):
		
		features_batch1 = self.get_features(train_gene_mat)
		features_batch2 = self.get_features(test_gene_mat)
		
		torch.manual_seed(config['seed'])
		torch.cuda.manual_seed(config['seed'])
		np.random.seed(config['seed'])
		torch.backends.cudnn.deterministic = True


		y_pred = self.predict(test_gene_mat)
		preds = np.argmax(y_pred, axis=1)

		test_prop = np.array([np.sum(preds==i) for i in range(self.n_classes)])
		test_prop = test_prop/np.sum(test_prop)

		train_labels = self.labels
		train_prop = np.array([np.sum(train_labels==i) for i in range(self.n_classes)])
		train_prop = train_prop/np.sum(train_prop)

		factor_update = test_prop/train_prop
		factor_update[factor_update > 0.1] = 1.
		factor_update = factor_update * len(factor_update) / (np.sum(factor_update) + 1e-4)
		print(factor_update, train_prop, test_prop)


		batch1_dataset = DataLoaderCustom(features_batch1, labels=self.labels, weights=factor_update)
		batch2_dataset = DataLoaderCustom(features_batch2)

		# sys.exit()

		use_cuda = config['cuda']
		use_cuda = use_cuda and torch.cuda.is_available()

		device = torch.device("cuda" if use_cuda else "cpu")
		kwargs = {'num_workers': 4, 'pin_memory': False} if use_cuda else {}

		batch1_loader = torch.utils.data.DataLoader(batch1_dataset,
										   batch_size=config['batch_size'],
										   shuffle=True, **kwargs)

		batch2_loader = torch.utils.data.DataLoader(batch2_dataset,
										   batch_size=config['batch_size'],
										   shuffle=False, **kwargs)


		model1 = self.model.to(device)
		for param in model1.parameters():
			param.requires_grad = False
		# Define new model
		model_copy = Classifier(features_batch1.shape[1], LDIM, MODEL_WIDTH, self.n_classes).to(device)
		# Intialize it with the same parameter values as trained model
		model_copy.load_state_dict(model1.state_dict())
		for param in model_copy.parameters():
			param.requires_grad = False
		model2 = ClassifierBig(model_copy,features_batch1.shape[1], LDIM, GLDIM).to(device)

		disc = Discriminator(LDIM).to(device)

		# optimizer_G = torch.optim.Adam(model2.parameters(), lr=3e-4, betas=(0.5, 0.999))
		# optimizer_D = torch.optim.Adam(disc.parameters(), lr=1e-4, betas=(0.5, 0.999))
		optimizer_G = torch.optim.RMSprop(model2.parameters(), lr=1e-4, weight_decay=1e-2)
		optimizer_D = torch.optim.RMSprop(disc.parameters(), lr=1e-4, weight_decay=1e-6)
		adversarial_weight = torch.nn.BCELoss(reduction='none')
		adversarial_loss = torch.nn.BCELoss()
		sample_loss = torch.nn.BCELoss()


		Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

		bs = min(config['batch_size'], len(features_batch2), len(features_batch1))
		count = 0
		for epoch in range(config['epochs']):
			if len(batch2_loader) < 50:
				pBar = tqdm(range(40))
			else:
				pBar = tqdm(batch2_loader)
			model1.eval()
			model2.eval()
			disc.train()
			c1, s1 = 0, 0
			c2, s2 = 0, 0
			for sample in pBar:

				valid = Variable(Tensor(bs, 1).fill_(1.0), requires_grad=False)
				fake = Variable(Tensor(bs, 1).fill_(0.0), requires_grad=False)

				sample_loss = torch.nn.BCELoss()
				disc.eval()
				for i in range(1):
					ind = np.random.randint(0, (len(features_batch2)), bs)
					batch2_inps = Variable(torch.from_numpy(features_batch2[ind])).to(device).type(Tensor)
					optimizer_G.zero_grad()

					batch2_code, penalty = model2.get_repr(batch2_inps)
					g_loss = adversarial_weight(disc(batch2_code), valid)
					# print(np.mean(weights.numpy()))
					# weights = torch.exp(g_loss.detach() - 0.8).clamp(0.9, 1.5)
					# sample_loss = torch.nn.BCELoss(weight=weights.detach())
					g_loss = sample_loss(disc(batch2_code), valid) #+ 0.001 * penalty
					# g_loss = -torch.mean(disc(batch2_code))
					g_loss.backward()
					optimizer_G.step()
					s2 = ((s2*c2)+(float(g_loss.item())*len(batch2_code)))/(c2+len(batch2_code))
					c2 += len(batch2_code)

				if s2 < 0.8:
					sample_loss = torch.nn.BCELoss()
					model2.eval()
					disc.train()
					for i in range(2):
						if i != 0:
							ind = np.random.randint(0, (len(features_batch2)), bs)
							batch2_inps = Variable(torch.from_numpy(features_batch2[ind])).to(device).type(Tensor)
							batch2_code, _ = model2.get_repr(batch2_inps)
						optimizer_D.zero_grad()
						ind = np.random.randint(0, (len(features_batch1)), bs)
						batch1_inps = Variable(torch.from_numpy(features_batch1[ind])).to(device).type(Tensor)
						batch1_weights = Variable(torch.from_numpy(factor_update[train_labels[ind]])).to(device).type(Tensor).reshape(-1, 1)
						# print(factor_update[train_labels[ind]].shape, batch1_weights.shape)
						batch1_code = model1.get_repr(batch1_inps)
						
						# real_loss = adversarial_weight(disc(batch1_code), valid[:batch1_code.size()[0]])
						# weights = torch.exp(real_loss.detach() - 0.8).clamp(1., 1.2)
						sample_loss = torch.nn.BCELoss(weight=batch1_weights.detach())
						real_loss = sample_loss(disc(batch1_code), valid[:batch1_code.size()[0]])

						# fake_loss = adversarial_weight(disc(batch2_code.detach()), fake)
						# weights = torch.exp(fake_loss.detach() - 0.8).clamp(1., 1.2)
						sample_loss = torch.nn.BCELoss()
						fake_loss = sample_loss(disc(batch2_code.detach()), fake)
						# real_loss = -torch.mean(disc(batch1_code))
						# fake_loss = torch.mean(disc(batch2_code.detach()))
						d_loss = 0.5 * (real_loss + fake_loss)

						d_loss.backward()
						optimizer_D.step()
						# for p in disc.parameters():
						# 	p.data.clamp_(-0.01, 0.01)
						s1 = ((s1*c1)+(float(d_loss.item())*len(batch1_code)))/(c1+len(batch1_code))
						c1 += len(batch1_code)

				pBar.set_description('Epoch {} G Loss: {:.3f} D Loss: {:.3f}'.format(epoch, s2, s1))
			if (s2 < 0.78) and (s1 < 0.78):
				count += 1
				self.test_model = model2
				torch.save(model2.state_dict(), self.path+"/best_br.pth")
				if test_labels is not None:
					print("Evaluating....")
					self.evaluate(test_gene_mat, test_labels, frac=0.05, name=None, test=True)

				if count >= 3:
					break

		if not os.path.isfile(self.path+"/best_br.pth"):
			torch.save(model2.state_dict(), self.path+"/best_br.pth")
			
		model2.load_state_dict(torch.load(self.path+"/best_br.pth"))
		self.test_model = model2

	def ftune_encoder(self, test_gene_mat, config, cmat=True):
		features = self.get_features(test_gene_mat)

		y_pred = self.predict(test_gene_mat, test=False)
		preds = self.filter_pred(y_pred, 0.)

		ind = preds != self.n_classes

		filtered_features = features[ind]
		filtered_labels = preds[ind]

		torch.manual_seed(config['seed'])
		torch.cuda.manual_seed(config['seed'])
		np.random.seed(config['seed'])
		torch.backends.cudnn.deterministic = True
		X_train, X_val, y_train, y_val = train_test_split(
			filtered_features, filtered_labels, test_size=config['val_frac'], shuffle=True, random_state=config['seed'])

		train_dataset = DataLoaderCustom(X_train, y_train)
		val_dataset = DataLoaderCustom(X_val, y_val)


		use_cuda = config['cuda']
		use_cuda = use_cuda and torch.cuda.is_available()

		device = torch.device("cuda" if use_cuda else "cpu")
		kwargs = {'num_workers': 4, 'pin_memory': False} if use_cuda else {}

		train_loader = torch.utils.data.DataLoader(train_dataset,
										   batch_size=config['batch_size'],
										   shuffle=True, **kwargs)

		val_loader = torch.utils.data.DataLoader(val_dataset,
										   batch_size=config['batch_size'],
										   shuffle=False, **kwargs)

		weights, n_classes = self.get_class_weights()
		class_weights = torch.FloatTensor(weights).to(device)

		criterion = torch.nn.NLLLoss(weight=class_weights)

		model = Classifier(X_train.shape[1], LDIM, MODEL_WIDTH, n_classes).to(device)
		model.load_state_dict(torch.load(self.path+"/best.pth"))

		for param in model.fc.parameters():
			param.requires_grad = False


		optimizer = optim.Adam(model.parameters(), lr=1e-4)
		sch = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, threshold=0.05, verbose=True)

		logger = {'tloss': [], 'val_acc': []}
		best_val_acc = 0.
		for epoch in range(config['epochs']):
			c, s = 0, 0
			pBar = tqdm(train_loader)
			model.train()
			for sample in pBar:
				x = sample['x'].to(device)
				y = sample['y'].to(device)
				
				optimizer.zero_grad()
				p = model.predict(x)
				loss = criterion(p, y)
				# print(loss)
				s = ((s*c)+(float(loss.item())*len(p)))/(c+len(p))
				c += len(p)
				pBar.set_description('Epoch {} Train: '.format(epoch) +str(round(float(s),4)))
				loss.backward()
				optimizer.step()
			logger['tloss'].append(s)
			sch.step(s)

			model.eval()
			y_pred, y_true = [], []
			with torch.no_grad():
				for sample in val_loader:
					x = sample['x'].to(device)
					y = sample['y'].to(device)
					
					p = model.predict_proba(x)
					y_pred.append(p.cpu().detach().numpy())
					y_true.append(y.cpu().detach().numpy())
			y_pred = np.concatenate(y_pred)
			y_true = np.concatenate(y_true)

			val_acc = (y_true == y_pred.argmax(axis=1)).mean()
			logger['val_acc'].append(val_acc)
			print("Validation Accuracy {:.4f}".format(val_acc))
			if val_acc >= best_val_acc:
				# print('Model improved')
				best_val_acc = val_acc
				torch.save(model.state_dict(), self.path+"/bestbr_ftuneencoder.pth")
				val_stats = {'pred': y_pred, 'true': y_true}

		if cmat:
			# Plot validation confusion matrix
			self.plot_cfmt(val_stats['pred'], val_stats['true'], 0.05, 'val_cfmtftuneencoder.pdf')

		# Finally keep the best model
		model.load_state_dict(torch.load(self.path+"/bestbr_ftuneencoder.pth"))
		self.modelftuned = model
		self.modelftuned.eval()

	def ftune(self, test_gene_mat, config, cmat=True):
		features = self.get_features(test_gene_mat)

		y_pred = self.predict(test_gene_mat, test=True)
		preds = self.filter_pred(y_pred, 0.1)

		ind = preds != self.n_classes

		filtered_features = features[ind]
		filtered_labels = preds[ind]

		values, counts = np.unique(filtered_labels, return_counts=True)

		torch.manual_seed(config['seed'])
		torch.cuda.manual_seed(config['seed'])
		np.random.seed(config['seed'])
		torch.backends.cudnn.deterministic = True

		if np.min(counts) > 1:
			X_train, X_val, y_train, y_val = train_test_split(
				filtered_features, filtered_labels, test_size=config['val_frac'], stratify=filtered_labels, shuffle=True, random_state=config['seed'])
		else:
			X_train, X_val, y_train, y_val = train_test_split(
				filtered_features, filtered_labels, test_size=config['val_frac'], shuffle=True, random_state=config['seed'])

		train_dataset = DataLoaderCustom(X_train, y_train)
		val_dataset = DataLoaderCustom(X_val, y_val)


		use_cuda = config['cuda']
		use_cuda = use_cuda and torch.cuda.is_available()

		device = torch.device("cuda" if use_cuda else "cpu")
		kwargs = {'num_workers': 4, 'pin_memory': False} if use_cuda else {}

		train_loader = torch.utils.data.DataLoader(train_dataset,
										   batch_size=config['batch_size'],
										   shuffle=True, **kwargs)

		val_loader = torch.utils.data.DataLoader(val_dataset,
										   batch_size=config['batch_size'],
										   shuffle=False, **kwargs)

		weights, n_classes = self.get_class_weights()
		class_weights = torch.FloatTensor(weights).to(device)

		criterion = torch.nn.NLLLoss(weight=class_weights)

		model1 = self.model.to(device)
		for param in model1.parameters():
			param.requires_grad = False
		# Define new model
		model_copy = Classifier(X_train.shape[1], LDIM, MODEL_WIDTH, self.n_classes).to(device)
		# Intialize it with the same parameter values as trained model
		model_copy.load_state_dict(model1.state_dict())


		for param in model_copy.parameters():
			param.requires_grad = True

		# model = model_copy
		model = ClassifierBig(model_copy, X_train.shape[1], LDIM, GLDIM).to(device)

		model.load_state_dict(self.test_model.to(device).state_dict())


		for param in model.parameters():
			param.requires_grad = False

		for param in model.m1.parameters():
			param.requires_grad = True


		optimizer = optim.Adam(model.parameters(), lr=1e-4)
		sch = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, threshold=0.05, verbose=True)

		logger = {'tloss': [], 'val_acc': []}
		best_val_acc = 0.
		for epoch in range(config['epochs']):
			c, s = 0, 0
			pBar = tqdm(train_loader)
			model.train()
			for sample in pBar:
				x = sample['x'].to(device)
				y = sample['y'].to(device)
				
				optimizer.zero_grad()
				p = model.predict(x)
				loss = criterion(p, y)
				# print(loss)
				s = ((s*c)+(float(loss.item())*len(p)))/(c+len(p))
				c += len(p)
				pBar.set_description('Epoch {} Train: '.format(epoch) +str(round(float(s),4)))
				loss.backward()
				optimizer.step()
			logger['tloss'].append(s)
			sch.step(s)

			model.eval()
			y_pred, y_true = [], []
			with torch.no_grad():
				for sample in val_loader:
					x = sample['x'].to(device)
					y = sample['y'].to(device)
					
					p = model.predict_proba(x)
					y_pred.append(p.cpu().detach().numpy())
					y_true.append(y.cpu().detach().numpy())
			y_pred = np.concatenate(y_pred)
			y_true = np.concatenate(y_true)

			val_acc = (y_true == y_pred.argmax(axis=1)).mean()
			logger['val_acc'].append(val_acc)
			print("Validation Accuracy {:.4f}".format(val_acc))
			if val_acc >= best_val_acc:
				# print('Model improved')
				best_val_acc = val_acc
				torch.save(model.state_dict(), self.path+"/bestbr_ftune.pth")
				val_stats = {'pred': y_pred, 'true': y_true}

		if cmat:
			# Plot validation confusion matrix
			self.plot_cfmt(val_stats['pred'], val_stats['true'], 0.05, 'val_cfmtftune.pdf')

		# Finally keep the best model
		model.load_state_dict(torch.load(self.path+"/bestbr_ftune.pth"))
		self.test_model = model
		self.test_model.eval()

	def set_test_model(self, model_type="BR"):
		if model_type=="BR":
			self.test_model.load_state_dict(torch.load(self.path+"/best_br.pth"))
		elif model_type=="BR_ftune":
			self.test_model.load_state_dict(torch.load(self.path+"/bestbr_ftune.pth"))
		self.test_model.eval()

	def to_pickle(self, name):
		self.raw_features = None
		self.reduced_features = None
		with open('{}/{}'.format(self.path, name), 'wb') as f:
			pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)


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


	obj = scRNALib(train_gene_mat, train_labels, path="blood_results")
	# obj.preprocess()
	obj.dim_reduction(5000, 'Var')

	train_config = {'val_frac': 0.2, 'seed': 0, 'batch_size': 128, 'cuda': False,
					'epochs': 15}
	
	obj.train_classifier(True, train_config, cmat=True)

	obj.raw_features = None
	obj.reduced_features = None
	with open('blood_results/scRNALib_obj.pkl', 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
	
	obj.evaluate(test_gene_mat, test_labels, frac=0.05, name="testcfmt.pdf")



	# pdb.set_trace()


if __name__ == "__main__":
	# MODEL_WIDTH = 3000
	main()


