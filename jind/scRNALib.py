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
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class JindLib:
	global MODEL_WIDTH, LDIM
	MODEL_WIDTH = 1500
	LDIM = 256

	def __init__(self, gene_mat, cell_labels, path):
		self.class2num = None
		self.num2class = None
		self.reduced_features = None
		self.reduce_method = None
		self.model = None
		self.preprocessed = False
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

	def preprocess(self):
		print('Applying log transformation ...')
		self.preprocessed = True
		self.raw_features = np.log(1 + self.raw_features)


	def dim_reduction(self, num_features=5000, method='var', save_as=None):
		dim_size = num_features
		self.reduce_method = method

		if method == 'PCA':
			print('Performing PCA ...')
			self.pca = PCA(n_components=dim_size)
			self.reduced_features = self.pca.fit_transform(self.raw_features)
			if save_as is not None:
				np.save('{}_{}'.format(save_as, method), self.reduced_features)

		elif method == 'Var':
			print('Variance based reduction ...')
			self.variances = np.argsort(-np.var(self.raw_features, axis=0))[:dim_size]
			self.reduced_features = self.raw_features[:, self.variances]
			if save_as is not None:
				np.save('{}_{}'.format(save_as, method), self.reduced_features)

	def normalize(self):
		scaler = MinMaxScaler((0, 1))
		self.reduced_features = scaler.fit_transform(self.reduced_features)
		self.scaler = scaler


	def train_classifier(self, use_red, config, cmat=True):
		if use_red:
			if self.reduced_features is None:
				print("Please run obj.dim_reduction() or use use_red=False")
				sys.exit()
			features = self.reduced_features
		else:
			features = self.raw_features

		labels = self.labels

		torch.manual_seed(config['seed'])
		torch.cuda.manual_seed(config['seed'])
		np.random.seed(config['seed'])
		torch.backends.cudnn.deterministic = True
		X_train, X_val, y_train, y_val = train_test_split(
			features, labels, test_size=config['val_frac'], shuffle=True, random_state=config['seed'])

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

	def predict(self, test_gene_mat, test=False):
		features = test_gene_mat.values
		if self.preprocessed:
			features = np.log(1+features)
		if self.reduce_method is not None:
			if self.reduce_method == "Var":
				features = features[:, self.variances]
			elif self.reduce_method == "PCA":
				features = self.pca.transform(features)
		if self.scaler is not None:
			features = self.scaler.transform(features)

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
				
				p = model.predict_proba(x)
				y_pred.append(p.cpu().detach().numpy())
		y_pred = np.concatenate(y_pred)

		return y_pred

	def get_encoding(self, test_gene_mat, test=False):
		features = test_gene_mat.values
		if self.preprocessed:
			features = np.log(1+features)
		if self.reduce_method is not None:
			if self.reduce_method == "Var":
				features = features[:, self.variances]
			elif self.reduce_method == "PCA":
				features = self.pca.transform(features)
		if self.scaler is not None:
			features = self.scaler.transform(features)

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
				
				p = model.get_repr(x)
				y_pred.append(p.cpu().detach().numpy())
		y_pred = np.concatenate(y_pred, axis=0)

		return y_pred

	def evaluate(self, test_gene_mat, test_labels, frac=0.05, name=None, test=False):
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
		print('Test Acc Pre {:.4f} Post {:.4f} Eff {:.4f}'.format(pretest_acc, test_acc, pred_acc))

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
			fig = plt.figure(figsize=(10*factor,7*factor))
			cm_ob.plot(values_format='0.2f', ax=fig.gca())

			plt.title('Accuracy Pre {:.3f} Post {:.3f} Eff {:.3f} mAP {:.3f}'.format(pretest_acc, test_acc, pred_acc, np.mean(aps)))
			plt.tight_layout()
			plt.savefig('{}/{}'.format(self.path, name))

		return np.array([self.num2class[i] for i in preds])

	def plot_cfmt(self, y_pred, y_true, frac=0.05, name=None):
		if frac != 0:
			preds = self.filter_pred(y_pred, frac)
		else:
			preds = np.argmax(y_pred, axis=1)
		pretest_acc = (y_true == np.argmax(y_pred, axis=1)).mean() 
		test_acc = (y_true == preds).mean()
		ind = preds != self.n_classes
		pred_acc = (y_true[ind] == preds[ind]).mean()
		print('Test Acc Pre {:.4f} Post {:.4f} Eff {:.4f}'.format(pretest_acc, test_acc, pred_acc))

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
			fig = plt.figure(figsize=(10*factor,7*factor))
			cm_ob.plot(values_format='0.2f', ax=fig.gca())

			plt.title('Accuracy Pre {:.3f} Post {:.3f} Eff {:.3f} mAP {:.3f}'.format(pretest_acc, test_acc, pred_acc, np.mean(aps)))
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
		embeddings = TSNE(n_components=2, verbose=1, n_jobs=-1, perplexity=50).fit_transform(reduced_feats)
		return embeddings

	def remove_effect(self, train_gene_mat, test_gene_mat, config, test_labels=None):
		features_batch1 = train_gene_mat.values
		features_batch2 = test_gene_mat.values
		if self.preprocessed:
			features_batch1 = np.log(1+features_batch1)
			features_batch2 = np.log(1+features_batch2)
		if self.reduce_method is not None:
			if self.reduce_method == "Var":
				features_batch1 = features_batch1[:, self.variances]
				features_batch2 = features_batch2[:, self.variances]
			elif self.reduce_method == "PCA":
				features_batch1 = self.pca.transform(features_batch1)
				features_batch2 = self.pca.transform(features_batch2)
		if self.scaler is not None:
			features_batch1 = self.scaler.transform(features_batch1)
			features_batch2 = self.scaler.transform(features_batch2)
		
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
		model2 = ClassifierBig(model_copy,features_batch1.shape[1], LDIM, 256).to(device)

		disc = Discriminator(LDIM).to(device)

		# optimizer_G = torch.optim.Adam(model2.parameters(), lr=3e-4, betas=(0.5, 0.999))
		# optimizer_D = torch.optim.Adam(disc.parameters(), lr=1e-4, betas=(0.5, 0.999))
		optimizer_G = torch.optim.RMSprop(model2.parameters(), lr=1e-4, weight_decay=1e-5)
		optimizer_D = torch.optim.RMSprop(disc.parameters(), lr=5e-5, weight_decay=1e-6)
		adversarial_weight = torch.nn.BCELoss(reduction='none')
		adversarial_loss = torch.nn.BCELoss()
		sample_loss = torch.nn.BCELoss()


		Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

		bs = min(config['batch_size'], len(features_batch2), len(features_batch1))
		count = 0
		for epoch in range(config['epochs']):
			if len(batch2_loader) < 50:
				pBar = tqdm(range(50))
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
					weights = torch.exp(g_loss.detach() - 0.8).clamp(0.9, 1.5)
					sample_loss = torch.nn.BCELoss(weight=weights.detach())
					g_loss = sample_loss(disc(batch2_code), valid) + 0.01 * penalty
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
						batch1_code = model1.get_repr(batch1_inps)
						
						real_loss = adversarial_weight(disc(batch1_code), valid[:batch1_code.size()[0]])
						# weights = torch.exp(real_loss.detach() - 0.8).clamp(1., 1.2)
						# sample_loss = torch.nn.BCELoss(weight=weights.detach())
						real_loss = sample_loss(disc(batch1_code), valid[:batch1_code.size()[0]])

						fake_loss = adversarial_weight(disc(batch2_code.detach()), fake)
						# weights = torch.exp(fake_loss.detach() - 0.8).clamp(1., 1.2)
						# sample_loss = torch.nn.BCELoss(weight=weights.detach())
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

		model2.load_state_dict(torch.load(self.path+"/best_br.pth"))
		self.test_model = model2

	def ftune(self, test_gene_mat, config, cmat=True):
		features = test_gene_mat.values
		if self.preprocessed:
			features = np.log(1+features)
		if self.reduce_method is not None:
			if self.reduce_method == "Var":
				features = features[:, self.variances]
			elif self.reduce_method == "PCA":
				features = self.pca.transform(features)
		if self.scaler is not None:
			features = self.scaler.transform(features)

		y_pred = self.predict(test_gene_mat, test=True)
		preds = self.filter_pred(y_pred, 0.3)

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
		model = ClassifierBig(model_copy, X_train.shape[1], LDIM, 256).to(device)

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

		if cmat:
			# Plot validation confusion matrix
			self.plot_cfmt(self.val_stats['pred'], self.val_stats['true'], 0.05, 'val_cfmtftune.pdf')

		# Finally keep the best model
		model.load_state_dict(torch.load(self.path+"/bestbr_ftune.pth"))
		self.test_model = model
		self.test_model.eval()



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


