import torch
from torch import nn, optim
from torch.nn import functional as F

class Classifier(nn.Module):
	def __init__(self, inp_dim, headdim, hdim, n_classes):
		super(Classifier, self).__init__()
		self.fc1 = nn.Sequential(
			# nn.Dropout(p=0.8),
			# nn.Dropout(p=0.3),
			nn.Linear(inp_dim, hdim),
			nn.Dropout(p=0.2),
			# nn.ReLU(inplace=False),
			nn.Linear(hdim, headdim),
			nn.Dropout(p=0.2),
			# nn.BatchNorm1d(headdim),
			# nn.ReLU(True),
			# nn.Tanh(),
			# nn.ReLU(inplace=False),
		)
		self.fc2 = nn.Sequential(
			# nn.Dropout(p=0.8),
			nn.Dropout(p=0.1),
			nn.Linear(inp_dim, n_classes),
		)

		self.fc = nn.Sequential(
			nn.ReLU(inplace=False),
			nn.Linear(headdim, n_classes),
		)
	
	def predict(self, x):
		h = self.fc1(x)
		y = self.fc(h) #+ self.fc2(x)

		return F.log_softmax(y, dim=1)

	def predict_proba(self, x):
		h = self.fc1(x)
		y = self.fc(h) #+ self.fc2(x)

		return F.softmax(y, dim=1)

	def get_repr(self, x):
		return self.fc1(x)

class ClassifierBig(nn.Module):
	def __init__(self, model, inp_dim, headdim, hdim):
		super(ClassifierBig, self).__init__()
		self.m1 = model
		for param in self.m1.parameters():
			param.requires_grad = False

		self.fc2 = nn.Sequential(
			# nn.Dropout(p=0.8),
			# nn.Dropout(p=0.3),
			nn.Linear(inp_dim, hdim),
			# nn.LeakyReLU(0.2, inplace=False),
			nn.ReLU(inplace=False),
			# nn.BatchNorm1d(hdim),
			nn.Linear(hdim, hdim),
			# nn.LeakyReLU(0.2, inplace=False),
			nn.ReLU(inplace=False),
			# nn.BatchNorm1d(hdim),
			nn.Linear(hdim, headdim),
			# nn.Tanh(),
		)

	def predict(self, x):
		rep = self.m1.get_repr(x)
		out = torch.cat([rep, x], dim=1)
		h = self.fc2(x) + rep
		# h = h.clamp(-1., 1.)
		y = self.m1.fc(h) #+ self.fc2(x)

		return F.log_softmax(y, dim=1)

	def predict_proba(self, x):
		rep = self.m1.get_repr(x)
		out = torch.cat([rep, x], dim=1)
		h = self.fc2(x) + rep
		# h = h.clamp(-1., 1.)
		y = self.m1.fc(h) #+ self.fc2(x)

		return F.softmax(y, dim=1)

	def get_repr(self, x):
		rep = self.m1.get_repr(x)
		out = torch.cat([rep, x], dim=1)
		h = self.fc2(x) + rep
		# h = h.clamp(-1., 1.)
		return h

class Discriminator(nn.Module):
	def __init__(self, dim):
		super(Discriminator, self).__init__()

		self.model = nn.Sequential(
			nn.Linear(dim, 256),
			# nn.Dropout(p=0.2),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(256, 256),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(256, 256),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(256, 1),
			nn.Sigmoid(),
		)

	def forward(self, z):
		validity = self.model(z)
		return validity
