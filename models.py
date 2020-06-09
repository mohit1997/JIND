import torch
from torch import nn, optim
from torch.nn import functional as F

class Classifier(nn.Module):
	def __init__(self, inp_dim, headdim, hdim, n_classes):
		super(Classifier, self).__init__()
		self.fc1 = nn.Sequential(
			# nn.Dropout(p=0.8),
			# nn.Dropout(p=0.3),
			nn.Linear(inp_dim, headdim),
			# nn.Dropout(p=0.2),
			# nn.ReLU(inplace=False),
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

class GaussianNoise(nn.Module):
	"""Gaussian noise regularizer.

	Args:
		sigma (float, optional): relative standard deviation used to generate the
			noise. Relative means that it will be multiplied by the magnitude of
			the value your are adding the noise to. This means that sigma can be
			the same regardless of the scale of the vector.
		is_relative_detach (bool, optional): whether to detach the variable before
			computing the scale of the noise. If `False` then the scale of the noise
			won't be seen as a constant but something to optimize: this will bias the
			network to generate vectors with smaller values.
	"""

	def __init__(self, sigma=0.1, is_relative_detach=True):
		super().__init__()
		self.sigma = sigma
		self.is_relative_detach = is_relative_detach
		self.noise = torch.tensor(0, dtype=torch.float)

	def forward(self, x):
		if self.training and self.sigma != 0:
			scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
			sampled_noise = self.noise.to(x.device).repeat(*x.size()).normal_() * scale
			x = x + sampled_noise
		return x

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
			# GaussianNoise(0.5),
			# nn.Dropout(p=0.1),
			nn.ReLU(inplace=False),
			# nn.BatchNorm1d(hdim),
			nn.Linear(hdim, hdim),
			# nn.LeakyReLU(0.2, inplace=False),
			nn.ReLU(inplace=False),
			# nn.BatchNorm1d(hdim),
			nn.Linear(hdim, headdim),
			# nn.Tanh(),
		)

		self.fc3 = nn.Sequential(
			# nn.Dropout(p=0.8),
			# nn.Dropout(p=0.3),
			nn.Linear(inp_dim, hdim),
			# nn.LeakyReLU(0.2, inplace=False),
			# GaussianNoise(0.5),
			# nn.Dropout(p=0.1),
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
		h = self.fc2(x) + rep * (1 + self.fc3(x))
		# h = h.clamp(-1., 1.)
		y = self.m1.fc(h) #+ self.fc2(x)

		return F.log_softmax(y, dim=1)

	def predict_proba(self, x):
		rep = self.m1.get_repr(x)
		out = torch.cat([rep, x], dim=1)
		h = self.fc2(x) + rep * (1 + self.fc3(x))
		# h = h.clamp(-1., 1.)
		y = self.m1.fc(h) #+ self.fc2(x)

		return F.softmax(y, dim=1)

	def get_repr(self, x):
		rep = self.m1.get_repr(x)
		out = torch.cat([rep, x], dim=1)
		h = self.fc2(x) + rep * (1 + self.fc3(x))
		# h = h.clamp(-1., 1.)
		return h, torch.norm(self.fc3(x))# + torch.norm(self.fc2(x))

class Discriminator(nn.Module):
	def __init__(self, dim):
		super(Discriminator, self).__init__()

		self.model = nn.Sequential(
			nn.Linear(dim, 512),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(512, 256),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(256, 256),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(256, 1),
			nn.Sigmoid(),
		)

	def forward(self, z):
		validity = self.model(z)
		return validity
