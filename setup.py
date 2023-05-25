import setuptools
from setuptools import setup

setup(name='jind',
	version='1.0.2',
	description='single cell classification method',
	url='https://github.com/mohit1997/JIND',
	author='Mohit Goyal',
	author_email='goyal.mohit999@gmail.com',
	license='MIT',
	packages=setuptools.find_packages(),
	install_requires=[
			"h5py==2.10.0",
			"matplotlib==3.2.1",
			"numpy==1.18.4",
			"pandas==1.1.5",
			"scanpy==1.4.6",
			"scikit-learn==0.22.2.post1",
			"scipy==1.4.1",
			"seaborn==0.9.0",
			"sklearn==0.0",
			"statsmodels==0.11.1",
			"torch==1.5.1",
			"torchvision==0.6.1",
			"tornado==6.3.2",
			"tqdm==4.43.0",
			"plotly",
			"kaleido",
			# "rpy2",
			"notebook",
			],
	zip_safe=False)
