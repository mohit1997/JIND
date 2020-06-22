import setuptools
from setuptools import setup

setup(name='jind',
	version='0.1',
	description='single cell classification method',
	url='https://github.com/mohit1997/JIND',
	author='Flying Circus',
	author_email='goyal.mohit999@gmail.com',
	license='MIT',
	packages=setuptools.find_packages(),
	install_requires=[
			"h5py==2.10.0",
			"matplotlib==3.2.1",
			"numpy==1.18.4",
			"pandas==0.25.3",
			"scanpy==1.4.6",
			"scikit-learn==0.22.2.post1",
			"scipy==1.4.1",
			"seaborn==0.9.0",
			"sklearn==0.0",
			"statsmodels==0.11.1",
			"torch==1.3.1+cpu",
			"torchvision==0.4.2+cpu",
			"tornado==6.0.3",
			"tqdm==4.43.0",
			],
	zip_safe=False)