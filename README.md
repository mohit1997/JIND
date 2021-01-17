# JIND

 JIND is a framework for automated cell-type identification based on neural networks. It directly learns a low-dimensional representation (latent code) inwhich cell-types can be reliably determined. To account for batch effects, JIND performs a novel asymmetric alignment in which the transcriptomic profileof unseen cells is mapped onto the previously learned latent space, hence avoiding the need of retraining the model whenever a new dataset becomes available. JIND also learns cell-type-specific confidence thresholds to identify and reject cells that cannot be reliably classified. We show on datasets with and without batch effects that JIND classifies cells more accurately than previously proposed methods while rejecting only a small proportion of cells. Moreover, JIND batch alignment is parallelizable, being more than five or six times faster than Seurat integration.

<img src="/figs/JINDOverviewIllustration-1.png" width="900px"/>


# Prerequisites
1. Linux or macOS
2. Python 3.6 or + (tested on 3.6.8)
3. CPU or NVIDIA GPU + CUDA CuDNN

# Installation

```bash
git clone https://github.com/mohit1997/JIND.git
cd JIND
python3 -m venv tc
source tc/bin/activate
pip install -e .
```

# Examples

## Demo Notebooks are avaliable here [PBMC Demo](/notebooks/PBMC-demo.ipynb), [PBMC Scratch](/notebooks/Process-data.ipynb)

## Executing JIND

### 1. Data
---
```python
from jind import JindLib

source_batch # Contains source batch gene expression matrix and cell types
target_batch # Contains target batch gene expression matrix and cell types

train_labels = source_batch['labels'] # extract cell-types (Cells X 1)
train_gene_mat =  source_batch.drop(['labels'], 1) # extract gene expression matrix (Cells X Genes)

test_labels = target_batch['labels'] # extract cell-types (Cells X 1)
test_gene_mat =  target_batch.drop(['labels'], 1) # extract gene expression matrix (Cells X Genes)
```

### 2. Create JIND Object and Train
---
```python
# Create object
obj = JindLib(train_gene_mat, train_labels, path="my_results") # all outputs would be saved in "my_results" directory

# Log transform the dataset if the data is integral
mat = train_gene_mat.values
mat_round = np.rint(mat)
error = np.mean(np.abs(mat - mat_round))
if error == 0:
	print("Data is int")
	obj.preprocess()

# Select top 5000 genes by maximum variance (all genes are used if less than 5000 are avialable)
obj.dim_reduction(5000, 'Var')

# Training hyperparameters
train_config = {'val_frac': 0.2, 'seed': 0, 'batch_size': 128, 'cuda': False, 'epochs': 10} 
# val_frac : proportion of data used for validation
# seed : random seed
# batch size : number of data points used for on iteration of gradient descent
# cuda : True if GPU avaliable
# epoch : Number of epochs/passes over the whole training data
obj.train_classifier(train_config, cmat=True) #cmat=True plots and saves the validation confusion matrix

# save object for later evaluation
obj.to_pickle("jindobj.pkl")
```


### 3. JIND Asymmetric Alignment
---
```python
# Load JIND Model. JIND doesn't save the training data for efficient memory usage. Therefore training data needs to explicitly provided and preprocessed again.
import pickle
path = "my_results"

with open('{}/jindobj.pkl'.format(path), 'rb') as f:
	obj = pickle.load(f)

obj.raw_features = train_gene_mat.values

# Log transform the dataset if gene expression matrix is integral
mat = train_gene_mat.values
mat_round = np.rint(mat)
error = np.mean(np.abs(mat - mat_round))
if error == 0:
	print("Data is int")
	obj.preprocess()

obj.dim_reduction(5000, 'Var')

# JIND Batch Alignment
train_config = {'seed': 0, 'batch_size': 512, 'cuda': False,
				'epochs': 20}
obj.remove_effect(train_gene_mat, test_gene_mat, train_config)

# For evaluation (test labels are needed in this case)
predicted_label  = obj.evaluate(test_gene_mat, test_labels, frac=0.05, name="testcfmtbr.pdf", test=True)
# frac: outlier fraction (theta) filtering underconfident predictions
# test: False when using JIND without batch alignment
# name: file name dor saving the confusion matrix. Confusion matrix is not plotted if None provided.
# predicted_label has three columns: cellname, raw_predictions (before rejection) and predictions (after rejection). If test_labels are provided, then labels are added as a column in the output.

# For just prediction (test labels are not needed in this case)
predicted_label  = obj.evaluate(test_gene_mat, frac=0.05, name="testcfmtbr.pdf", test=True)

# Save the predictions for downstream tasks
predicted_label.to_csv("labels.csv")
```

### 4. JIND+ Self Training
---
```python

# JIND + (this step must be performed after batch alignment)
train_config = {'val_frac': 0.1, 'seed': 0, 'batch_size': 32, 'cuda': False,
				'epochs': 10}
obj.ftune(test_gene_mat, train_config)

# For evaluation
predicted_label  = obj.evaluate(test_gene_mat, test_labels, frac=0.05, name="testcfmtbr.pdf", test=True)
```
