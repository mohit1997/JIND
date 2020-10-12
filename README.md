# JIND

 Single-cell RNA-seq is a powerful tool in the study of the cellular composition of different tissues and organisms. A key step in the analysis pipeline is the annotation of cell-types based on the expression of specific marker genes. Since manual annotation is labor-intensive and does not scale to large datasets, several methods for automated cell-type annotation have been proposed based on supervised learning. However, these methods generally require feature extraction and batch alignment prior to classification, and their performance may become unreliable in the presence of cell-types with very similar transcriptomic profiles, such as differentiating cells. We propose JIND, a framework for automated cell-type identification based on neural networks that directly learns a low-dimensional representation (latent code) in which cell-types can be reliably determined. To account for batch effects, JIND performs a novel asymmetric alignment in which the transcriptomic profile of unseen cells is mapped onto the previously learned latent space, hence avoiding the need of retraining the model whenever a new dataset becomes available. JIND also learns cell-type-specific confidence thresholds to identify and reject cells that cannot be reliably classified. We show on datasets with and without batch effects that JIND classifies cells more accurately than previously proposed methods while rejecting only a small proportion of cells. Moreover, JIND batch alignment is parallelizable, being more than five or six times faster than Seurat integration.

<img src="/figs/JINDOverviewIllustration-1.png" width="900px"/>
<!-- ![alt text](/figs/JINDOverviewIllustration.png?raw=true) -->


# Requirements
1. Python3
2. Numpy
3. Pandas
4. Pytorch

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

### Data
```python
from jind import JindLib

source_batch # Contains source batch gene expression matrix and cell types
target_batch # Contains target batch gene expression matrix and cell types

train_labels = source_batch['labels'] # extract cell-types (Cells X 1)
train_gene_mat =  source_batch.drop(['labels'], 1) # extract gene expression matrix (Cells X Genes)

test_labels = target_batch['labels'] # extract cell-types (Cells X 1)
test_gene_mat =  target_batch.drop(['labels'], 1) # extract gene expression matrix (Cells X Genes)
```

### Create JIND Object and Train
```python
# Create object
obj = JindLib(train_gene_mat, train_labels, path="my_results") # all outputs would be saved in "my_results" directory

# Select top 5000 genes by maximum variance (all genes are used if less than 5000 are avialable)
obj.dim_reduction(5000, 'Var')

# Training hyperparameters
train_config = {'val_frac': 0.2, 'seed': 0, 'batch_size': 128, 'cuda': False, 'epochs': 10}
obj.train_classifier(train_config, cmat=True) #cmat=True plots and saves the validation confusion matrix

# save object for later evaluation
obj.to_pickle("jindobj.pkl")


# For evaluation
predictions = obj.evaluate(test_gene_mat, test_labels, frac=0.05, name="testcfmt.pdf", test=False) # frac is the outlier fraction filtering underconfident predictions

# For just prediction
predicted_label  = obj.evaluate(test_mat, frac=0.05, name="testcfmtbr.pdf", test=False)
```


### JIND Asymmetric Alignment
```python
# JIND Batch Alignment
train_config = {'seed': 0, 'batch_size': 512, 'cuda': False,
				'epochs': 20}
obj.remove_effect(train_mat, test_mat, train_config)

# For evaluation
predicted_label  = obj.evaluate(test_mat, test_labels, frac=0.05, name="testcfmtbr.pdf", test=True)

# For just prediction
predicted_label  = obj.evaluate(test_mat, frac=0.05, name="testcfmtbr.pdf", test=True)


```

### JIND+ Self Training
```python

# JIND +
train_config = {'val_frac': 0.1, 'seed': 0, 'batch_size': 32, 'cuda': False,
				'epochs': 10}
obj.ftune(test_mat, train_config)

# For evaluation
predicted_label  = obj.evaluate(test_mat, test_labels, frac=0.05, name="testcfmtbr.pdf", test=True)

# For just prediction
predicted_label  = obj.evaluate(test_mat, frac=0.05, name="testcfmtbr.pdf", test=True)
```
