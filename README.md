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


# Documentation

## Download Dataset
```bash
wget ftp://ngs.sanger.ac.uk/production/teichmann/BBKNN/PBMC.merged.h5ad
```

## Process h5ad file (Required only for h5ad datasets)

Note: This code can be executed in an ipython terminal or jupyter notebook (avaliable here [PBMC Demo](/notebooks/PBMC-demo.ipynb), [PBMC Scratch](/notebooks/Process-data.ipynb))

```python
# Import Libraries
import scanpy as sc 
import pandas as pd
import numpy as np
```

```python
adata = sc.read("./PBMC.merged.h5ad")
adata # This prints all the attributes (columns) present in this object. We need to identify which one corresponds to cell type and batch (if available)
```

```python
data = adata.to_df() # Extracts gene expression matrix
data['batch'] = adata.obs['batch']
data['labels'] = adata.obs['Cell type']
```

```python
# Save pandas data frame to pickle file (This can be read in R as well)
data.to_pickle("data_annotated.pkl")
```

## Run JIND
```python
import numpy as np
from jind import JindLib

# Read Dataset
data = pd.read_pickle('data_annotated.pkl')

cell_ids = np.arange(len(data))
np.random.seed(0)

np.random.shuffle(cell_ids)
l = int(0.7*len(cell_ids))

train_data = data.iloc[cell_ids[:l]]
train_labels = train_data['labels'] # extract labels (Cells X 1)
train_gene_mat =  train_data.drop(['labels', 'batch'], 1) # extract gene expression matrix (Cells X Genes)

test_data = data.iloc[cell_ids[l:]]
test_labels = test_data['labels'] # extract labels (Cells X 1)
test_gene_mat =  test_data.drop(['labels', 'batch'], 1) # extract gene expression matrix (Cells X Genes)

# Create object
obj = JindLib(train_gene_mat, train_labels, path="my_results") # all outputs would be saved in "my_results" directory

# Select top 5000 features by maximum variance
obj.dim_reduction(5000, 'Var')

# Training hyperparameters
train_config = {'val_frac': 0.2, 'seed': 0, 'batch_size': 128, 'cuda': False, 'epochs': 10}
obj.train_classifier(train_config, cmat=True) #cmat=True plots and saves the validation confusion matrix
# Gives out Test Acc Pre 0.9889 Post 0.9442 Eff 0.9961 (reported on validation dataset)

# Evaluate
predictions = obj.evaluate(test_gene_mat, test_labels, frac=0.05, name="testcfmt.pdf") # frac is the outlier fraction filtering underconfident predictions
# Gives out Test Acc Pre 0.9854 Post 0.9421 Eff 0.9954 (reported on test dataset)

# save object for later evaluation
obj.to_pickle("jindobj.pkl")
```
