# scLearn

# Requirements
1. Python3
2. Numpy
3. Pandas
4. Pytorch


# Documentation

## Download Dataset
```bash
wget ftp://ngs.sanger.ac.uk/production/teichmann/BBKNN/PBMC.merged.h5ad
```

## Process h5ad file (Required only for h5ad datasets)

Note: This code can be executed in an ipython terminal or jupyter notebook (avaliable here)

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
from scRNALib import scRNALib

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
obj = scRNALib(train_gene_mat, train_labels, path="my_results") # all outputs would be saved in "my_results" directory

# Select top 5000 features by maximum variance
obj.dim_reduction(5000, 'Var')

# Training hyperparameters
train_config = {'val_frac': 0.2, 'seed': 0, 'batch_size': 128, 'cuda': False, 'epochs': 10}
obj.train_classifier(True, train_config, cmat=True) #cmat=True plots and saves the validation confusion matrix
# Gives out Test Acc Pre 0.9889 Post 0.9442 Eff 0.9961 (reported on validation dataset)

# Evaluate
predictions = obj.evaluate(test_gene_mat, test_labels, frac=0.05, name="testcfmt.pdf") # frac is the outlier fraction filtering underconfident predictions
# Gives out Test Acc Pre 0.9854 Post 0.9421 Eff 0.9954 (reported on test dataset)


