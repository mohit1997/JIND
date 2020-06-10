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
data.to_pickle("dendrites_annotated.pkl")
```
