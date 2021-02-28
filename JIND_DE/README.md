# JIND_DE

## Introduction
Here you can find the code to reproduce the differential expression (DE) analysis to replicate the figures of the paper:
JIND: Joint Integration and Discriminationfor Automated Single-Cell Annotation.

## Scripts
* __plot_DEwithtSNE.R__: Runs a DE analysis between a correct classification cell type and the misclassified selected label. This script will produce as output two heatmaps with the results of the DE analysis, one with all the genes, and a second one with a few selection of genes. It also outputs the results of the DE analysis as a xlsx file and a tSNE with depicting the probabilities of the studied cells to belong to the correct label.
* __plot_DE_NC.R__: Runs a negative control DE analysis between the same selected type of cells, maintaining the same proportion of cells than in the misclassification DE analysis.
* __plot_cfmt_allmethods.R__: Computes the confusion matrices for all the methods studied on the paper.
* __plot_cfmt_ggplot.R__: Computes the confusion matrices for Seurat, JIND and JIND with rejection. 

<!--
## Data availability:
The output of the predictions for all the methods are available on the _Data_ folder.
The original gene expression data used for the DE analysis and for generating the predictions are available at XXX
-->
