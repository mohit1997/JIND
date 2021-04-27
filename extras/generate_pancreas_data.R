if(!require("argparse")){
  install.packages("argparse")
}
library("argparse")
library("scPred")
library("tidyverse")
library('Seurat')
library(caret)
library(irlba)
if(!require(reshape)){
  install.packages("reshape")
}
library(reshape)
library(reticulate)
use_virtualenv("~/mohit/torch-cpu", required = TRUE)
py_config()

# Python Code for downloading data
# if not os.path.exists('rawdata-pancreas'):
#   #download SCESet objects
#   os.makedirs('rawdata-pancreas')
# os.system('wget https://scrnaseq-public-datasets.s3.amazonaws.com/scater-objects/baron-human.rds')
# os.system('wget https://scrnaseq-public-datasets.s3.amazonaws.com/scater-objects/muraro.rds')
# os.system('wget https://scrnaseq-public-datasets.s3.amazonaws.com/scater-objects/segerstolpe.rds')
# os.system('wget https://scrnaseq-public-datasets.s3.amazonaws.com/scater-objects/wang.rds')
# os.system('mv *.rds rawdata-pancreas')

data1 = readRDS("/home/mohit/mohit/seq-rna/pancreas/data/rawdata-pancreas/baron-human.rds")
data2 = readRDS("/home/mohit/mohit/seq-rna/pancreas/data/rawdata-pancreas/muraro.rds")
data3 = readRDS("/home/mohit/mohit/seq-rna/pancreas/data/rawdata-pancreas/segerstolpe.rds")
data4 = readRDS("/home/mohit/mohit/seq-rna/pancreas/data/rawdata-pancreas/wang.rds")

baron = as.data.frame(t(data1@assays$data[[1]]))
baron$labels = data1@colData$cell_type1
baron$batch = "baron"
py_save_object(baron, "/home/mohit/mohit/seq-rna/Comparison/data/pancreas_raw/baron.pkl")

muraro = as.data.frame(t(data2@assays$data[[1]]))
muraro$labels = data2@colData$cell_type1
muraro$batch = "muraro"
py_save_object(muraro, "/home/mohit/mohit/seq-rna/Comparison/data/pancreas_raw/muraro.pkl")

segerstolpe = as.data.frame(t(data3@assays$data[[1]]))
segerstolpe$labels = data3@colData$cell_type1
segerstolpe$batch = "segerstolpe"
py_save_object(segerstolpe, "/home/mohit/mohit/seq-rna/Comparison/data/pancreas_raw/segerstolpe.pkl")

wang = as.data.frame(t(data4@assays$data[[1]]))
wang$labels = data4@colData$cell_type1
wang$batch = "wang"
py_save_object(wang, "/home/mohit/mohit/seq-rna/Comparison/data/pancreas_raw/wang.pkl")


