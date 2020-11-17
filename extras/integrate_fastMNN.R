if(!require("argparse")){
  install.packages("argparse")
}
library("argparse")
library(Seurat)
library(SeuratData)
library(SeuratWrappers)

library(reticulate)
use_virtualenv("~/mohit/torch-cpu", required = TRUE)
py_config()

parser <- ArgumentParser(description='Integrate with FastMNN')
parser$add_argument('--input_path', default="data/pancreas_annotatedbatched.pkl", type="character",
                    help='path to input data frame')
parser$add_argument('--output_path', type="character",
                    help='path to output data frame')
parser$add_argument('--column', type="character", default='batch',
                    help='column name to split along')
parser$add_argument('--removables', type="character", nargs='+',
                    help='columns to be removed')
args <- parser$parse_args()
input = args$input_path
out_path = args$output_path
removable = args$removables
removable = c(removable, "batch", "labels")

pd <- import("pandas")
df <- pd$read_pickle(input)
mat = df[,!(names(df) %in% removable)]
metadata = df[,(names(df) %in% removable)]

sobj <- CreateSeuratObject(t(mat), meta.data = metadata)

# sobj <- NormalizeData(sobj)
sobj <- FindVariableFeatures(sobj)

sobj.integrated <- RunFastMNN(object.list = SplitObject(sobj, split.by = "batch"))

output = sobj.integrated[['mnn']]
int_mat = as.matrix(output@cell.embeddings)
print(int_mat)
int_mat = as.data.frame(int_mat)
final_mat = cbind(int_mat, metadata)

pkl <- import("pickle")
py_save_object(final_mat, out_path)