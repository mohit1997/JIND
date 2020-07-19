if(!require("argparse")){
  install.packages("argparse")
}
library("argparse")
library(Seurat)
library(reticulate)
use_virtualenv("~/mohit/torch-cpu", required = TRUE)
py_config()

parser <- ArgumentParser(description='Process some integers')
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
removable = c("batch", "labels")

pd <- import("pandas")
df <- pd$read_pickle(input)
mat = df[,!(names(df) %in% removable)]
metadata = df[,(names(df) %in% removable)]

sobj <- CreateSeuratObject(t(mat), meta.data = metadata)

sobj.list <- SplitObject(sobj, split.by = "batch")

# for (i in 1:length(pancreas.list)) {
#   pancreas.list[[i]] <- NormalizeData(pancreas.list[[i]], verbose = FALSE)
#   pancreas.list[[i]] <- FindVariableFeatures(pancreas.list[[i]], selection.method = "vst", 
#                                              nfeatures = 2000, verbose = FALSE)
# }
filt_list = list()
for (i in 1:length(sobj.list)) {
  if (ncol(sobj.list[[i]]) >= 50){
    filt_list = append(filt_list, sobj.list[[i]])
  }
}

reference.list <- filt_list
sobj.anchors <- FindIntegrationAnchors(object.list = reference.list, dims = 1:30)

sobj.integrated <- IntegrateData(anchorset = sobj.anchors, dims = 1:30)

output = sobj.integrated[['integrated']]
int_mat = as.matrix(output@data)
print(int_mat)
int_mat = as.data.frame(t(int_mat))
final_mat = cbind(int_mat, metadata)

pkl <- import("pickle")
py_save_object(final_mat, out_path)
