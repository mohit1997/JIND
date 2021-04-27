if(!require("argparse")){
  install.packages("argparse")
}
# devtools::install_github("immunogenomics/harmony")
library("argparse")
library(Seurat)
library(harmony)
library(reticulate)
use_virtualenv("~/mohit/torch-cpu", required = TRUE)
py_config()



parser <- ArgumentParser(description='Run Harmony Integration')
parser$add_argument('--input_path1', default="datasets/pancreas_raw_01/train.pkl", type="character",
                    help='path to input data frame1')
parser$add_argument('--input_path2', default="datasets/pancreas_raw_01/test.pkl", type="character",
                    help='path to input data frame2')
parser$add_argument('--output_path', type="character", default="data/pancreas_raw_01_harmony_integrated.pkl",
                    help='path to output data frame')
parser$add_argument('--column', type="character", default='batch',
                    help='column name to split along')
parser$add_argument('--removables', type="character", nargs='+',
                    help='columns to be removed')
args <- parser$parse_args()
input1 = args$input_path1
input2 = args$input_path2
out_path = args$output_path
removable = args$removables
removable = c(removable, "labels")

removable

pd <- import("pandas")

df1 <- pd$read_pickle(input1)
mat1 = df1[,!(names(df1) %in% removable)]
metadata1 = as.data.frame(df1[,(names(df1) %in% removable)])
colnames(metadata1) = "labels"
metadata1['batch'] = 1


df2 <- pd$read_pickle(input2)
mat2 = df2[,!(names(df2) %in% removable)]
metadata2 = as.data.frame(df2[,(names(df2) %in% removable)])
colnames(metadata2) = "labels"
metadata2['batch'] = 2

mat = rbind(mat1, mat2)
metadata = rbind(metadata1, metadata2)
rownames(metadata) = rownames(mat)

start_time <- Sys.time()

sobj <- CreateSeuratObject(t(mat), meta.data = metadata)
sobj <- FindVariableFeatures(sobj, selection.method = "vst", nfeatures = 5000)
sobj <- ScaleData(sobj, assay = 'RNA')

sobj <- RunPCA(sobj, assay = 'RNA', npcs = 5000, approx = FALSE)

sobj <- RunHarmony(sobj, "batch")
sobj <- RunUMAP(sobj, reduction="harmony", dims=1:5000)

DimPlot(sobj, group.by="batch")
DimPlot(sobj, reduction='umap', group.by='labels')

out_df <- sobj@reductions$pca@cell.embeddings
nrow(out_df)
ncol(out_df)

int_mat = as.data.frame(out_df)

end_time <- Sys.time()
end_time - start_time
final_mat = cbind(int_mat, metadata)

pkl <- import("pickle")
py_save_object(final_mat, out_path)
