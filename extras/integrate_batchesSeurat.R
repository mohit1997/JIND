if(!require("argparse")){
  install.packages("argparse")
}
library("argparse")
library(Seurat)
library(reticulate)
use_virtualenv("~/mohit/torch-cpu", required = TRUE)
py_config()



parser <- ArgumentParser(description='Run Seurat Integration')
parser$add_argument('--input_path1', default="datasets/human_blood_01/train.pkl", type="character",
                    help='path to input data frame1')
parser$add_argument('--input_path2', default="datasets/human_blood_01/test.pkl", type="character",
                    help='path to input data frame2')
parser$add_argument('--output_path', type="character",
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

sobj.list <- SplitObject(sobj, split.by = "batch")
# 
# # for (i in 1:length(pancreas.list)) {
# #   pancreas.list[[i]] <- NormalizeData(pancreas.list[[i]], verbose = FALSE)
# #   pancreas.list[[i]] <- FindVariableFeatures(pancreas.list[[i]], selection.method = "vst", 
# #                                              nfeatures = 2000, verbose = FALSE)
# # }
# filt_list = list()
# for (i in 1:length(sobj.list)) {
#   if (ncol(sobj.list[[i]]) >= 50){
#     filt_list = append(filt_list, sobj.list[[i]])
#   }
# }
# 
reference.list <- sobj.list

sobj.anchors <- FindIntegrationAnchors(object.list = reference.list, dims = 1:30)

sobj.integrated <- IntegrateData(anchorset = sobj.anchors, dims = 1:30)

output = sobj.integrated[['integrated']]
int_mat = as.matrix(output@data)
print(int_mat)
int_mat = as.data.frame(t(int_mat))

end_time <- Sys.time()
end_time - start_time
final_mat = cbind(int_mat, metadata)

pkl <- import("pickle")
py_save_object(final_mat, out_path)
