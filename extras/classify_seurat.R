if(!require("argparse")){
  install.packages("argparse")
}
library("argparse")
library(Seurat)
library(ggplot2)
library(caret)
library(dplyr)
library(purrr)
library(tidyr)
library(yardstick)
library(reticulate)
if(!require(reshape)){
  install.packages("reshape")
}
library(reshape)
use_virtualenv("~/mohit/torch-cpu", required = TRUE)
py_config()

start_time <- Sys.time()

parser <- ArgumentParser(description='Run Seurat Classifier')
parser$add_argument('--train_path', default="datasets/pancreas_integrated_01/train.pkl", type="character",
                    help='path to train data frame with labels')
parser$add_argument('--test_path', default="datasets/pancreas_integrated_01/test.pkl", type="character",
                    help='path to test data frame with labels')
parser$add_argument('--column', type="character", default='labels',
                    help='column name for cell types')

args <- parser$parse_args()

pd <- import("pandas")
lname = args$column

batch1 = pd$read_pickle(args$train_path)

batch2 = pd$read_pickle(args$test_path)

mat1 = batch1[,!(names(batch1) %in% c(lname))]
metadata1 = as.matrix(batch1[lname])
colnames(metadata1) <- c("labels")

for(i in 1:nrow(metadata1)) 
{
  old = metadata1[i,"labels"]
  new = gsub(" ", ".", old, fixed = TRUE)
  new = gsub("-", ".", new, fixed = TRUE)
  new = gsub("_", ".", new, fixed = TRUE)
  new = gsub("/", ".", new, fixed = TRUE)
  new = sprintf("lab.%s", new)
  metadata1[i,"labels"] = new
}

ctypes = unique(unlist(metadata1))

mat2 = batch2[,!(names(batch2) %in% c(lname))]
metadata2 = as.matrix(batch2[lname])
colnames(metadata2) <- c("labels")


for(i in 1:nrow(metadata2)) 
{
  old = metadata2[i,"labels"]
  new = gsub(" ", ".", old, fixed = TRUE)
  new = gsub("-", ".", new, fixed = TRUE)
  new = gsub("_", ".", new, fixed = TRUE)
  new = gsub("/", ".", new, fixed = TRUE)
  new = sprintf("lab.%s", new)
  metadata2[i,"labels"] = new
}

metadata1 = as.data.frame(metadata1)
metadata2 = as.data.frame(metadata2)

reference <- CreateSeuratObject(t(mat1), meta.data = metadata1)
query <- CreateSeuratObject(t(mat2), meta.data = metadata2)
# pancreas.list <- SplitObject(pancreas, split.by = "batch")


# reference <- pancreas.list[[c("0")]]
reference <- FindVariableFeatures(reference, selection.method = "vst", nfeatures = 2000)
reference <- ScaleData(reference, verbose = FALSE)
reference <- RunPCA(reference, npcs = 30, verbose = FALSE)

# query <- pancreas.list[[c("1")]]
query <- FindVariableFeatures(query, selection.method = "vst", nfeatures = 2000)

anchors <- FindTransferAnchors(reference = reference, query = query, dims = 1:30)
predictions <- TransferData(anchorset = anchors, refdata = reference$labels, dims = 1:30)
query <- AddMetaData(query, metadata = predictions)

sprintf("Test Accuracy %f", mean(query$predicted.id == query$labels))
common_ctypes = sort(ctypes)
predicted = query$predicted.id
for (i in 1:length(common_ctypes)){
  st = sprintf("prediction.score.%s", common_ctypes[i])
  print(st)
  # st = gsub(" ", ".", st)
  predicted = cbind(predicted, query@meta.data[st])
}
predicted = cbind(predicted, query$labels)
colnames(predicted) <- c(colnames(predicted)[1:ncol(predicted)-1], "labels")
cm = confusionMatrix(factor(predicted$predicted), factor(query$labels))
cm = as.data.frame.matrix(cm$table)
cm = t(cm/colSums(cm)[col(cm)])

pkl <- import("pickle")
path = sprintf("%s/seurat", dirname(args$train_path))
dir.create(path, showWarnings = FALSE)

file = sprintf("%s/test.log", path)
end_time <- Sys.time()
cat(sprintf("Test Accuracy %f \n", mean(query$predicted.id == query$labels)), file = file)
cat(capture.output(end_time - start_time), file=file, append=TRUE)

output_path = sprintf("%s/seurat_matrix.pkl", path)
py_save_object(predicted, output_path)


cm = data.frame(melt(cm))
name = sprintf("Test Accuracy %f", mean(query$predicted.id == query$labels))
thisplot <- ggplot(cm, aes(x = X2, y = X1)) +
  geom_raster(aes(fill=value)) +
  geom_text(aes(label = sprintf("%.2f", value)), vjust = 1) +
  scale_fill_gradient(low="red", high="green") +
  labs(x="Predicted Labels", y="True Labels", title=name) +
  xlim((levels(cm$X2))) +
  ylim(rev(levels(cm$X1))) +
  theme_bw() + theme(axis.text.x=element_text(size=9, angle=0, vjust=0.3),
                     axis.text.y=element_text(size=9),
                     plot.title=element_text(size=11))

savePlot <- function(myPlot) {
  pdf(sprintf("%s/test_confusion_matrix.pdf", path))
  print(myPlot)
  dev.off()
}

savePlot(thisplot)
