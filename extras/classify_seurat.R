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
library(Rfast)
if(!require(reshape)){
  install.packages("reshape")
}
library(reshape)
use_virtualenv("~/mohit/torch-cpu", required = TRUE)
py_config()

f1_score <- function(predicted, expected, positive.class="1") {
  predicted <- factor(as.character(predicted), levels=levels(factor(expected)))
  expected  <- as.factor(expected)
  cm = as.matrix(table(expected, predicted))
  
  precision <- diag(cm) / colSums(cm)
  recall <- diag(cm) / rowSums(cm)
  f1 <-  ifelse(precision + recall == 0, 0, 2 * precision * recall / (precision + recall))
  
  #Assuming that F1 is zero when it's not possible compute it
  f1[is.na(f1)] <- 0
  
  return (list(f1, rowsums(cm)/sum(rowsums(cm))))
}

start_time <- Sys.time()

parser <- ArgumentParser(description='Run Seurat Classifier')
parser$add_argument('--train_path', default="/home/mohit/mohit/seq-rna/Comparison/datasets/pancreas_raw_01/train.pkl", type="character",
                    help='path to train data frame with labels')
parser$add_argument('--test_path', default="/home/mohit/mohit/seq-rna/Comparison/datasets/pancreas_raw_01/test.pkl", type="character",
                    help='path to test data frame with labels')
parser$add_argument('--column', type="character", default='labels',
                    help='column name for cell types')

args <- parser$parse_args()

replacestring <- function(inp, x, y) {
  for (i in 1:length(inp)){
    st = inp[i]
    st = gsub(x, y, st)
    inp[i] = st
  }
  return(inp)
}

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


reference <- FindVariableFeatures(reference, selection.method = "vst", nfeatures = 2000)
reference <- ScaleData(reference, verbose = FALSE)
reference <- RunPCA(reference, npcs = 30, verbose = FALSE)

query <- FindVariableFeatures(query, selection.method = "vst", nfeatures = 2000)

anchors <- FindTransferAnchors(reference = reference, query = query, dims = 1:30)
predictions <- TransferData(anchorset = anchors, refdata = reference$labels, dims = 1:30)
query <- AddMetaData(query, metadata = predictions)

sprintf("Test Accuracy %f", mean(query$predicted.id == query$labels))
common_ctypes = sort(ctypes)
predictions = query$predicted.id
predictions = replacestring(predictions, "lab.", "")

labels = query$labels
labels = replacestring(as.character(labels), "lab.", "")

results = cbind(predictions, predictions, labels)
colnames(results) = c("raw_predictions", "predictions", "labels")

for (i in 1:length(common_ctypes)){
  st = sprintf("prediction.score.%s", common_ctypes[i])
  # st = gsub(" ", ".", st)
  if (i == 1){
    scores = query@meta.data[st]
  } else {
    scores = cbind(scores, query@meta.data[st])
  }
  
}

colnames(scores) = replacestring(colnames(scores), "prediction.score.lab.", "")
out = apply(scores, 1, max)

results[, "predictions"][out < 0.9] = "Unassigned"
filtered_results = results[out > 0.9]

outputs = f1_score(factor(results[, "raw_predictions"], levels=levels(factor(results[, "labels"]))), factor(results[, "labels"]))
mean_f1 = mean(outputs[[1]])
median_f1 = median(outputs[[1]])
weighted_f1 = sum(outputs[[2]] * outputs[[1]])

cm = confusionMatrix(factor(results[, "raw_predictions"], levels=levels(factor(results[, "labels"]))), factor(results[, "labels"]))
cm = as.data.frame.matrix(cm$table)
cm = t(cm/colSums(cm)[col(cm)])

pkl <- import("pickle")
path = sprintf("%s/seurat", dirname(args$train_path))
dir.create(path, showWarnings = FALSE)

file = sprintf("%s/test.log", path)
end_time <- Sys.time()
cat(sprintf("Raw Accuracy %f \n", mean(results[,"raw_predictions"] == results[, "labels"])), file = file)
cat(sprintf("Eff Accuracy %f \n", mean(results[,"predictions"] == results[, "labels"])), file = file, append=TRUE)
cat(sprintf("Filtered %f \n", mean(out < 0.9)), file = file, append=TRUE)
cat(sprintf("mf1 %f \n", mean_f1), file = file, append=TRUE)
cat(sprintf("medf1 %f \n", median_f1), file = file, append=TRUE)
cat(sprintf("wf1 %f \n", weighted_f1), file = file, append=TRUE)
cat(capture.output(end_time - start_time), file=file, append=TRUE)

output_path = sprintf("%s/seurat_assignment.pkl", path)
py_save_object(as.data.frame(results), output_path)


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

print(thisplot)
savePlot <- function(myPlot) {
  pdf(sprintf("%s/test_confusion_matrix.pdf", path))
  print(myPlot)
  dev.off()
}

savePlot(thisplot)
