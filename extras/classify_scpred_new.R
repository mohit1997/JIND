if(!require("argparse")){
  install.packages("argparse")
}
# install.packages("doParallel", repos="http://R-Forge.R-project.org")
# devtools::install_github("powellgenomicslab/scPred")
library("argparse")
library(Seurat)
library("scPred")
library(ggplot2)
library(caret)
library(dplyr)
library(purrr)
library(tidyr)
library(yardstick)
library(doParallel)
library(reticulate)
library(Rfast)
if(!require(reshape)){
  install.packages("reshape")
}
library(reshape)
use_virtualenv("~/mohit/torch-cpu", required = TRUE)
py_config()

check.integer <- function(x) {
  mean(x == round(x))
}

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

parser <- ArgumentParser(description='Run scPred Classifier')
parser$add_argument('--train_path', default="/home/mohit/mohit/seq-rna/Comparison/datasets/mouse_dataset_random/train.pkl", type="character",
                    help='path to train data frame with labels')
parser$add_argument('--test_path', default="/home/mohit/mohit/seq-rna/Comparison/datasets/mouse_dataset_random/test.pkl", type="character",
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
m1var = apply(mat1, 2, var)
indices = m1var != 0


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

mat1_filt = mat1[, indices]
mat2_filt = mat2[, indices]
reference <- CreateSeuratObject(t(mat1_filt), meta.data = metadata1)
query <- CreateSeuratObject(t(mat2_filt), meta.data = metadata2)

isint = check.integer(mat1_filt) == 1.
cl <- makePSOCKcluster(30)
registerDoParallel(cl)

if (isint == TRUE){
  reference <- reference %>%
    NormalizeData() %>%
    FindVariableFeatures() %>%
    ScaleData() %>%
    RunPCA()
} else {
  reference <- reference %>%
    # NormalizeData() %>%
    FindVariableFeatures() %>%
    ScaleData() %>%
    RunPCA()
}




reference <- getFeatureSpace(reference, "labels")
reference <- trainModel(reference, model = "mda", allowParallel = TRUE)



scp <- get_scpred(reference)
if (isint == TRUE){
  query <- NormalizeData(query)
}

query <- scPredict(query, reference, recompute_alignment = TRUE, threshold = 0.0)
preds_raw = query$scpred_prediction

query <- scPredict(query, reference, recompute_alignment = FALSE, threshold = 0.9)
preds = query$scpred_prediction
labs = query$labels

preds_raw = gsub('lab.', '', preds_raw)
preds = gsub('lab.', '', preds)
preds[preds == "unassigned"] = "Unassigned"
labs = gsub('lab.', '', labs)

results = cbind(preds_raw, preds, labs)
colnames(results) = c("raw_predictions", "predictions", "labels")
index = preds != "Unassigned"

comparison = labs[index] == preds[index]

eff = mean(comparison)
filtered = 1 - mean(index)
raw_acc = mean(preds_raw == labs)

outputs = f1_score(factor(preds_raw, levels=levels(factor(labs))), factor(labs))
mean_f1 = mean(outputs[[1]])
median_f1 = median(outputs[[1]])
weighted_f1 = sum(outputs[[2]] * outputs[[1]])

stopCluster(cl)

pkl <- import("pickle")
path = sprintf("%s/scPrednew", dirname(args$train_path))
dir.create(path, showWarnings = FALSE)

file = sprintf("%s/test.log", path)

end_time <- Sys.time()
print(sprintf("Test Accuracy %.4f w.f. %.4f filtered %.4f mf1 %.4f medf1 %.4f wf1 %.4f", raw_acc, eff, filtered, mean_f1, median_f1, weighted_f1))
cat(sprintf("Test Accuracy %.4f w.f. %.4f filtered %.4f mf1 %.4f medf1 %.4f wf1 %.4f", raw_acc, eff, filtered, mean_f1, median_f1, weighted_f1), file = file)
cat(capture.output(end_time - start_time), file=file, append=TRUE)

output_path = sprintf("%s/scPred_assignment.pkl", path)
py_save_object(results, output_path)