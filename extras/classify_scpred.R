if(!require("argparse")){
  install.packages("argparse")
}
# devtools::install_github("powellgenomicslab/scPred", ref = "31e7358952578b88a1d4ab4798a7f2e10bf37c42")
library("argparse")
library("scPred")
library("tidyverse")
library('Seurat')
library(Rfast)
library(doParallel)
library(caret)
library(irlba)
if(!require(reshape)){
  install.packages("reshape")
}
library(reshape)
library(reticulate)
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

parser <- ArgumentParser(description='Run scPred')
parser$add_argument('--train_path', default="/home/mohit/mohit/seq-rna/Comparison/datasets/pancreas_raw_sintegrated_01_01/train.pkl", type="character",
                    help='path to train data frame with labels')
parser$add_argument('--test_path', default="/home/mohit/mohit/seq-rna/Comparison/datasets/pancreas_raw_sintegrated_01_01/test.pkl", type="character",
                    help='path to test data frame with labels')
parser$add_argument('--column', type="character", default='labels',
                    help='column name for cell types')

eigenDecompose2 <- function(expData, n = 10, pseudo = TRUE, returnData = TRUE, seed = 66){
  
  # Parameter validations
  
  if(!is(expData, "matrix")){
    stop("Expression data must be a matrix object")
  }
  
  expData <- t(expData)
  
  zeroVar <- apply(expData, 2, var) == 0
  if(any(zeroVar)){
    expData <- expData[,!zeroVar]
    # message(paste0(sum(zeroVar), " following genes were removed as their variance is zero across all cells:"))
    # cat(paste0(names(zeroVar), collapse = "\n"), "\n", sep = "")
  }
  
  # Call prcomp() function
  message("Performing Lanczos bidiagonalization...")
  set.seed(66)
  svd <- prcomp_irlba(expData, n = n, center = TRUE, scale. = TRUE)
  class(svd)
  
  rownames(svd$x) <- rownames(expData)
  rownames(svd$rotation) <- colnames(expData)
  
  
  expData <- expData + 0
  nCells <- nrow(expData)
  
  
  f <- function(i) sqrt(sum((expData[, i] - svd$center[i])^2)/(nCells -  1L))
  scale. <- vapply(seq(ncol(expData)), f, pi, USE.NAMES = TRUE)
  
  
  names(scale.) <- names(svd$center)
  svd$scale <- scale.
  
  # Extract variance
  varianceExplained <- svd$sdev**2 / sum(svd$sdev**2)*100
  names(varianceExplained) <- colnames(svd$x)
  
  message("DONE!")
  
  svd <- svd[c("x", "rotation", "center", "scale", "sdev")]
  
  obj = new("scPred", svd = svd, expVar = varianceExplained, pseudo = pseudo, trainData = Matrix(t(expData), sparse=TRUE))
  return(obj)
  
}

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

r1 = rownames(batch1)
for(i in 1:nrow(batch1)) 
{
  old = r1[i]
  new = gsub(" ", ".", old, fixed = TRUE)
  new = gsub("-", ".", new, fixed = TRUE)
  new = gsub("_", ".", new, fixed = TRUE)
  new = gsub("/", ".", new, fixed = TRUE)
  new = sprintf("cell%d.%s",i, "train")
  r1[i] = new
}
rownames(batch1) = r1

r2 = rownames(batch2)
for(i in 1:nrow(batch2)) 
{
  old = r2[i]
  new = gsub(" ", ".", old, fixed = TRUE)
  new = gsub("-", ".", new, fixed = TRUE)
  new = gsub("_", ".", new, fixed = TRUE)
  new = gsub("/", ".", new, fixed = TRUE)
  new = sprintf("cell%d.%s",i, "test")
  r2[i] = new
}
rownames(batch2) = r2

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

m1var = apply(mat1, 2, var)
indices = m1var != 0
mat1_filt = mat1[, indices]
mat2_filt = mat2[, indices]
isint = check.integer(mat1_filt[1:100, 1:100]) == 1.

if (isint == TRUE){
  # mat1_filt = as.data.frame(scale(mat1_filt, center = FALSE,
  #       scale = colSums(mat1_filt))) * 1e4
  # mat2_filt = as.data.frame(scale(mat2_filt, center = FALSE,
  #                   scale = colSums(mat2_filt))) * 1e4
  mat1_filt = mat1_filt / (1e-5 + rowSums(mat1_filt))
  mat2_filt = mat2_filt / (1e-5 + rowSums(mat2_filt))
  mat1_filt = log(1 + mat1_filt)
  mat2_filt = log(1 + mat2_filt)
}


cl <- makePSOCKcluster(60)
registerDoParallel(cl)


params = preProcess(mat1_filt)
scaled_mat1 = predict(params, mat1_filt)
scaled_mat2 = predict(params, mat2_filt)

train_data = as.matrix(t(scaled_mat1))

test_data = as.matrix(t(scaled_mat2))

scp <- eigenDecompose2(train_data, n=10, pseudo=FALSE)
scPred::metadata(scp) <- metadata1
scp <- getFeatureSpace(scp, pVar = "labels")

plotEigen(scp, group = "labels")
scp <- trainModel(scp, seed = 66, allowParallel = TRUE)

# scp1 <- scPredict(scp, newData = test_data, threshold = 0.0)
# scp1@predMeta <- metadata2
# out = crossTab(scp1, true = "labels")
# out = as.matrix(out)
# out <- out[, order(as.numeric(as.factor(colnames(out))))]
# out <- out[order(as.numeric(as.factor(rownames(out)))),]
# pred = getPredictions(scp1)
# true = metadata2[,'labels']

scp_raw <- scPredict(scp, newData = test_data, threshold = 0.)
scp_raw@predMeta <- metadata2
pred_raw = getPredictions(scp_raw)

scp_filt <- scPredict(scp, newData = test_data, threshold = 0.9)
scp_filt@predMeta <- metadata2
pred_filt = getPredictions(scp_filt)

names(pred_raw)[names(pred_raw) == "predClass"] = "raw_predictions"
names(pred_filt)[names(pred_filt) == "predClass"] = "predictions"
pred_raw$raw_predictions = gsub('lab.', '', pred_raw$raw_predictions)
pred_filt$predictions = gsub('lab.', '', pred_filt$predictions)
pred_filt$predictions[pred_filt$predictions == "unassigned"] = "Unassigned"

metadata2$labels = gsub('lab.', '', metadata2$labels)

outputs = f1_score(factor(pred_raw$raw_predictions, levels=levels(factor(metadata2$labels))), factor(metadata2$labels))
mean_f1 = mean(outputs[[1]])
median_f1 = median(outputs[[1]])
weighted_f1 = sum(outputs[[2]] * outputs[[1]])

results = cbind(pred_raw["raw_predictions"], pred_filt["predictions"], metadata2["labels"])
colnames(results) = c("raw_predictions", "predictions", "labels")
index = pred_filt['predictions'] != "Unassigned"

rcomparison = metadata2$labels == pred_raw$raw_predictions
raw = mean(rcomparison)

comparison = metadata2[index, 'labels'] == pred_filt[index, 'predictions']

eff = mean(comparison)
filtered = 1 - mean(index)

stopCluster(cl)



pkl <- import("pickle")
path = sprintf("%s/scPred", dirname(args$train_path))
dir.create(path, showWarnings = FALSE)

file = sprintf("%s/test.log", path)

end_time <- Sys.time()
print(sprintf("Test raw %.4f eff %.4f rej %.4f mf1 %.4f medf1 %.4f wf1 %.4f", raw, eff, filtered, mean_f1, median_f1, weighted_f1))
cat(sprintf("Test raw %.4f eff %.4f rej %.4f mf1 %.4f medf1 %.4f wf1 %.4f", raw, eff, filtered, mean_f1, median_f1, weighted_f1), file = file)
cat(capture.output(end_time - start_time), file=file, append=TRUE)

colnames(pred_raw) = replacestring(colnames(pred_raw), "lab.", "")
predictions = results$predictions
raw_predictions = results$raw_predictions
labels = results$labels
results = cbind(pred_raw, predictions, labels)

output_path = sprintf("%s/scPred_assignment.pkl", path)
py_save_object(results, output_path)
