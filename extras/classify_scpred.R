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

start_time <- Sys.time()

parser <- ArgumentParser(description='Run scPred')
parser$add_argument('--train_path', default="datasets/mouse_dataset_random/train.pkl", type="character",
                    help='path to train data frame with labels')
parser$add_argument('--test_path', default="datasets/mouse_dataset_random/test.pkl", type="character",
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

params = preProcess(mat1)
scaled_mat1 = predict(params, mat1)
scaled_mat2 = predict(params, mat2)

train_data = as.matrix(t(scaled_mat1))

test_data = as.matrix(t(scaled_mat2))

scp <- eigenDecompose2(train_data, n=10, pseudo=FALSE)
scPred::metadata(scp) <- metadata1
scp <- getFeatureSpace(scp, pVar = "labels")

plotEigen(scp, group = "labels")
scp <- trainModel(scp, seed = 66)

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

comparison = metadata2['labels'] == pred_raw['predClass']
raw = mean(comparison)

scp_filt <- scPredict(scp, newData = test_data, threshold = 0.9)
scp_filt@predMeta <- metadata2
pred_filt = getPredictions(scp_filt)

names(pred_raw)[names(pred_raw) == "predClass"] = "raw_predictions"
names(pred_filt)[names(pred_filt) == "predClass"] = "predictions"
pred_raw$raw_predictions = gsub('lab.', '', pred_raw$raw_predictions)
pred_filt$predictions = gsub('lab.', '', pred_filt$predictions)
pred_filt$predictions[pred_filt$predictions == "unassigned"] = "Unassigned"

metadata2$labels = gsub('lab.', '', metadata2$labels)

results = cbind(pred_raw["raw_predictions"], pred_filt["predictions"], metadata2["labels"])
colnames(results) = c("raw_predictions", "predictions", "labels")
index = pred_filt['predictions'] != "Unassigned"

comparison = metadata2[index, 'labels'] == pred_filt[index, 'predictions']

eff = mean(comparison)
filtered = 1 - mean(index)

pkl <- import("pickle")
path = sprintf("%s/scPred", dirname(args$train_path))
dir.create(path, showWarnings = FALSE)

file = sprintf("%s/test.log", path)

end_time <- Sys.time()
print(sprintf("Test Accuracy %f w.f. %f filtered %f", raw, eff, filtered))
cat(sprintf("Test Accuracy %f w.f. %f filtered %f \n ", raw, eff, filtered), file = file)
cat(capture.output(end_time - start_time), file=file, append=TRUE)

output_path = sprintf("%s/scPred_assignment.pkl", path)
py_save_object(results, output_path)
