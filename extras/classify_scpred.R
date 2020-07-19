if(!require("argparse")){
  install.packages("argparse")
}
library("argparse")
library("scPred")
library("tidyverse")
library('Seurat')
library(caret)
if(!require(reshape)){
  install.packages("reshape")
}
library(reshape)
library(reticulate)
use_virtualenv("~/mohit/torch-cpu", required = TRUE)
py_config()

parser <- ArgumentParser(description='Process some integers')
parser$add_argument('--train_path', default="datasets/human_blood_integrated_01/train.pkl", type="character",
                    help='path to train data frame with labels')
parser$add_argument('--test_path', default="datasets/human_blood_integrated_01/test.pkl", type="character",
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
    message(paste0(sum(zeroVar), " following genes were removed as their variance is zero across all cells:"))
    cat(paste0(names(zeroVar), collapse = "\n"), "\n", sep = "")
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

scp1 <- scPredict(scp, newData = test_data, threshold = 0.0)
scp1@predMeta <- metadata2
out = crossTab(scp1, true = "labels")
out = as.matrix(out)
out <- out[, order(as.numeric(as.factor(colnames(out))))]
out <- out[order(as.numeric(as.factor(rownames(out)))),]
pred = getPredictions(scp1)
true = metadata2[,'labels']

pred1 = cbind(pred, metadata2)


comparison = metadata2['labels'] == pred['predClass']
raw = mean(comparison)

scp2 <- scPredict(scp, newData = test_data, threshold = 0.9)
scp2@predMeta <- metadata2
out = crossTab(scp2, true = "labels")
out = as.matrix(out)
out <- out[, order(as.numeric(as.factor(colnames(out))))]
out <- out[order(as.numeric(as.factor(rownames(out)))),]
pred = getPredictions(scp2)
true = metadata2[,'labels']

pred2 = cbind(pred, metadata2)
index = pred['predClass'] != "unassigned"

comparison = metadata2[index, 'labels'] == pred[index, 'predClass']

eff = mean(comparison)
filtered = 1 - mean(index)

pkl <- import("pickle")
path = sprintf("%s/scPred", dirname(args$train_path))
dir.create(path, showWarnings = FALSE)

file = sprintf("%s/test.log", path)
cat(sprintf("Test Accuracy %f w.f. %f filtered %f", raw, eff, filtered), file = file)

output_path = sprintf("%s/scPred_matrix.pkl", path)
py_save_object(pred1, output_path)