if(!require("argparse")){
  install.packages("argparse")
}
library("argparse")

if (!requireNamespace("devtools", quietly = TRUE)) install.packages("devtools")
if(!require(scibet)){
  devtools::install_github("PaulingLiu/scibet", force = TRUE)
}

library(scibet)
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

parser <- ArgumentParser(description='Run SciBet Classifier')
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
mat1 = exp(mat1) - 1
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
mat2 = exp(mat2) - 1
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

colnames(metadata1) <- c("label")
colnames(metadata2) <- c("label")

train_set = cbind(mat1, metadata1)
test_set = cbind(mat2, metadata2)
train_set['label'] = lapply(train_set['label'], as.character)
test_set['label'] = lapply(test_set['label'], as.character)

prd <- SciBet(train_set, test_set)
prd = as.matrix(prd)
colnames(prd) = c("predictions")
rownames(prd) = rownames(test_set)

index = prd == metadata2
acc = mean(index)
sprintf("Test Accuracy %f", acc)

prd = as.data.frame(prd)
prd = cbind(prd, metadata2)

pkl <- import("pickle")
path = sprintf("%s/scibet", dirname(args$train_path))
dir.create(path, showWarnings = FALSE)

file = sprintf("%s/test.log", path)
end_time <- Sys.time()
cat(sprintf("Test Accuracy %f", acc), file = file)
cat(capture.output(end_time - start_time), file=file, append=TRUE)

output_path = sprintf("%s/scibet_matrix.pkl", path)
py_save_object(prd, output_path)

# path_da <- "data/test.rds.gz"
# expr <- readr::read_rds(path = path_da)
# 
# tibble(
#   ID = 1:nrow(expr),
#   label = expr$label
# ) %>%
#   dplyr::sample_frac(0.7) %>%
#   dplyr::pull(ID) -> ID
# 
# train_set <- expr[ID,]      #construct reference set
# test_set <- expr[-ID,]      #construct query set
# 
# prd <- SciBet(train_set, test_set)
# mean(prd == test_set$label)
