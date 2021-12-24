if(!require("argparse")){
  install.packages("argparse")
}
library(reticulate)

check.integer <- function(x) {
  mean(x == round(x))
}

use_condaenv("jind")
py_config()

parser <- ArgumentParser(description='Run Seurat Classifier')
parser$add_argument('--train_path', default="./data/train.pkl", type="character",
                    help='path to train data frame with labels')
parser$add_argument('--test_path', default="./data/test.pkl", type="character",
                    help='path to test data frame with labels')
parser$add_argument('--column', type="character", default='labels',
                    help='column name for cell types')

args <- parser$parse_args()

jind <- import("jind")

pd <- import("pandas")
lname = args$column

batch1 = pd$read_pickle(args$train_path)

batch2 = pd$read_pickle(args$test_path)

mat1 = as.data.frame(batch1[,!(names(batch1) %in% c(lname))])
metadata1 = batch1[lname]
colnames(metadata1) <- c("labels")
# metadata1 = as.data.frame(metadata1)

mat2 = batch2[,!(names(batch2) %in% c(lname))]
metadata2 = batch2[lname]
colnames(metadata2) <- c("labels")
# metadata2 = as.data.frame(metadata2)

obj = jind$JindLib(mat1, as.list(metadata1$labels), path="my_results")

isint = check.integer(mat1[1:100, 1:100]) == 1.

if (isint == TRUE){
  obj$preprocess(count_normalize=TRUE, logt=TRUE)
}

obj$dim_reduction(5000L, 'Var')

train_config = list('val_frac'= 0.2, 'seed' = 0L, 'batch_size'=128L, 'cuda'= FALSE, 'epochs'=15L)
obj$train_classifier(config=train_config, cmat=TRUE)
predicted_label  = obj$get_filtered_prediction(mat2, frac=0.05, test=FALSE)
# gdecay: Generator weight decay
# ddecay: Discriminator weight decay
# maxcount: Number of total epochs  where the Generator Loss and Discrimiantor Loss is less than 0.78
train_config = list('gdecay'= 0.01, 'seed' = 0L, 'batch_size'=128L, 'cuda'= FALSE, 'epochs'=15L, 'ddecay'=0.001, 'maxcount'= 7L)
obj$remove_effect(mat1, mat2, train_config)
predicted_label  = obj$evaluate(mat2, as.list(metadata2$labels), frac=0.05, name="testcfmtbr.pdf", test=TRUE)

train_config = list('val_frac'=0.1, 'seed'=0L, 'batch_size'=32L, 'cuda'=FALSE, 'epochs'=10L)
obj$ftune_top(mat2, train_config)
predicted_label  = obj$evaluate(mat2, as.list(metadata2$labels), frac=0.05, name="testcfmtbrft.pdf", test=TRUE)