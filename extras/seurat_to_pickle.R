library(Seurat)
library(SeuratData)
library(reticulate)
use_virtualenv("~/mohit/torch-cpu", required = TRUE)

InstallData("panc8")
pd <- import("pandas")


data("panc8")
pancreas.list <- SplitObject(panc8, split.by = "tech")

# pancreas.list <- pancreas.list[c("celseq", "celseq2", "fluidigmc1", "smartseq2")]
source <- pancreas.list$celseq
source_mat = as.data.frame(t(as.matrix(source@assays$RNA@counts)))
cell_labels = source$celltype
source_mat['labels'] = cell_labels

target <- pancreas.list$celseq2
target_mat = as.data.frame(t(as.matrix(target@assays$RNA@counts)))
cell_labels = target$celltype
target_mat['labels'] = cell_labels

path = "/home/mohit/mohit/seurat_to_pkl"
dir.create(path)
output_path = sprintf("%s/train.pkl", path)
py_save_object(as.data.frame(source_mat), output_path)

output_path = sprintf("%s/test.pkl", path)
py_save_object(as.data.frame(target_mat), output_path)

system("source ~/mohit/torch-cpu/bin/activate")
system("python classify_JIND_rawdata_top.py --train_path /home/mohit/mohit/seurat_to_pkl/train.pkl --test_path /home/mohit/mohit/seurat_to_pkl/test.pkl --column labels")

preds = pd$read_pickle(sprintf("%s/JIND_rawtop_0/JIND_assignmentbrftune.pkl", path))