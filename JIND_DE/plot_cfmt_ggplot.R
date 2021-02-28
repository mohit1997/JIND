library(stringr)
library(reticulate)
library(limma)
library(pheatmap)
library(ggplot2)
library(gplots)
library(RColorBrewer)
library(geneplotter)
library(genefilter)
library(RColorBrewer)
library(lattice)
library(latticeExtra)
library(grid)
library(Rtsne)
library(xlsx)
library(gridExtra)
library(dendsort)
library(viridis)
library(grid)
library(reshape)
library(scales)


get_plot_dims <- function(heat_map)
{
  plot_height <- sum(sapply(heat_map$gtable$heights, grid::convertHeight, "in"))
  plot_width  <- sum(sapply(heat_map$gtable$widths, grid::convertWidth, "in"))
  dev.off()
  return(list(height = plot_height, width = plot_width))
}


process_CM <- function(data){
  b <- apply(data,2, FUN=function(x) (x/sum(x)))
  b[is.nan(b)] <- NA
  b[b==0] <- NA
  b <- b[,colSums(is.na(b))<nrow(b)]
  b <- b[rowSums(is.na(b))<ncol(b),]
  return(b)
}

mean_acc_pctype <- function(data_mat){
  return(round(mean(diag(data_mat[!grepl('Unassigned', rownames(data_mat)), ])),3))
}

mean_acc <- function(pred, labs){
  pred = as.character(pred)
  labs = as.character(labs)
  preds_filt = pred[pred != "Unassigned"]
  labs_filt = labs[pred != "Unassigned"]
  acc = mean(preds_filt == labs_filt)
  return(round(acc,3))
}

create_cm <- function(mat, title){
  a1 = rownames(mat)[!(rownames(mat) %in% c("Unassigned"))]
  a2 = rownames(mat)[(rownames(mat) %in% c("Unassigned"))]
  lst = c(a1, a2)
  mat = mat[lst,]
  ncells = ncol(mat)
  cm = data.frame(melt(mat))
  colnames(cm) = c("X1", "X2", "Acc")
  fontsize = 8
  fontsize_number = 8
  fontsizetitle = 8
  fontsizetitle = ncells  * fontsizetitle / 4
  
  cm$X1 = factor(cm$X1, level = lst) # Put Unassigned label at the end by specifying the order
  
  cmplot <- ggplot(cm, aes(x = X2, y = X1, fill=Acc)) +
    # geom_raster(aes(fill=Acc)) +
    geom_tile(colour="white",size=0.25)+
    geom_text(aes(label = sprintf("%.2f", Acc)), vjust = 0.5, size= min(3 / 8 * ncells, 6)) +
    # scale_fill_gradient(low="red", high="green") +
    scale_fill_gradient2(low = "#cb5b4c",
                         mid = "white",
                         high = '#1aab2d',
                         midpoint = 0.5,
                         space = "Lab",
                         na.value = "grey50",
                         guide = "colourbar",
                         aesthetics = "fill",
                         breaks = c(seq(0,  1.0, length.out= 5)),
                         labels = c(seq(0,  1.0, length.out= 5)),
                         limits=c(0,1)
                         ) +
    # scale_fill_manual( values = color_red_green, breaks = myBreaks) +
    # scale_y_discrete(expand=c(0,0))+
    # scale_x_discrete(expand=c(0,0))+
    labs(x="True Labels", y="Predicted Labels") +
    ggtitle(title) +
    xlim((levels(cm$X2))) +
    ylim(rev(levels(cm$X1))) +
    theme(axis.text.x=element_text(size=12, angle=45, colour = "black", vjust=1.0, hjust=1.0),
          axis.text.y=element_text(size=12, angle=0, colour = "black", hjust=1.0),
          legend.text=element_text(face="bold", size=fontsizetitle * 0.7),
          axis.title.x = element_text(size=fontsizetitle - 1),
          axis.title.y = element_text(size=fontsizetitle - 1),
          plot.title=element_text(size=fontsizetitle, hjust = 0.5),
          plot.background=element_blank(),
          panel.border=element_blank(),
          plot.margin = unit(c(4,4,4,4), "mm")
          )
  
  panel_height = unit(0.5,"npc") - sum(ggplotGrob(cmplot)[["heights"]][-3]) - unit(1,"line")
  cmplot <- cmplot + guides(fill= guide_colorbar(barheight=panel_height))
  # 
  return(cmplot)
}



pd <- import("pandas")
myBreaks <- c(seq(0,  0.2, length.out= 20), seq(0.21, 0.79, length.out=10), seq(0.8, 1, length.out=20))
color_red_green <- colorRampPalette(c('#4E62CC','#D8DBE2' , '#BA463E'))(50)
color_red_green <- colorRampPalette(c('#cb5b4c','#D8DBE2', '#1aab2d'))(50)


draw_cfmt <- function(dataSet, path = NULL, out_path = NULL){
  dir.create(out_path, showWarnings = FALSE)
  print(dataSet)
  switch(dataSet,
         human_dataset_random = {
           dataSet_name = 'Human Hematopoiesis'},
         mouse_atlas_random = {
           dataSet_name = 'Mouse Atlas'},
         mouse_dataset_random = {
           dataSet_name = 'Mouse Cortex'},
         pancreas_01 = {
           dataSet_name = 'Pancreas Bar16-Mur16'},
         pancreas_raw_01 = {
           dataSet_name = 'Pancreas Bar16-Mur16'},
         pancreas_02 = {
           dataSet_name = 'Pancreas Bar16-Seg16'},
         pancreas_raw_02 = {
           dataSet_name = 'Pancreas Bar16-Seg16'},
         human_blood_01 = {
           dataSet_name = 'PBMC 10x_v3-10x_v5'},
         pancreas_abcdnovel_01 = {
           dataSet_name = 'Pancreas Bar16-Mur16 (Novel Cell-type)'},
         stop("Does Not Exist!")
  )
  
  annotation <- pd$read_pickle(file.path(path, dataSet, 'JIND_raw_0', 'JIND_assignmentbrftune.pkl'))
  annotation$cell_names <- rownames(annotation)
  
  annotation_seurat <- pd$read_pickle(file.path(path, dataSet, 'seurat', 'seurat_assignment.pkl'))
  annotation_seurat$cell_names <- rownames(annotation_seurat)
  
  annotation_itcluster <- pd$read_pickle(file.path(path, dataSet, 'ItClusterFinal_0', 'ItCluster_assignment.pkl'))
  annotation_itcluster$cell_names <- rownames(annotation_seurat)
  
  file_xlsx <- file.path(out_path, paste0(dataSet, '_CM.xlsx' ))
  if (file.exists(file_xlsx)) {
    file.remove(file_xlsx)
  }
  
  data <- as.data.frame.matrix(table(annotation$raw_predictions, as.character(annotation$labels)))
  macc = mean_acc(annotation$raw_predictions, annotation$labels)
  b <- process_CM(data)
  b[is.na(b)] <- 0
  write.xlsx(b, file=file_xlsx, sheetName='JIND_raw', row.names = TRUE, append=TRUE)
  
  name = paste0('JIND+ (raw) ', dataSet_name, '\n Eff. Accuracy: ', format(round(macc, 3), nsmall = 3))
  jind_raw <- create_cm(b, name)
  
  data <- as.data.frame.matrix(table(annotation$predictions, as.character(annotation$labels)))
  macc = mean_acc(annotation$predictions, annotation$labels)
  b <- process_CM(data)
  b[is.na(b)] <- 0
  write.xlsx(b, file=file_xlsx, sheetName='JIND', row.names = TRUE, append=TRUE)
  
  name = paste0('JIND+ ', dataSet_name, '\n Eff. Accuracy: ', format(round(macc, 3), nsmall = 3))
  jind<- create_cm(b, name)
  
  data_seurat <- as.data.frame.matrix(table(annotation_seurat$raw_predictions, as.character(annotation$labels)))
  macc = mean_acc(annotation_seurat$raw_predictions, annotation_seurat$labels)
  b_seurat <- process_CM(data_seurat)
  b_seurat[is.na(b_seurat)] <-0
  write.xlsx(b_seurat, file=file_xlsx, sheetName='seurat', row.names = TRUE, append=TRUE)
  
  name = paste0('Seurat-LT ', dataSet_name, '\n Eff. Accuracy: ', format(round(macc, 3), nsmall = 3))
  seurat_raw <- create_cm(b_seurat, name)
  
  data_seurat <- as.data.frame.matrix(table(annotation_seurat$predictions, as.character(annotation$labels)))
  macc = mean_acc(annotation_seurat$predictions, annotation_seurat$labels)
  b_seurat <- process_CM(data_seurat)
  b_seurat[is.na(b_seurat)] <-0
  write.xlsx(b_seurat, file=file_xlsx, sheetName='seurat_raw', row.names = TRUE, append=TRUE)
  
  name = paste0('Seurat-LT (rej)', dataSet_name, '\n Eff. Accuracy: ', format(round(macc, 3), nsmall = 3))
  seurat <- create_cm(b_seurat, name)
  
  
  data_itcluster <- as.data.frame.matrix(table(annotation_itcluster$predictions, as.character(annotation_itcluster$labels)))
  macc = mean_acc(annotation_itcluster$predictions, annotation_itcluster$labels)
  b_itcluster <- process_CM(data_itcluster)
  b_itcluster[is.na(b_itcluster)] <-0
  write.xlsx(b_seurat, file=file_xlsx, sheetName='itcluster', row.names = TRUE, append=TRUE)
  
  name = paste0('ItCluster ', dataSet_name, '\n Eff. Accuracy: ', format(round(macc, 3), nsmall = 3))
  itcluster <- create_cm(b_itcluster, name)
  
  ncells = ncol(b)
  plotsize = ncells * 0.6
  
  pdf(file.path(out_path, paste0(dataSet, '_CM.pdf')), family="Times", height = plotsize * 2.1, width = plotsize * 2.3)
  grid.arrange(grobs = list(jind, jind_raw, seurat_raw, seurat), ncol=2)
  dev.off()
  
  pdf(file.path(out_path, paste0(dataSet, '_CMItCluster.pdf')), family="Times", height = plotsize * 2.1, width = plotsize * 2.3)
  grid.arrange(grobs = list(jind, jind_raw, seurat_raw, itcluster), ncol=2)
  dev.off()
  
  pdf(file.path(out_path, paste0(dataSet, '_CM_onlySJ.pdf')), family="Times", height = plotsize * 2.1, width = plotsize * 2.3)
  grid.arrange(grobs = list(jind_raw, jind, seurat_raw), ncol=2, nrow =2)
  dev.off()
  
}


path = "/home/mohit/mohit/seq-rna/Comparison/datasets"
out_path = "/home/mohit/mohit/seq-rna/Comparison/JIND_DE/Plots/MohitPlotsNew"

draw_cfmt("pancreas_raw_01", path = path, out_path = out_path)
draw_cfmt("pancreas_raw_02", path = path, out_path = out_path)
draw_cfmt("human_blood_01", path = path, out_path = out_path)
draw_cfmt("human_dataset_random", path = path, out_path = out_path)
draw_cfmt("mouse_atlas_random", path = path, out_path = out_path)
draw_cfmt("mouse_dataset_random", path = path, out_path = out_path)
# draw_cfmt("pancreas_abcdnovel_01", path = path, out_path = out_path)