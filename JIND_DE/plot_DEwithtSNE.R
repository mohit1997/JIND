
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
library(wesanderson)

callback = function(hc, mat)
{
  dendsort(hc, isReverse=T)
  
}


"graphContrast" <- function(data, name, Bth, FCth, namecol) {
  hist(data$P.Value, main = name, xlab = "p-value", ylab = "Genes");
  hist(data$logFC, main = name, xlab="Log2(FoldChange)", ylab="Genes", 50);
  #hist(treatm_vs_ctrl$B, main = name, xlab="B", ylab="Genes", 50);
  volcanoCol(data, Bth, FCth, name, namecol)
}

"volcanoCol" <- function(res, Bth, FCth, title, namecol) {
  colVP <- rep("black", length(res$B))
  colVP[which(res$B>Bth & res$logFC>FCth)] <- "red"
  colVP[which(res$B>Bth & res$logFC<(FCth*(-1)))] <- "green"
  plot(res$logFC, (-1)*log(res$P.Value), pch = ".", col = colVP, main = title, xlab = "foldchange", ylab = "-log(pvalue)")
  abline(v = FCth)
  abline(v = (FCth*(-1)))
  abline(h = (-1)*log(max(res$P.Value[res$B>Bth])))
  selFC <- res$logFC[which(res$B>Bth & abs(res$logFC)>FCth)]
  colVP2 <- rep("red", length(selFC))
  colVP2[which(selFC<((-1)*FCth))] <- "green"
  if (length(res[which(res$B>Bth & abs(res$logFC)>FCth),namecol])>0)
    text(res$logFC[which(res$B>Bth & abs(res$logFC)>FCth)], (-1)*log(res$P.Value[which(res$B>Bth & abs(res$logFC)>FCth)]), res[which(res$B>Bth & abs(res$logFC)>FCth),namecol], pos = 3, cex=0.7, offset = 0.5, col = colVP2)
}

get_plot_dims <- function(heat_map)
{
  plot_height <- sum(sapply(heat_map$gtable$heights, grid::convertHeight, "in"))
  plot_width  <- sum(sapply(heat_map$gtable$widths, grid::convertWidth, "in"))
  dev.off()
  return(list(height = plot_height, width = plot_width))
}

get_topk_features <- function(data, k = 5000){
  print(paste0("Reudcing to top", k, " features"))
  var_ <- apply(data, 1, var)
  names(var_) <- rownames(data)
  genes_2_keep <- names(sort(var_, decreasing=T)[1:5000])
  red_data <- data[rownames(data) %in% genes_2_keep, ]
  return(red_data)
}

perform_DE <- function(data, group){
  ### DE
  # create the linear model
  fit_tmp <- lmFit(data, group)
  # model correction
  fit_tmp <- eBayes(fit_tmp)
  # results <- topTable(fit_tmp, n=Inf)
  x <- paste0('G1', '-', 'G2')
  contrast_mat_tmp <- makeContrasts(contrasts=x, levels= c('G1', 'G2'))
  fit2_tmp <- contrasts.fit(fit_tmp, contrast_mat_tmp)
  fit2_tmp <- eBayes(fit2_tmp)
  tmp   <- topTable(fit2_tmp, adjust="fdr", n=Inf)
  tmp$gene_name <- rownames(tmp)
  
  tmp[tmp$P.Value == 0, 'P.Value'] <- 1.445749e-281
  tmp[tmp$adj.P.Val == 0, 'adj.P.Val'] <- 1.445749e-281
  
  return(tmp)
}


process_CM <- function(data){
  b <- apply(data,2, FUN=function(x) (x/sum(x)))
  b[is.nan(b)] <- NA
  b[b==0] <- NA
  b <- b[,colSums(is.na(b))<nrow(b)]
  b <- b[rowSums(is.na(b))<ncol(b),]
  return(b)
}

mean_acc <- function(data_mat){
  return(round(mean(diag(data_mat[!grepl('Unassigned', rownames(data_mat)), ])),3))
}


write_excel <- function(frame, sheetname, file){
  if (nrow(frame) != 0){
    write.xlsx(frame, file=file, sheetName=sheetname, row.names = FALSE, append=TRUE)
  }
  else {
    dataframeempty <- t(as.data.frame(rep('NA', ncol(frame))))
    colnames(dataframeempty) = colnames(frame)
    rownames(dataframeempty) = c("NA")
    write.xlsx(dataframeempty, file=file, sheetName=sheetname, row.names = FALSE, append=TRUE, showNA = TRUE)
  }
}



pd <- import("pandas")

color_red_green <- colorRampPalette(c('#4E62CC','#D8DBE2' , '#BA463E'))(50)
color_red_green <- colorRampPalette(c('#cb5b4c','#D8DBE2', '#1aab2d'))(50)
myBreaks <- c(seq(0,  0.4, length.out= 20), seq(0.41, 0.79, length.out=10), seq(0.8, 1, length.out=20))

DE_train <- function(dataSet, target, obj, genes_displ, plot_selected_genes = NULL,data_path = data_path, plots_path = plots_path){
  dir.create(plots_path, showWarnings = FALSE)
  print(dataSet)
  switch(dataSet,
         human_dataset_random = {
           dataSet_name = 'Human Hematopoiesis'},
         mouse_atlas_random = {
           dataSet_name = 'Mouse Atlas'},
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
         stop("Does Not Exist!")
  )
  
  
  df  <- pd$read_pickle(file.path(data_path, dataSet, 'train.pkl'))
  annotation <- data.frame(cell_names = rownames(df), labels = df$labels, stringsAsFactors = F)
  rownames(annotation) <- annotation$cell_names
  
  all_data <- t(df[, -which(colnames(df) %in% c('labels'))])
  
  stand_dev <- apply(all_data,1, sd)
  all_data <- all_data[!rownames(all_data) %in% names(which(stand_dev ==0)), ]
  
  # tsne_out <- Rtsne(as.matrix(t(all_data)))
  # plotter <-  as.data.frame(tsne_out$Y)
  
  annotation$labels <- gsub(' ', '_', annotation$labels)
  
  G1 <- annotation[annotation$labels == target , 'cell_names']
  G2 <- annotation[ annotation$labels == obj, 'cell_names']
  
  selection <- c(G1, G2)
  
  data_tmp <- all_data[ , colnames(all_data) %in%  selection]
  stand_dev <- apply(data_tmp,1, sd)
  data_tmp <- data_tmp[!rownames(data_tmp) %in% names(which(stand_dev ==0)), ]
  
  
  DESIGN <- data.frame(cells = colnames(data_tmp))
  labels <- c()
  for (cell in DESIGN$cells){
    ifelse(cell %in% G1, labels <- c(labels, 'G1'), labels <- c(labels, 'G2'))
  }
  DESIGN$labels <- labels
  
  design_tmp <- as.matrix(DESIGN[, 'labels'])
  design_tmp[design_tmp != 'G1'] <-1
  design_tmp[design_tmp == 'G1'] <-0
  design_tmp <- model.matrix(~0+as.factor(design_tmp[,1]))
  colnames(design_tmp) <- c('G1', 'G2')
  rownames(design_tmp) <- colnames(data_tmp)
  
  
  ### DE
  tmp <- perform_DE(data_tmp, design_tmp)
  
  tmp$logpval <- -log(tmp$P.Value)
  tmp2 <- tmp[ order(-tmp$logpval), c('gene_name', 'logFC', 'P.Value' ,'logpval', 'adj.P.Val')]
  
  file_xlsx <- file.path(plots_path, paste0(dataSet,'_',target,'Vs',obj,'_DEtrain.xlsx'))
  if (file.exists(file_xlsx)) {
    file.remove(file_xlsx)
  }
  
  res = data.frame(
    Column = c("gene_name","logFC","P.Value","logpval","adj.P.Val"),
    Meaning = c("Name of the gene",
                "log of Fold Change",
                "P-value output by package Limma",
                "-log(P-value)",
                "FDR Adjusted P-value")
  )
  
  write_excel(res, "Results", file_xlsx)
  
  write_excel(tmp2, 'Cell Annotations', file_xlsx)
  write_excel(tmp2[tmp2$adj.P.Val<0.05,], 'Cell Annotations (FDR < 0.05)', file_xlsx)
  
  # pdf('./Plots/12_CD14.Mono.2_VP.pdf')
  # 	graphContrast(tmp," (Ctrl B > 0, FC>0.5)", 0, 0.5, 1)
  # dev.off()
  
  data2heat <- data_tmp[tmp2[tmp2$adj.P.Val<0.05,'gene_name'],]
  # data2heat <- data_tmp[rownames(data_tmp) %in%  tmp[tmp$adj.P.Val<0.05 & abs(tmp$logFC)>0.5,'gene_name'],]
  data2heat[data2heat > 5] <- 5
  
  
  colors = c(rgb(31, 119, 180, max = 255), rgb(255, 127, 14, max =255),
             rgb(44, 160, 44, max = 255), rgb(214, 39, 40, max =255),
             rgb(148, 103, 189, max = 255), rgb(140, 86, 75, max =255),
             rgb(227, 119, 194, max = 255), rgb(127, 127, 127, max =255),
             rgb(188, 189, 34, max = 255), rgb(23, 190, 207, max =255))
  
  
  ann <- data.frame(Group = annotation[rownames(annotation) %in% colnames(data2heat), 'labels'])
  ann$Group <-  ifelse(ann$Group == target, 'G1', 'G2')
  rownames(ann) <- colnames(data2heat)
  Group <- colors[1:length(unique(annotation$labels))]
  names(Group) <-names(sort(table(as.character(annotation$labels) ), decreasing=T))
  # names(Group)  <- c(unique(ann$Group))
  anno_colors <- c(Group[target],Group[obj])
  names(anno_colors) <- c('G1', 'G2')
  # anno_colors   <- list(anno_colors = anno_colors)
  anno_colors <- list(Group = anno_colors)
  HM <- pheatmap( data2heat, cluster_rows = F, treeheight_row = 0, annotation_col = ann,annotation_colors = anno_colors, clustering_distance_rows = "euclidean", clustering_distance_cols = "euclidean", cellheight= 4,cellwidth= 2, show_colnames = F, main = paste0('Heatmap between ', target,' (G1)\n and ',obj,' (G2)'), fontsize = 8,fontsize_row=8, callback = callback)
  plot_dims <- get_plot_dims(HM)
  
  pdf(file.path(plots_path, paste0(dataSet,'_',target,'Vs',obj,'_HM_Seltrain.pdf')), family="Times", height = plot_dims$height, width = plot_dims$width)
  print(HM)
  dev.off()
  
  switch(dataSet,
         pancreas_01 = { golden_boys <- c('KRT19', 'PDX1', 'SOX9', 'UEA1', 'GP2', 'CD142', 'PRSS1', 'CTRC', 'CPA1', 'AMY2A', 'SYCN', 'RBPJL', 'MIST1', 'HNF1B', 'PTF1A', 'CA19.9', 'PARM1', 'GP2', 'CD142', 'RBPJ', 'MYC')},
         human_blood_01        = { golden_boys <- c('CD14', 'FCGR3A')}
  )
  golden_present <- tmp2[tmp2$gene_name %in% golden_boys,'gene_name']
  
  data2heat_small <- data2heat
  krtdata <- as.data.frame(data2heat_small[rownames(data2heat_small) %in% golden_present,])
  # rownames(krtdata) <- 'KRT19'
  data2heat_small <- data2heat_small[!rownames(data2heat_small) %in% golden_present,]
  data2heat_small <- rbind(krtdata, data2heat_small)
  ann$Cluster <- NULL
  
  cell_width = 1
  HM <- pheatmap( data2heat_small[1:genes_displ,],cluster_cols = HM$tree_col, cluster_rows = F, treeheight_row = 0, annotation_col = ann, clustering_distance_rows = "euclidean", clustering_distance_cols = "euclidean", cellwidth= cell_width, annotation_colors = anno_colors, show_colnames = F, cellheight= 7, main = paste0('Heatmap between ', target,' (G1)\n and ',obj,' (G2)'), fontsize = 8,fontsize_row=8, callback = callback)
  plot_dims <- get_plot_dims(HM)
  pdf(file.path(plots_path, paste0(dataSet,'_',target,'Vs',obj,'_HM_Seltrainsmall.pdf')), family="Times", height = plot_dims$height, width = plot_dims$width)
  # print(HM$tree_col)
  print(HM)
  dev.off()
}

data_path = "/home/mohit/mohit/seq-rna/Comparison/datasets"
plots_path = "/home/mohit/mohit/seq-rna/Comparison/JIND_DE/Plots/MohitPlotsDE"

a = DE_train('pancreas_01', 'ductal', 'acinar', 25, data_path = data_path, plots_path = plots_path)
a = DE_train('human_blood_01', 'Monocyte_FCGR3A', 'Monocyte_CD14', 25, data_path = data_path, plots_path = plots_path)


DE_with_TSNE <- function(dataSet, target, obj, genes_displ, plot_selected_genes = NULL,data_path = data_path, plots_path = plots_path, plottSNE=TRUE){
  dir.create(plots_path, showWarnings = FALSE)
  print(dataSet)
  switch(dataSet,
         human_dataset_random = {
           dataSet_name = 'Human Hematopoiesis'},
         mouse_atlas_random = {
           dataSet_name = 'Mouse Atlas'},
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
         stop("Does Not Exist!")
  )
  
  df  <- pd$read_pickle(file.path(data_path, dataSet, 'test.pkl'))
  annotation <- pd$read_pickle(file.path(data_path, dataSet, 'JIND_raw_0', 'JIND_assignmentbrftune.pkl'))
  annotation$cell_names <- rownames(annotation)
  
  all_data <- t(df[, -which(colnames(df) %in% c('labels'))])
  
  var_5k=FALSE
  if (var_5k==TRUE & !dataSet %in% c('human_blood_01', 'pancreas_01', 'pancreas_02')){
    all_data <- get_topk_features(all_data, k = 5000)
  }
  
  stand_dev <- apply(all_data,1, sd)
  all_data <- all_data[!rownames(all_data) %in% names(which(stand_dev ==0)), ]
  
  annotation$labels <- gsub(' ', '_', annotation$labels)
  annotation$predictions <- gsub(' ', '_', annotation$predictions)
  annotation$raw_predictions <- gsub(' ', '_', annotation$raw_predictions)
  
  G1 <- annotation[annotation$labels == target & annotation$prediction == target, 'cell_names']
  G2 <- annotation[annotation$labels == target & annotation$prediction == obj, 'cell_names']
  
  selection <- c(G1, G2)
  
  data_tmp <- all_data[ , colnames(all_data) %in%  selection]
  stand_dev <- apply(data_tmp,1, sd)
  data_tmp <- data_tmp[!rownames(data_tmp) %in% names(which(stand_dev ==0)), ]
  
  DESIGN <- data.frame(cells = colnames(data_tmp))
  labels <- c()
  for (cell in DESIGN$cells){
    ifelse(cell %in% G1, labels <- c(labels, 'G1'), labels <- c(labels, 'G2'))
  }
  DESIGN$labels <- labels
  
  design_tmp <- as.matrix(DESIGN[, 'labels'])
  design_tmp[design_tmp != 'G1'] <-1
  design_tmp[design_tmp == 'G1'] <-0
  design_tmp <- model.matrix(~0+as.factor(design_tmp[,1]))
  colnames(design_tmp) <- c('G1', 'G2')
  rownames(design_tmp) <- colnames(data_tmp)
  
  tmp <- perform_DE(data_tmp, design_tmp)
  
  # pdf('./Plots/12_CD14.Mono.2_VP.pdf')
  # 	graphContrast(tmp," (Ctrl B > 0, FC>0.5)", 0, 0.5, 1)
  # dev.off()
  
  tmp$logpval <- -log(tmp$P.Value)
  tmp2 <- tmp[ order(-tmp$logpval), c('gene_name', 'logFC', 'P.Value' ,'logpval', 'adj.P.Val')]
  
  file_xlsx <- file.path(plots_path, paste0(dataSet,'_',target,'Vs',obj,'_DE.xlsx'))
  if (file.exists(file_xlsx)) {
    file.remove(file_xlsx)
  }
  
  res = data.frame(
    Column = c("gene_name","logFC","P.Value","logpval","adj.P.Val"),
    Meaning = c("Name of the gene",
                "log of Fold Change",
                "P-value output by package Limma",
                "-log(P-value)",
                "FDR Adjusted P-value")
  )
  
  write_excel(res, "Results", file_xlsx)
  
  write_excel(tmp2, 'JIND+', file_xlsx)
  write_excel(tmp2[tmp2$adj.P.Val<0.05,], 'JIND+ (FDR < 0.05)', file_xlsx)
  
  # print(rownames(data_tmp))
  # print(tmp2[tmp2$adj.P.Val<0.001,'gene_name'])
  data2heat <- data_tmp[tmp2[tmp2$adj.P.Val<0.05,'gene_name'],]
  # data2heat <- data_tmp[rownames(data_tmp) %in%  tmp2[tmp2$adj.P.Val<0.05,'gene_name'],]
  
  data2heat[data2heat > 5] <- 5
  
  colors = c(rgb(31, 119, 180, max = 255), rgb(255, 127, 14, max =255),
             rgb(44, 160, 44, max = 255), rgb(214, 39, 40, max =255),
             rgb(148, 103, 189, max = 255), rgb(140, 86, 75, max =255),
             rgb(227, 119, 194, max = 255), rgb(127, 127, 127, max =255),
             rgb(188, 189, 34, max = 255), rgb(23, 190, 207, max =255))
  
  
  ann <- data.frame(Group = annotation[rownames(annotation) %in% colnames(data2heat), 'predictions'])
  ann$Group <-  ifelse(ann$Group == target, 'G1', 'G2')
  rownames(ann) <- colnames(data2heat)
  Group <- colors[1:length(unique(annotation$labels))]
  names(Group) <-names(sort(table(as.character(annotation$labels) ), decreasing=T))
  anno_colors <- c(Group[target],Group[obj])
  names(anno_colors) <- c('G1', 'G2')
  
  
  HM <- pheatmap( data2heat, cluster_rows = T, treeheight_row = 0, annotation_col = ann, clustering_distance_rows = "euclidean", clustering_distance_cols = "euclidean", cellheight= 4,cellwidth= 2, show_colnames = F, main = paste0('Heatmap between ',target,' classified as ',target,' (G1)\n and ',target,' classified as ',obj,' (G2)'), fontsize = 8,fontsize_row=4, callback = callback)
  
  plot_dims <- get_plot_dims(HM)
  
  cluster_names <- HM$tree_col$labels
  cluster_pos <- HM$tree_col$order
  
  cluster_seq <- which(cluster_names[cluster_pos] %in% G2)
  cluster_list <- split(cluster_seq, cumsum(c(1, diff(cluster_seq) != 1)))
  
  counter <- 1
  shape <- data.frame(cell_id = colnames(data2heat), Shape = 'G1', stringsAsFactors=FALSE)
  for (cl in cluster_list){
    shape[shape$cell_id %in% cluster_names[cluster_pos][cl] , 'Shape'] <- paste0('G2_', counter)
    counter <- counter +1
  }
  
  ann <- data.frame(cell_id = colnames(data2heat), Group = annotation[rownames(annotation) %in% colnames(data2heat), 'predictions'])
  ann$Group <-  ifelse(ann$Group == target, 'G1', 'G2')
  ann <- merge(ann, shape, by='cell_id')
  ann$cell_id <- NULL
  names(ann) <- c('Group', 'Cluster')
  rownames(ann) <- colnames(data2heat)
  Cluster <- colors[which(!colors %in% c(anno_colors))][1:counter]
  names(Cluster) <- c(unique(ann$Cluster))
  
  Cluster_small <- setNames(c(Cluster), c(names(Cluster)))
  Cluster <- setNames(c(Cluster, rgb(127, 127, 127, alpha = 50, max=255)), c(names(Cluster), 'G0'))
  
  callback = function(hc, mat)
  {
    dendsort(hc, isReverse=T)
  }
  anno_colors2  <- list(Group = anno_colors, Cluster=Cluster)
  anno_colors2$Cluster['G1'] <- anno_colors['G1']
  
  anno_colors_small  <- list(Group = anno_colors, Cluster=Cluster_small)
  anno_colors_small$Cluster['G1'] <- anno_colors['G1']
  
  ann_temp <- ann
  ann_temp$Cluster <- NULL
  HM <- pheatmap( data2heat, cluster_rows = T, treeheight_row = 0, annotation_col = ann_temp, clustering_distance_rows = "euclidean", clustering_distance_cols = "euclidean", cellheight= 4,cellwidth= 2, annotation_colors = anno_colors2, show_colnames = F, main = paste0('Heatmap between ',target,' classified as ',target,' (G1)\n and ',target,' classified as ',obj,' (G2)'), fontsize = 8,fontsize_row=4, callback = callback)
  
  plot_dims <- get_plot_dims(HM)
  
  pdf(file.path(plots_path, paste0(dataSet,'_',target,'Vs',obj,'_HM_All.pdf')), family="Times", height = plot_dims$height, width = plot_dims$width)
  # print(HM$tree_col)
  print(HM)
  dev.off()
  
  ann_temp <- ann
  HM <- pheatmap( data2heat, cluster_rows = T, treeheight_row = 0, annotation_col = ann_temp, clustering_distance_rows = "euclidean", clustering_distance_cols = "euclidean", cellheight= 4,cellwidth= 2, annotation_colors = anno_colors_small, show_colnames = F, main = paste0('Heatmap between ',target,' classified as ',target,' (G1)\n and ',target,' classified as ',obj,' (G2)'), fontsize = 8,fontsize_row=4, callback = callback)
  # ann_temp$Cluster <- NULL
  plot_dims <- get_plot_dims(HM)
  
  pdf(file.path(plots_path, paste0(dataSet,'_',target,'Vs',obj,'_HM_All_clusters.pdf')), family="Times", height = plot_dims$height, width = plot_dims$width)
  # print(HM$tree_col)
  print(HM)
  dev.off()
  
  if(dataSet %in% c('pancreas_01', 'pancreas_raw_01', 'human_blood_01')){
    switch(dataSet,
           pancreas_01 = { golden_boys <- c('KRT19', 'PDX1', 'SOX9', 'UEA1', 'GP2', 'CD142', 'PRSS1', 'CTRC', 'CPA1', 'AMY2A', 'SYCN', 'RBPJL', 'MIST1', 'HNF1B', 'PTF1A', 'CA19.9', 'PARM1', 'GP2', 'CD142', 'RBPJ', 'MYC')},
           pancreas_raw_01 = { golden_boys <- c('KRT19', 'PDX1', 'SOX9', 'UEA1', 'GP2', 'CD142', 'PRSS1', 'CTRC', 'CPA1', 'AMY2A', 'SYCN', 'RBPJL', 'MIST1', 'HNF1B', 'PTF1A', 'CA19.9', 'PARM1', 'GP2', 'CD142', 'RBPJ', 'MYC')},
           human_blood_01        = { golden_boys <- c('CD14', 'FCGR3A')}
    )
    golden_present <- tmp2[tmp2$gene_name %in% golden_boys,'gene_name']
    
    data2heat_small <- data2heat
    krtdata <- as.data.frame(data2heat_small[rownames(data2heat_small) %in% golden_present,])
    # rownames(krtdata) <- 'KRT19'
    data2heat_small <- data2heat_small[!rownames(data2heat_small) %in% golden_present,]
    data2heat_small <- rbind(krtdata, data2heat_small)
    ann_tmp <- ann
    ann_tmp$Cluster <- NULL
    
    cell_width = 1
    HM <- pheatmap( data2heat_small[1:genes_displ,],cluster_cols = HM$tree_col, cluster_rows = F, treeheight_row = 0, annotation_col = ann_tmp, clustering_distance_rows = "euclidean", clustering_distance_cols = "euclidean", cellwidth= cell_width, annotation_colors = anno_colors2, show_colnames = F, cellheight= 7, main = paste0('Heatmap between ',target,' classified as ',target,' (G1)\n and ',target,' classified as ',obj,' (G2)'), fontsize = 8,fontsize_row=8, callback = callback)
    plot_dims <- get_plot_dims(HM)
    pdf(file.path(plots_path, paste0(dataSet,'_',target,'Vs',obj,'_HM_Sel.pdf')), family="Times", height = plot_dims$height, width = plot_dims$width)
    # print(HM$tree_col)
    print(HM)
    dev.off()
  }
  
  ##
  # Plot only the marker genes
  ##
  if (!is.null(plot_selected_genes)){
    data2heat_small <- data_tmp[rownames(data_tmp) %in% plot_selected_genes,]
    ann <- data.frame(cell_id = colnames(data2heat_small), Group = annotation[rownames(annotation) %in% colnames(data2heat_small), 'predictions'])
    ann$Group <-  ifelse(ann$Group == target, 'G1', 'G2')
    ann$cell_id <- NULL
    rownames(ann) <- colnames(data2heat_small)
    data2heat_small <- data2heat_small[,order(-data2heat_small[plot_selected_genes[1],])]
    HM_1 <- pheatmap( data2heat_small[plot_selected_genes[1],,drop=F], cluster_rows = F, cluster_cols = F, treeheight_row = 0, annotation_col = ann, clustering_distance_rows = "euclidean", clustering_distance_cols = "euclidean", cellheight= 4,cellwidth= 2, annotation_colors = anno_colors, show_colnames = F, main = paste0('Heatmap between ',target,' classified as ',target,' (G1)\n and ',target,' classified as ',obj,' (G2)'), fontsize = 8,fontsize_row=4)
    data2heat_small <- data2heat_small[,order(-data2heat_small[plot_selected_genes[2],])]
    HM_2 <- pheatmap( data2heat_small[plot_selected_genes[2],,drop=F], cluster_rows = F, cluster_cols = F, treeheight_row = 0, annotation_col = ann, clustering_distance_rows = "euclidean", clustering_distance_cols = "euclidean", cellheight= 4,cellwidth= 2, annotation_colors = anno_colors, show_colnames = F, main = paste0('Heatmap between ',target,' classified as ',target,' (G1)\n and ',target,' classified as ',obj,' (G2)'), fontsize = 8,fontsize_row=4)
    
    plot_dims <- get_plot_dims(HM_1)
    
    pdf(file.path(plots_path, paste0(dataSet,'_',target,'Vs',obj,'_HM_2genes.pdf')), family="Times", height = plot_dims$height*3, width = plot_dims$width)
    grid.arrange(grobs = list(HM_1[[4]], HM_2[[4]]), nrow=2)
    dev.off()
  }
  
  if (plottSNE){
    #############
    # TSNE
    #############
    G3 <- annotation[annotation$labels == target & annotation$prediction == target, 'cell_names']
    G4 <- annotation[annotation$labels == obj & annotation$prediction == obj, 'cell_names']
    
    all_data2 <- all_data[, colnames(all_data) %in%  c(G3, G4, G2)]
    # all_data2 <- all_data[, colnames(all_data) %in% annotation[annotation$labels %in% c(target, obj), 'cell_names']]
    
    # set.seed(123)
    tsne_out <- Rtsne(as.matrix(t(all_data2)))
    plotter <-  as.data.frame(tsne_out$Y)
    
    plotter$labels <- annotation[rownames(annotation) %in% colnames(all_data2), 'labels']
    plotter$predictions <- annotation[rownames(annotation) %in% colnames(all_data2), 'predictions']
    plotter$cell_id <- rownames(annotation[rownames(annotation) %in% colnames(all_data2),])
    
    labels <- data.frame(cell_id = colnames(all_data2) )
    labels$Cluster <- 'G0'
    labels[labels$cell_id %in% G1,'Cluster'] <- 'G1'
    counter <- 1
    for (cl in cluster_list){
      labels[labels$cell_id %in% cluster_names[cluster_pos][cl], 'Cluster'] <- paste0('G2_', counter)
      counter <- counter +1
    }
    
    plotter <- merge(plotter, labels, by ='cell_id')
    
    
    euc.dist <- function(x1, x2) sqrt(sum((x1 - x2) ^ 2))
    
    ratio_dist <- function(a, b, c) {
      return(as.character(euc.dist(unlist(a),unlist(b))/euc.dist(unlist(a),unlist(c))))
    }
    
    mean_dist <- list()
    for (gr in unique(plotter$Cluster)){
      mean_dist[[gr]] <- c(c(mean(plotter[plotter$Cluster %in% c(gr), 'V1']), mean(plotter[plotter$Cluster %in% c(gr), 'V2'])))
    }
    
    title <- c()
    for (i in seq_along(mean_dist)){
      tmp <- paste0(names(mean_dist[i]), ' ratio dist.  ', paste(ratio_dist(mean_dist[i],  mean_dist['G0'], mean_dist['G1'])))
      title <- paste(title, tmp, sep=' \n')
    }
    title <- paste(title, paste0('Ratio:  ',obj,'/',target,''), sep='\n')
    
    pdf(file.path(plots_path, paste0(dataSet,'_',target,'Vs',obj,'_TSNE.pdf')), family="Times", 15,15)
    print(ggplot(plotter, aes(x=V1, y=V2, color = Cluster, label = labels)) + theme_classic() + geom_text()  +scale_color_manual(values = anno_colors2$Cluster) + ggtitle(title)+ theme(legend.text=element_text(size=15)))
    print(ggplot(plotter, aes(x=V1, y=V2, color = Cluster)) + theme_classic() + geom_point(alpha = 0.8,size = 8)  +scale_color_manual(values = anno_colors2$Cluster) + ggtitle(title)  + 
            theme(legend.text=element_text(size=15)) +
            theme(axis.title.x=element_blank(),
                  axis.text=element_blank(),
                  axis.ticks=element_blank()) )
    dev.off()
    
    probs <- data.frame(cell_id = NULL, prob = NULL)
    for (cell_name in plotter$cell_id){
      tmp <- data.frame(cell_id = cell_name, prob = annotation[annotation$cell_names == cell_name, as.character(annotation[annotation$cell_names == cell_name, 'labels'])])
      probs <- rbind(probs,tmp)
    }
    
    plotter <- merge(plotter, probs, by='cell_id')
    pal <- wes_palette("Zissou1", 100, type = "continuous")
    
    pdf(file.path(plots_path, paste0(dataSet,'_',target,'Vs',obj,'_TSNE_probs.pdf')), family="Times", 15,15)
    print(ggplot(plotter, aes(x=V1, y=V2, color = prob, label = labels)) + theme_bw() + geom_text()  +   scale_color_gradientn(colours = pal))
    print(ggplot(plotter, aes(x=V1, y=V2, fill = prob)) + theme_bw() + geom_point(aes(fill=prob),colour="black", shape=21, size = 5) +scale_fill_gradientn(colours = pal) )# scale_fill_viridis(option="inferno"))
    dev.off()
    return(setNames(list(cluster_list, cluster_names), c('cluster_list', 'cluster_names')))
  }
  
}

data_path = "/home/mohit/mohit/seq-rna/Comparison/datasets"
plots_path = "/home/mohit/mohit/seq-rna/Comparison/JIND_DE/Plots/MohitPlotsDENew"

a = DE_with_TSNE('pancreas_01', 'ductal', 'acinar', 25, data_path = data_path, plots_path = plots_path, plottSNE = TRUE)
a = DE_with_TSNE('pancreas_raw_01', 'ductal', 'acinar', 25, data_path = data_path, plots_path = plots_path, plottSNE = TRUE)
a = DE_with_TSNE('human_blood_01', 'Monocyte_FCGR3A', 'Monocyte_CD14', 25, data_path = data_path, plots_path = plots_path, plottSNE = TRUE)
