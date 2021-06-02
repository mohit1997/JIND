
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

NegativeControl_DE <- function(dataSet, target, obj, data_path = data_path, plots_path = plots_path){
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
  annotation <- pd$read_pickle(file.path(data_path, dataSet, 'JIND_rawtop_0', 'JIND_assignmentbrftune.pkl'))
  annotation$cell_names <- rownames(annotation)
  
  all_data <- t(df[, -which(colnames(df) %in% c('labels'))])
  
  stand_dev <- apply(all_data,1, sd)
  all_data <- all_data[!rownames(all_data) %in% names(which(stand_dev ==0)), ]
  
  misclassifications = annotation[annotation$labels == target & annotation$prediction == obj, 'cell_names']
  
  G1 <- annotation[annotation$labels == target & annotation$prediction != "Unassigned", 'cell_names']
  G2 <- sample(G1, length(misclassifications))
  G1 <- setdiff(G1, G2)
  
  selection <- c(G1, G2)
  
  data_tmp <- all_data[ , colnames(all_data) %in%  selection]
  
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
  
  file_xlsx <- file.path(plots_path, paste0(dataSet,'_',target, '_DENC.xlsx'))
  
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
  write_excel(tmp2, 'NC', file_xlsx)
  write_excel(tmp2[tmp2$adj.P.Val<0.05,], 'NC (FDR < 0.05)', file_xlsx)
  
  data2heat <- data_tmp[tmp2[tmp2$adj.P.Val<0.05,'gene_name'],]
  
  cond = nrow(data2heat)
  
  plot_file = file.path(plots_path, paste0(dataSet,'_',target, '_HM_NC.pdf'))
  
  if (file.exists(plot_file)) {
    file.remove(plot_file)
  }

  if (!is.null(cond)){
    if (cond > 0){
      data2heat[data2heat > 5] <- 5
      
      colors = c(rgb(31, 119, 180, max = 255), rgb(255, 127, 14, max =255),
                 rgb(44, 160, 44, max = 255), rgb(214, 39, 40, max =255),
                 rgb(148, 103, 189, max = 255), rgb(140, 86, 75, max =255),
                 rgb(227, 119, 194, max = 255), rgb(127, 127, 127, max =255),
                 rgb(188, 189, 34, max = 255), rgb(23, 190, 207, max =255))
      
      ann <- data.frame(Group = c(G1, G2))
      ann$Group <-  ifelse(ann$Group %in% G1, 'G1', 'G2')
      rownames(ann) <- colnames(data2heat)
      Var1 <- c(colors[1], colors[2])
      Group         <- c(Var1)
      names(Group) <- c('G1', 'G2')
      anno_colors   <- list(Group = Group)
      
      title = paste0('Heatmap between randomly selected subsets of ',target,' cells (Negative Control)')
      p <- pheatmap( data2heat, cluster_rows = T, treeheight_row = 0, annotation_col = ann, clustering_distance_rows = "euclidean", clustering_distance_cols = "euclidean", cellheight= 10,annotation_colors = anno_colors, show_colnames = F, main = title, fontsize = 6, fontsize_row = 6, callback = callback)
      
      plot_dims <- get_plot_dims(p)
      
      columns = ncol(data2heat)
      multiplier = 200 / columns
      
      plot_file = file.path(plots_path, paste0(dataSet,'_',target, '_HM_NC.pdf'))
      
      pdf(plot_file, family="Times", height = plot_dims$height, width = plot_dims$width * multiplier)
      print(p)
      dev.off()
    }
   
  }
}

data_path = "/home/mohit/mohit/seq-rna/Comparison/datasets"
plots_path = "/home/mohit/mohit/seq-rna/Comparison/JIND_DE/Plots/MohitPlotsNCJIND"

a = NegativeControl_DE('pancreas_01', 'ductal', 'acinar', data_path = data_path, plots_path = plots_path)
a = NegativeControl_DE('pancreas_raw_01', 'ductal', 'acinar', data_path = data_path, plots_path = plots_path)
a = NegativeControl_DE('human_blood_01', 'Monocyte_FCGR3A', 'Monocyte_CD14', data_path = data_path, plots_path = plots_path)
