#### 安装包
if (!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install("clusterProfiler") 
BiocManager::install("GO.db")
BiocManager::install("org.Hs.eg.db") 
install.packages('ggplot2')
# install.packages("C:/Users/Administrator/Downloads/DOSE_3.24.2.zip", repos = NULL, type = "source")
# install.packages("C:/Users/Administrator/Downloads/GO.db_3.16.0.tar.gz", repos = NULL, type = "source")
# install.packages("C:/Users/Administrator/Downloads/org.Hs.eg.db_3.16.0.tar.gz", repos = NULL, type = "source")

#### 加载包
library(clusterProfiler)
library(tidyverse)
library(ggplot2)
library(forcats)
library(org.Hs.eg.db)
library(dplyr)
library(patchwork)

setwd("/data/UKBioBank/FileHandler_And_Dataset/Prote_MDD_Subtype_Clustering数据/GO&KEGG/")

###加载数据
Prote_cols_subtype1 <- c('RARRES2','IL1RN','ASGR1','PRAP1','PIGR','MVK','NTRK3','DPY30','GSTA1','HAO1','PON3','ADH4','OCLN','IGSF9','NCAM1','GUSB','GFER','CD83','LBR','FURIN')
Prote_cols_subtype2 <- c('GZMB','CCER2','CD38','HNMT','FAP','ADGRE2','HGF','KHK','TIMP1','KYNU','CDH15','RBP5','ADH4','LGALS1','MB','TNFRSF1A','PRL','MSTN','ACY1','ASGR1')
Prote_cols_subtype1_and_2 <- c('CCER2','PIGR','ADAMTS8','FAP','NTRK3','GZMB','SELENOP','CRISP2','LEP','CXCL14','SCGB1A1','CDH15','LGALS3','CST5','MSTN','CRTAC1','PRL','CPM','SUSD5','ASAH2')

data_protein <- data.frame(
   proteinName = Prote_cols_subtype1_and_2
)
file_name <- 'MDD'

# data_protein <- read.csv(file='../cox_all_covs_result_boffni.csv', header=TRUE, sep=",")
# data_protein <- data.frame(
#   proteinName = data_protein$Cols
# )

#### 差异基因提取
#提取case组较control组表达的基因，padj<0.05, 1og2Foldchange >2
#sig_prote <- data %>%
#  dplyr::filter(padj<0.05 & log2FoldChange > 2)

#### GO富集分析
ids <- bitr(data_protein$proteinName,'SYMBOL', 'ENTREZID', 'org.Hs.eg.db')
data_protein <- merge(data_protein, ids, by.x='proteinName', by.y='SYMBOL')
prote_diff<- data_protein$ENTREZID
ego <- enrichGO(prote_diff, 
                OrgDb = "org.Hs.eg.db",
                pAdjustMethod = 'fdr',
                qvalueCutoff = 0.05,
                pvalueCutoff = 0.05,
                ont="all")
write.csv(ego, file=paste("./GO", file_name, "rst1.csv", sep="_"))
ego <- setReadable(ego, OrgDb = org.Hs.eg.db, keyType="ENTREZID")
ego <- ego %>% arrange(desc(Count))
write.csv(ego, file=paste("./GO", file_name, "rst2.csv", sep="_"))

#### 经典出图
#### 从高到低，按p值升序排列;横轴标示差异基因与通路基因的重复基因数
# color_fun <- colorRampPalette(c("red", "yellow", "green"))
# , col = color_fun(ego$'p.adjust')

go_bar1 <- barplot(ego, showCategory=20)+ 
  scale_y_discrete( labels=function(x) str_wrap(x, width = 100))+
  theme(
    axis.text.y = element_text(size = 10, angle = 0, face='bold'),
    # panel.grid.major.x = element_line(colour = 'white'),
    # panel.grid.major.y = element_line(colour = 'white'),
    # panel.grid.minor.x = element_line(colour = 'white'),
    # panel.grid.minor.y = element_line(colour = 'white')
  )
ggsave(filename = paste("./GO", file_name, "bar1.png", sep="_"), plot = go_bar1, device = "png")

#### 仅展示感兴趣通路
# GO_result <- ego@result
# interesting_set = c("chemokine-mediated signaling pathway","response to lipopolysaccharide", "regulation of leukocyte cell-cell adhesion")
# barplot(ego, showCategory=interesting_set) +
#  scale_y_discrete( labels=function(x) str_wrap(x, width = 100))

#### 针对分类关系的基因集，可分别展示
go_bar2 <- barplot(ego,
                   split="ONTOLOGY")+
  facet_grid(ONTOLOGY~., scale="free")+
  scale_y_discrete( labels=function(x) str_wrap(x, width = 100))+
  theme(
    axis.text.y = element_text(size = 10, angle = 0, face='bold'),
    # axis.text = element_text(size = 1, angle = 0, hjust = 1, vjust = 0.5),
  )
ggsave(filename = paste("./GO", file_name, "bar2.png", sep="_"), plot = go_bar2, device = "png")

#### dotplot
####### 不论类别，选前10个展示
go_dot1 <- dotplot(ego)
ggsave(filename = paste("./GO", file_name, "dot1.png", sep="_"), plot = go_dot1, device = "png")
###### 选取每个类别前10展示
go_dot2 <- dotplot(ego,split="ONTOLOGY")
ggsave(filename = paste("./GO", file_name, "dot2.png", sep="_"), plot = go_dot2, device = "png")
##### 分 面 展示
go_dot3 <- dotplot(ego,split ="ONTOLOGY")+
  facet_grid(ONTOLOGY~., scale="free")+
  scale_y_discrete( labels=function(x) str_wrap(x, width = 100))+
  theme(
    axis.text = element_text(size = 1, angle = 0, hjust = 1, vjust = 0.5),
  )
ggsave(filename = paste("./GO", file_name, "dot3.png", sep="_"), plot = go_dot3, device = "png")

####棒棒糖图
go_dot4 <- ggplot(ego, showCategory = 20, 
                  aes(GeneRatio, fct_reorder(Description, GeneRatio))) +
  geom_segment(aes(xend=0,yend = Description)) + 
  geom_point(aes(color=p.adjust, size = Count)) +
  scale_color_viridis_c(guide=guide_colorbar(reverse=TRUE))+
  scale_size_continuous(range=c(2,10)) +
  theme_minimal() +
  ylab(NULL)
ggsave(filename = paste("./GO", file_name, "dot4.png", sep="_"), plot = go_dot4, device = "png")

#### 富集通路与差异基因的网络关系图

geneList = data_protein$proteinName
names(geneList) <- data_protein$proteinName

p1 = cnetplot(ego, foldChange=geneList,showCategory =3)
p2 = cnetplot(ego, foldChange=geneList,showCategory =3, circular=T)
p3 = cnetplot(ego, foldChange=geneList,showCategory =3, colorEdge=T, node_label='gene') 
p4 = cnetplot(ego, foldChange=geneList,showCategory =3, layout= 'gem')
net1 <- (p1 + p2) / (p3 + p4)
ggsave(filename = paste("./", file_name, "net1.png", sep="_"), plot = net1, device = "png")

# heatplot(ego, foldChange=geneList)

#### 富集通路之间的网络关系图
res1 <- enrichplot::pairwise_termsim(ego)
empp <- emapplot(res1, showCategory = 15, layout="kk", cex_category=1.5, min_edge = 0.8)
ggsave(filename = paste("./", file_name, "empp.png", sep="_"), plot = empp, device = "png")


###KEGG富集分析###
KEGG_enrich = enrichKEGG(gene = data_protein$ENTREZID, #即待富集的基因列表
                         keyType = "kegg",
                         pAdjustMethod = 'fdr',  #指定p值校正方法
                         organism= "human",  #hsa，可根据你自己要研究的物种更改，可在https://www.kegg.jp/brite/br08611中寻找
                         qvalueCutoff = 1, #指定 p 值阈值（可指定 1 以输出全部）
                         pvalueCutoff=1) #指定 q 值阈值（可指定 1 以输出全部）

KEGG_enrich  = data.frame(KEGG_enrich)
write.csv(KEGG_enrich,paste("./KEGG", file_name, "rst1.csv", sep="_")) #数据导出

# 从CSV文件中读取GO富集结果
KEGG_enrich <- read.csv(paste("./KEGG", file_name, "rst1.csv", sep="_"))
# 创建新的一列'term'，将'ID'和'Description'合并
KEGG_enrich$term <- paste(KEGG_enrich$ID, KEGG_enrich$Description, sep = ': ')

# 将'term'列转换为有序因子，以便绘制纵向柱状图时按照指定的顺序显示
# go_enrich$term <- factor(go_enrich$term, levels = go_enrich$term, ordered = TRUE)
KEGG_enrich$ONTOLOGY <- 'KEGG'

# 筛选p值从小到大的前20个条目
KEGG_enrich_filtered <- KEGG_enrich %>%
  arrange(pvalue) %>%
  group_by(ONTOLOGY) %>%
  slice_head(n = 20) %>%
  ungroup()

# 根据p值对term进行重新排序
# KEGG_enrich_filtered$term <- factor(KEGG_enrich_filtered$term, levels = rev(KEGG_enrich_filtered$term))
KEGG_enrich_filtered <- KEGG_enrich_filtered %>% arrange(desc(Count))

# 绘制纵向柱状图并调整字体大小
kegg_p <- ggplot(KEGG_enrich_filtered, aes(x = term, y = Count, fill = ONTOLOGY)) +
  geom_bar(stat = "identity", width = 0.8) +
  scale_fill_manual(values = c("#3690c0")) +
  facet_wrap(~ONTOLOGY, ncol = 1, scales = "free_y") +
  coord_flip() +
  # xlab("GO term") +
  # ylab("Gene_Number") +
  labs(title = "KEGG Pathways Enrichment") +
  theme_bw() +
  theme(axis.text.y = element_text(size = 10, angle = 0, face='bold'),
        panel.grid.major.x = element_line(colour = 'white'),
        panel.grid.major.y = element_line(colour = 'white'),
        panel.grid.minor.x = element_line(colour = 'white'),
        panel.grid.minor.y = element_line(colour = 'white'))
# ,
# axis.text.y = element_text(size = 11, angle = 0, hjust = 1),
ggsave(filename = paste("./", file_name, "KEGG_p.png", sep="_"), plot = kegg_p, device = "png")

# 气泡图
kegg_p2 <- ggplot(KEGG_enrich_filtered,
                  aes(y = term, x = Count)) +
  geom_point(aes(size = Count, color = pvalue)) +
  facet_wrap(~ONTOLOGY, ncol = 1, scales = "free_y") +
  scale_color_gradient(low = "red", high = "blue") +
  labs(color = expression(PValue, size = "Count"),
       x = "Gene Ratio", y = "GO term", title = "GO Enrichment") +
  theme_bw()
ggsave(filename = paste("./", file_name, "KEGG_p2.png", sep="_"), plot = kegg_p2, device = "png")


# 数据导入
kk_result <- read.csv(paste("./KEGG", file_name, "rst1.csv", sep="_"), sep = ",")

# 按照p-value从小到大排序
kk_result <- kk_result[order(kk_result$pvalue), ]

# 数据处理
display_number <- 20 # 显示数量设置
kk <- head(kk_result, n = display_number)

# 重新编码order列为有序因子
kk$order <- factor(rev(as.integer(rownames(kk))), labels = rev(kk$Description))

# 绘制柱状图
kegg_p3 <- ggplot(kk, aes(y = order, x = Count, fill = pvalue)) +
  geom_bar(stat = "identity", width = 0.8) +
  scale_fill_gradient(low = "red", high = "blue") +
  labs(title = "KEGG Pathways Enrichment",
       # x = "Gene number",
       y = "Pathway"
  ) +
  theme_bw() +
  theme(
    axis.text.y = element_text(size = 10, angle = 0, face='bold')
    # axis.title.x = element_text(face = "bold", size = 11),
    # axis.title.y = element_text(face = "bold", size = 11),
    # legend.title = element_text(face = "bold", size = 11)
  )
ggsave(filename = paste("./", file_name, "KEGG_p3.png", sep="_"), plot = kegg_p3, device = "png")

# 绘制气泡图
kegg_p4 <- ggplot(kk, aes(y = order, x = GeneRatio)) +
  geom_point(aes(size = Count, color = pvalue)) +
  scale_color_gradient(low = "red", high = "blue") +
  labs(color = expression(PValue, size = "Count"),
       x = "Gene Ratio", y = "Pathways", title = "KEGG Pathway Enrichment") +
  theme_bw()
ggsave(filename = paste("./", file_name, "KEGG_p4.png", sep="_"), plot = kegg_p4, device = "png")