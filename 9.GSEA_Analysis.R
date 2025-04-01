#清空环境变量
rm(list = ls())
options(stringsAsFactors = F)
setwd("/data/UKBioBank/FileHandler_And_Dataset/Prote_MDD_Subtype_Clustering数据/")
#加载相关的包
# if (!requireNamespace("BiocManager", quietly = TRUE))
#   install.packages("BiocManager")
# library(BiocManager)
# BiocManager::install("ReactomePA")
# remove.packages("AnnotationDbi")
# install.packages("./AnnotationDbi_1.66.0.tar.gz", repos = NULL, type = "source")
# install.packages("./org.Hs.eg.db_3.19.1.tar.gz", repos = NULL, type = "source")
# install.packages("./reactome.db_1.88.0.tar.gz", repos = NULL, type = "source")
# install.packages("./ReactomePA_1.48.0.tar.gz", repos = NULL, type = "source")
library(data.table)
library(clusterProfiler)
library(dplyr)
library(org.Hs.eg.db)#加载所需物种基因组注释信息，可有20个物种选择，http://bioconductor.org/packages/release/BiocViews.html#___OrgDb
library(ggplot2)
library(enrichplot)
# https://mp.weixin.qq.com/s/jd-T3v9rc1mJLaoopOcM7g

#新建文件夹储存Result
directory_path <- "./result/GO_KEGG_GSEA/"
directory_exists <- dir.exists(directory_path)
if (!directory_exists) {
  dir.create(directory_path)
}

# 读入差异分析后文件，包含基因列（gene symbol或entrizid）和logFC列
clst <- 'clst2'
data <- read.csv(file=paste('./result/GSEA_data_', clst, '.csv', sep=''), header=TRUE, sep=",")
#查看文件内容
head(data)
#    SYMBOL     logFC
#1  FAM182B -5.612055
#2    MEP1B -5.134941
#3 TMEM167B -3.383565
#4    GRIK5 -4.903500
#5     XKR9 -5.718563
#6    NDST2 -6.843956
#本次文件基因名为gene symbol，将gene symbol转换为ENTREZID
#如果你的文件是entrizid则跳过该条语句
gene.df <- bitr(data$GeneName,
               fromType = "UNIPROT",
               toType = c("ENTREZID"),
               OrgDb = org.Hs.eg.db)
#查看前6行数据
head(gene.df)
#  SYMBOL ENTREZID
#1  FAM182B   728882
#2    MEP1B     4225
#3 TMEM167B    56900
#4    GRIK5     2901
#5     XKR9   389668
#6    NDST2     8509
data2=merge(data,gene.df,by.x="GeneName", by.y="UNIPROT")#合并数据
head(data2)
geneList<-data2$logFC #第二列可以是foldchange，也可以是logFC
names(geneList)=data2$ENTREZID #使用转换好的ENTREZID
geneList=sort(geneList,decreasing = T) #从高到低排序
# 接下来开始正式的GSEA分析啦
# 使用GO数据库进行GSEA分析
ego <- gseGO(
 geneList     = geneList,
 OrgDb        = org.Hs.eg.db,
 ont          = "ALL",
 nPerm        = 1000,
 minGSSize    = 100,
 maxGSSize    = 500,
 pvalueCutoff = 0.05,
 verbose      = FALSE,
 seed = FALSE, by = "fgsea")
##geneList是前面整理好的排序后的ENTREZID
##OrgDb选择物种注释的数据库
##ont同enrichGO分析，可选3大分类BP、CC、MF或ALL
##nPerm置换检验的次数，默认为1000
##minGSSize富集到某个条目的最小包含基因数，如果基因数小于该值则这个条目将被过滤掉，默认为10
##maxGSSize富集到某个条目的最大包含基因数，如果基因数大于该值则这个条目将被过滤掉，默认为500
##verbose是否输出提示信息
##seed是否使结果具有可重复性
##by选择使用的统计学方法，默认为fgsea
head(ego)[1:6][1:6]
##                   ID                              Description setSize enrichmentScore      NES
##GO:0098687 GO:0098687                       chromosomal region     320       0.4187365 1.667091
##GO:0000793 GO:0000793                     condensed chromosome     209       0.4459184 1.713171
##GO:0000775 GO:0000775           chromosome, centromeric region     191       0.4502879 1.722934
##GO:0000776 GO:0000776                              kinetochore     133       0.4537928 1.677579
##GO:0000779 GO:0000779 condensed chromosome, centromeric region     118       0.5091699 1.850798
##GO:0072562 GO:0072562                      blood microparticle     114       0.4655046 1.680016
##                pvalue
##GO:0098687 0.001052632
##GO:0000793 0.001113586
##GO:0000775 0.001123596
##GO:0000776 0.001154734
##GO:0000779 0.001187648
##GO:0072562 0.001197605
##保存结果
write.csv(ego, file=paste('./result/GO_KEGG_GSEA/goGSEA_', clst, '.csv', sep=''), row.names = FALSE)
#使用KEGG数据库进行GSEA分析
kk <- gseKEGG(geneList, organism = "hsa",pvalueCutoff = 0.2,
             nPerm = 1000, minGSSize = 100, maxGSSize = 500,
             verbose = TRUE, seed = FALSE, by = "fgsea")
##参数同上
##按照enrichment score从高到低排序，便于查看富集通路
kk_order<-kk[order(kk$enrichmentScore,decreasing=T)]
head(kk_order)[1:6][1:6]
##               ID                         Description setSize enrichmentScore      NES      pvalue
##hsa03030 hsa03030                     DNA replication      36       0.5600885 1.664196 0.008415147
##hsa04740 hsa04740              Olfactory transduction     287       0.5005390 1.968686 0.001054852
##hsa04110 hsa04110                          Cell cycle     124       0.4983236 1.807226 0.001165501
##hsa04610 hsa04610 Complement and coagulation cascades      84       0.4868227 1.685794 0.004944376
##hsa03008 hsa03008   Ribosome biogenesis in eukaryotes      78       0.4607935 1.581720 0.009962640
##hsa05206 hsa05206                 MicroRNAs in cancer     160       0.3695695 1.371388 0.019252548
##保存结果
write.csv(kk_order, file=paste('./result/GO_KEGG_GSEA/keggGSEA_', clst, '.csv', sep=''), row.names = FALSE)
#使用MsigDb数据库进行GSEA分析
##下载离线GMT文件，可以根据你的需要下载，也可以是全部，网址如下
##https://www.gsea-msigdb.org/gsea/downloads.jsp
# all<-read.gmt("msigdb.v7.4.entrez.gmt") #读gmt文件
all<-read.gmt("./msigdb_v2023.2.Hs_files_to_download_locally/msigdb_v2023.2.Hs_GMTs/msigdb.v2023.2.Hs.entrez.gmt") #读gmt文件
gsea<-GSEA(geneList,TERM2GENE = all)
##参数大致同上，不同是多了TERM2GENE和TERM2NAME
##TERM2GENE数据集中条目名称TERM与基因的对应关系,一般是两列的数据框形式
##TERM2NAME条目名称与名称的对应关系，可用于个性化的GSEA分析，如系定义的基因集，可以为空
write.csv(gsea, file=paste('./result/GO_KEGG_GSEA/GSEA_', clst, '.csv', sep=''), row.names = FALSE)

#使用reactome数据库进行GSEA分析
library(ReactomePA)
pathway <- gsePathway(geneList)
write.csv(pathway, file=paste('./result/GO_KEGG_GSEA/pathwayGSEA_', clst, '.csv', sep=''), row.names = FALSE)

#最简单的点图
dotplot(kk,x="GeneRatio",showCategory = 10)
ggsave(paste('./result/GO_KEGG_GSEA/dotplot1_', clst, '.png', sep=''), width = 8, height = 6, dpi = 300)
##x为横坐标的变量，除了GeneRatio' or 'Count'以外，还有~GeneRatio/BgRatio，~-log(qvalue)
##showCategory为展示前10个条目


#分面点图激活和抑制
dotplot(kk,split=".sign")+facet_grid(~.sign)
ggsave(paste('./result/GO_KEGG_GSEA/dotplot2_', clst, '.png', sep=''), width = 8, height = 6, dpi = 300)
# 通过改变scales改变Y轴展示
dotplot(kk,split=".sign")+facet_wrap(~.sign,scales = "free")
ggsave(paste('./result/GO_KEGG_GSEA/dotplot3_', clst, '.png', sep=''), width = 8, height = 6, dpi = 300)

#网络图
cnetplot(kk, showCategory = 3,foldChange = geneList,node_label="category",
          colorEdge = TRUE)
ggsave(paste('./result/GO_KEGG_GSEA/cnetplot_', clst, '.png', sep=''), width = 8, height = 6, dpi = 300)
##colorEdge不同的term展示不同的颜色
##node_label 可以选择label的内容，category', 'gene', 'all' and 'none', 默认 "all".
##也可以把想要展示的通路名称赋值给y#，然后showCategory = y
##categorySize="pvalue"
##circular=true以环形展示


#网络图
library(enrichplot)
ego2 <- pairwise_termsim(ego)
##直接运行emapplot会报错，要加上这条语句
emapplot(ego2,showCategory=10,color="pvalue",cex_category=1,layout="kk")
ggsave(paste('./result/GO_KEGG_GSEA/emapplot_', clst, '.png', sep=''), width = 8, height = 6, dpi = 300)
##cex_category调节节点的大小
##showCategory展示显示的条目个数


#高级韦恩图
upsetplot(kk)
ggsave(paste('./result/GO_KEGG_GSEA/upsetplot_', clst, '.png', sep=''), width = 8, height = 6, dpi = 300)


# 经典的gsea结果图
library(enrichplot)
gseaplot2(gsea,1,color="red",pvalue_table = T,title="",base_size=10,ES_geom="line")
ggsave(paste('./result/GO_KEGG_GSEA/gsea_', clst, '.png', sep=''), width = 8, height = 6, dpi = 300)
# 按第一个条目做二维码图，并显示p值
#color是enrichment score线的颜色
#base_sizexy轴标题字的大小
#ES_geom是enrichment score线用线或点显示


##当然也可以在一张图上展示多个条目
gseaplot2(gsea,1:4,color="red",pvalue_table = T,title=gsea$Description[1],base_size=10,ES_geom="line")
ggsave(paste('./result/GO_KEGG_GSEA/gsea_multi_', clst, '.png', sep=''), width = 8, height = 6, dpi = 300)

#山脊图
ridgeplot(kk,fill = "pvalue",5)+scale_fill_continuous(type = "viridis")
ggsave(paste('./result/GO_KEGG_GSEA/ridge_plot_', clst, '.png', sep=''), width = 8, height = 4, dpi = 300)

