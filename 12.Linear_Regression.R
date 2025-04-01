if (!requireNamespace("readr", quietly = TRUE)) {
  install.packages("readr")
}
if (!requireNamespace("stats", quietly = TRUE)) {
  install.packages("stats")
}
library(stats)
# install.packages("readxl")
library(readr)
library(readxl)
library(dplyr)
library(plyr)
library(survival)
library(survminer)
setwd("/data/UKBioBank/FileHandler_And_Dataset/Prote_MDD_Subtype_Clustering数据/")


# df$sex <- as.factor(df$sex)
# df$alcohol <- as.numeric(mapvalues(df$alcohol, c("low", "moderate", "high"), c(0, 1, 2)))
# df$PA1 <- as.numeric(mapvalues(df$PA1, c("inactive", "intermediate", "active"), c(0, 1, 2)))

facter_transform <- function(df){
  df <- df %>%
    mutate(
      final_cluster = as.factor(final_cluster),
      Sex_cate = as.factor(Sex_cate))
}

# Coefficients:
#   Estimate Std. Error t value Pr(>|t|)
# (Intercept)     53.844754   0.309188 174.149  < 2e-16 ***
#   sleepscore       0.107161   0.122231   0.877    0.381
# I(sleepscore^2) -0.005165   0.019210  -0.269    0.788
# sexmale         -2.784151   0.046902 -59.362  < 2e-16 ***
# .......
# ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
#
# Residual standard error: 4.112 on 31874 degrees of freedom
# Multiple R-squared:  0.3089,	Adjusted R-squared:  0.3088
# F-statistic:  2036 on 7 and 31874 DF,  p-value: < 2.2e-16
# formula_str <- paste("f.26501.2.0", " ~ HbA1c1 + I(HbA1c1^2) + age + sex + eth + education + Townsen + smoke + alcohol + bmi + PA1 + SBP + Chol + medicine1 + medicine2 + CHD + cancer", sep="")
# formula_str <- paste("f.26501.2.0", " ~ HbA1c_status + age + sex + eth + education + Townsen + smoke + alcohol + bmi + PA1 + SBP + Chol + medicine1 + medicine2 + CHD + cancer", sep="")
# model <- lm(formula_str, data = df)
# summary_result <- summary(model)
# summary_result$coefficients[2,4]


lm_performance <- function(df, cols, X_str, file_name){
  result_lst <-list()
  result_lst <- append(result_lst, "Col,clust1_Estimate,clust1_StdError,clust1_Tval,clust1_Pval,clust2_Estimate,clust2_StdError,clust2_Tval,clust2_Pval,Fstatistic")
  for (i in 1:length(cols)) {
    formula_str <- paste(cols[i], X_str, sep="")
    print(formula_str)
    model <- lm(formula_str, data = df)
    summary_result <- summary(model)

    clust1_Estimate <- summary_result$coefficients[2,1]
    clust1_StdError <- summary_result$coefficients[2,2]
    clust1_Tval <- summary_result$coefficients[2,3]
    clust1_Pval <- summary_result$coefficients[2,4]
    
    clust2_Estimate <- summary_result$coefficients[3,1]
    clust2_StdError <- summary_result$coefficients[3,2]
    clust2_Tval <- summary_result$coefficients[3,3]
    clust2_Pval <- summary_result$coefficients[3,4]
    Fstatistic <- summary_result$fstatistic[[1]]
    result_lst <- append(result_lst, paste(cols[i], clust1_Estimate, clust1_StdError, clust1_Tval, clust1_Pval, clust2_Estimate, clust2_StdError, clust2_Tval, clust2_Pval, Fstatistic, sep=","))
  }
  writeLines(unlist(result_lst), paste("result/LR", file_name, "rst.csv", sep="_"))
  df_result <- read.csv(file=paste("result/LR", file_name, "rst.csv", sep="_"), sep=",")
  df_result$P2_fdr <- p.adjust(df_result$clust2_Pval, method = "fdr")
  df_result$P1_fdr <- p.adjust(df_result$clust1_Pval, method = "fdr")
  write.csv(df_result, file = paste("result/LR", file_name, "rst.csv", sep="_"), row.names = FALSE)
  # df_P2_fdr <- subset(df_result, P2_fdr<=0.05 )
  # write.csv(df_P2_fdr, file = paste("HbA1c1", file_name, "only_fdr_P2.csv", sep="_"), row.names = FALSE)
  # df_P1_fdr <- subset(df_result, P1_fdr<=0.05 )
  # write.csv(df_P1_fdr, file = paste("HbA1c1", file_name, "only_fdr_P1.csv", sep="_"), row.names = FALSE)
}


df <- read_csv("./dataset/Linear_Regression_Prote_Dataset.csv")
df <- facter_transform(df)
# model <- lm('TMSB10 ~ final_cluster', data = df)
# summary_result <- summary(model)
df_cols_Prote <- read_csv("./result/t_test_result_bonferroni_Diag.csv")
lm_performance(df, df_cols_Prote$col, " ~ final_cluster","Prote")
lm_performance(df, df_cols_Prote$col, " ~ final_cluster + Sex_cate + Age","Prote_with_Sex_Age")

df <- read_csv("./dataset/Linear_Regression_PRS_Dataset.csv")
df <- facter_transform(df)
lm_performance(df, c('PRS_SCORE_5e08', 'PRS_SCORE_1e07', 'PRS_SCORE_5e07', 'PRS_SCORE_1e06', 'PRS_SCORE_5e06'), " ~ final_cluster","PRS")
lm_performance(df, c('PRS_SCORE_5e08', 'PRS_SCORE_1e07', 'PRS_SCORE_5e07', 'PRS_SCORE_1e06', 'PRS_SCORE_5e06'), " ~ final_cluster + Sex_cate + Age","PRS_with_Sex_Age")

df <- read_csv("./dataset/Linear_Regression_MRI_Dataset.csv")
df <- facter_transform(df)
# cols_Cortex <- paste0("f", read_excel("./Cate192_cols.xlsx")$FieldID)  # 192
# print(paste("cols_Cortex", length(cols_Cortex), sep=":"))
cols_Cortex_Area <-c('f26721', 'f26822', 'f26722', 'f26823', 'f26723', 'f26824', 'f26724', 'f26825', 'f26725', 'f26826', 'f26726', 'f26827', 'f26752', 'f26853', 'f26727', 'f26828', 'f26728', 'f26829', 'f26729', 'f26830', 'f26754', 'f26855', 'f26730', 'f26831', 'f26731', 'f26832', 'f26732', 'f26833', 'f26733', 'f26834', 'f26734', 'f26835', 'f26735', 'f26836', 'f26737', 'f26838', 'f26736', 'f26837', 'f26738', 'f26839', 'f26739', 'f26840', 'f26740', 'f26841', 'f26741', 'f26842', 'f26742', 'f26843', 'f26743', 'f26844', 'f26744', 'f26845', 'f26745', 'f26846', 'f26746', 'f26847', 'f26747', 'f26848', 'f26748', 'f26849', 'f26749', 'f26850', 'f26750', 'f26851', 'f26751', 'f26852', 'f26753', 'f26854')
cols_Cortex_Thickness <-  c('f26755', 'f26856', 'f26756', 'f26857', 'f26757', 'f26858', 'f26758', 'f26859', 'f26759', 'f26860', 'f26760', 'f26861', 'f26786', 'f26887', 'f26761', 'f26862', 'f26762', 'f26863', 'f26763', 'f26864', 'f26788', 'f26889', 'f26764', 'f26865', 'f26765', 'f26866', 'f26766', 'f26867', 'f26767', 'f26868', 'f26768', 'f26869', 'f26769', 'f26870', 'f26771', 'f26872', 'f26770', 'f26871', 'f26772', 'f26873', 'f26773', 'f26874', 'f26774', 'f26875', 'f26775', 'f26876', 'f26776', 'f26877', 'f26777', 'f26878', 'f26778', 'f26879', 'f26779', 'f26880', 'f26780', 'f26881', 'f26781', 'f26882', 'f26782', 'f26883', 'f26783', 'f26884', 'f26784', 'f26885', 'f26785', 'f26886', 'f26787', 'f26888')
cols_Cortex_Volume <- c('f26789', 'f26890', 'f26790', 'f26891', 'f26791', 'f26892', 'f26792', 'f26893', 'f26793', 'f26894', 'f26819', 'f26920', 'f26794', 'f26895', 'f26795', 'f26896', 'f26796', 'f26897', 'f26821', 'f26922', 'f26797', 'f26898', 'f26798', 'f26899', 'f26799', 'f26900', 'f26800', 'f26901', 'f26801', 'f26902', 'f26802', 'f26903', 'f26804', 'f26905', 'f26803', 'f26904', 'f26805', 'f26906', 'f26806', 'f26907', 'f26807', 'f26908', 'f26808', 'f26909', 'f26809', 'f26910', 'f26810', 'f26911', 'f26811', 'f26912', 'f26812', 'f26913', 'f26813', 'f26914', 'f26814', 'f26915', 'f26815', 'f26916', 'f26816', 'f26917', 'f26817', 'f26918', 'f26818', 'f26919', 'f26820', 'f26921')
cols_MD <- c('f25515', 'f25516', 'f25517', 'f25518', 'f25519', 'f25520', 'f25523', 'f25524', 'f25525', 'f25526', 'f25527', 'f25528', 'f25529', 'f25530', 'f25532', 'f25533', 'f25531', 'f25521', 'f25522', 'f25534', 'f25535', 'f25536', 'f25537', 'f25538', 'f25539', 'f25540', 'f25541')
cols_FA <- c( 'f25488', 'f25489', 'f25490', 'f25491', 'f25492', 'f25493', 'f25496', 'f25497', 'f25498', 'f25499', 'f25500', 'f25501', 'f25502', 'f25503', 'f25505', 'f25506', 'f25504', 'f25494', 'f25495', 'f25507', 'f25508', 'f25509', 'f25510', 'f25511', 'f25512', 'f25513', 'f25514')
cols_subCortex <- c('f25023', 'f25024', 'f25021', 'f25022', 'f25013', 'f25014', 'f25019', 'f25020', 'f25017', 'f25018', 'f25015', 'f25016', 'f25011', 'f25012')

lm_performance(df, cols_Cortex_Area, " ~ final_cluster","MRI_Area")
lm_performance(df, cols_Cortex_Thickness, " ~ final_cluster","MRI_Thickness")
lm_performance(df, cols_Cortex_Volume, " ~ final_cluster","MRI_Volume")
lm_performance(df, cols_MD, " ~ final_cluster","MRI_MD")
lm_performance(df, cols_FA, " ~ final_cluster","MRI_FA")
lm_performance(df, cols_subCortex, " ~ final_cluster","MRI_subCortex")

lm_performance(df, cols_Cortex_Area, " ~ final_cluster + Sex_cate + Age","MRI_Area_with_Sex_Age")
lm_performance(df, cols_Cortex_Thickness, " ~ final_cluster + Sex_cate + Age","MRI_Thickness_with_Sex_Age")
lm_performance(df, cols_Cortex_Volume, " ~ final_cluster + Sex_cate + Age","MRI_Volume_with_Sex_Age")
lm_performance(df, cols_MD, " ~ final_cluster + Sex_cate + Age","MRI_MD_with_Sex_Age")
lm_performance(df, cols_FA, " ~ final_cluster + Sex_cate + Age","MRI_FA_with_Sex_Age")
lm_performance(df, cols_subCortex, " ~ final_cluster + Sex_cate + Age","MRI_subCortex_with_Sex_Age")