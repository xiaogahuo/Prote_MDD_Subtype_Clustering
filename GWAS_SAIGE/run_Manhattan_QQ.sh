#!/bin/bash

exp_name="GWAS_Prote_MDD_Subtype_Clustering"
f_names=("0_VS_12" "0_VS_1" "0_VS_2" "2_VS_1")

for file_name in "${f_names[@]}"; do
    echo "${file_name}"
    cd "./${exp_name}/${file_name}/"
    Rscript Manhattan_plot.R
    Rscript QQ_plot.R
    cd "../../"
done