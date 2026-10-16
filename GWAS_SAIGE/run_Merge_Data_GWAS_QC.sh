#!/bin/bash

exp_name="GWAS_Prote_MDD_Subtype_Clustering"
f_names=("0_VS_12" "0_VS_1" "0_VS_2" "2_VS_1")

for file_name in "${f_names[@]}"; do
    echo "${file_name}"
    cd ./raw_data
    echo "Start filter Individuals on each Chrome"
    for ((chr=1; chr<=22; chr++))
    do
      plink --bfile "c${chr}"  --keep "../${exp_name}/${file_name}/${file_name}.csv" --make-bed --out "../${exp_name}/${file_name}/selIndivi_c${chr}"
    done
    # plink --bfile "cX"  --keep "../${exp_name}/${file_name}/${file_name}.csv" --make-bed --out "../${exp_name}/${file_name}/selIndivi_cX"
    echo "Done: filter Individuals on each Chrome"

    echo "copy allchromes.txt to ${exp_name}/${file_name}/"
    cp -f "../allchromes.txt" "../${exp_name}/${file_name}/"

    echo "start merge all chromes on ${file_name}"
    cd "../${exp_name}/${file_name}/"
    plink --merge-list allchromes.txt --make-bed --out "${file_name}_1"
    rm -rf *.log
    rm -rf *.nosex
    echo "Done: merge chromes all chromes on ${exp_name}/${file_name}_1"

    if [ -d "./GWAS_QC/" ]; then
       rm -rf "./GWAS_QC/"
       echo "删除文件夹 GWAS_QC/ ！"
    else
       echo "文件夹 ./GWAS_QC 不存在！"
    fi
    mkdir -p "./GWAS_QC"
    echo "文件夹 GWAS_QC 创建成功！"
    mv -f "${file_name}_1".*  "./GWAS_QC"

    cp  -f "../../Manhattan_plot.R"   "./"
    cp  -f "../../QQ_plot.R"   "./"
    cd "./GWAS_QC/"

    # Delete SNPs with missingness >0.2.
    plink --bfile "${file_name}_1" --geno 0.2 --make-bed --out "${file_name}_2"
    # Delete individuals with missingness >0.2.
    plink --bfile "${file_name}_2" --mind 0.2 --make-bed --out "${file_name}_3"
    # Delete SNPs with missingness >0.02.
    plink --bfile "${file_name}_3" --geno 0.02 --make-bed --out "${file_name}_4"
    # Delete individuals with missingness >0.02.
    plink --bfile "${file_name}_4" --mind 0.02 --make-bed --out "${file_name}_5"
    # Remove SNPs with a low MAF frequency.
    plink --bfile "${file_name}_5" --maf 0.05 --make-bed --out "${file_name}_8"
    cd "../../../"
done

