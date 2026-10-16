#!/bin/bash

exp_name="GWAS_Prote_MDD_Subtype_Clustering"
f_names=("0_VS_12" "0_VS_1" "0_VS_2" "2_VS_1")

for file_name in "${f_names[@]}"; do
    echo "${file_name}"

    Rscript step1_fitNULLGLMM.R     \
        --plinkFile="/pro2/bed_Data/${exp_name}/${file_name}/GWAS_QC/${file_name}_8"  \
        --phenoFile="/pro2/bed_Data/${exp_name}/${file_name}/${file_name}_pheno.txt" \
        --phenoCol=final_cluster \
        --covarColList=sid \
        --qCovarColList=sid  \
        --sampleIDColinphenoFile=IID \
        --traitType=binary        \
        --outputPrefix="/pro2/bed_Data/${exp_name}/${file_name}/Step1_fitNULLGLMM" \
        --nThreads=80	\
        --IsOverwriteVarianceRatioFile=TRUE

    for((i=1;i<=22;i++)); do
        Rscript step2_SPAtests.R        \
            --bedFile="/pro2/bed_Data/${exp_name}/${file_name}/GWAS_QC/${file_name}_8.bed"       \
            --bimFile="/pro2/bed_Data/${exp_name}/${file_name}/GWAS_QC/${file_name}_8.bim"       \
            --famFile="/pro2/bed_Data/${exp_name}/${file_name}/GWAS_QC/${file_name}_8.fam"       \
            --AlleleOrder=alt-first \
            --SAIGEOutputFile="/pro2/bed_Data/${exp_name}/${file_name}/Step2_SPAtests_c${i}.txt" \
            --chrom=${i}       \
            --minMAF=0 \
            --minMAC=20 \
            --GMMATmodelFile="/pro2/bed_Data/${exp_name}/${file_name}/Step1_fitNULLGLMM.rda" \
            --varianceRatioFile="/pro2/bed_Data/${exp_name}/${file_name}/Step1_fitNULLGLMM.varianceRatio.txt"   \
            --LOCO=TRUE \
            --is_output_moreDetails=TRUE
    done
done