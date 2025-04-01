#!/bin/bash

# conda activate GWAS
# ./10.PRS_Cal.sh 1e05 1
# nohup ./10.PRS_Cal.sh 1e05 1 &

file_name=$1
parts=$2 # max 7

# https://enkre.net/cgi-bin/code/bgen/tktview/ac405ee793
# it seems not possible to query more than 9990 ranges at once. Doing so, bgenix throws an instance of a db::StatementPreparationError.
# Answer: split your list of ranges into two, and using cat-bgen to put them back together.
cmd=""
for ((chr=1; chr<=22; chr++))
do
    for ((part=1; part<=parts; part++))
    do
        file_path="/data/UKBioBank/FileHandler_And_Dataset/Prote_MDD_Subtype_Clustering数据/result/GWAS/${file_name}/GWAS_Betas_${file_name}_${chr}_${part}.csv"
        echo "file_path： ${file_path}"
        if [ -e "${file_path}" ]; then
            echo "文件存在"
            if [ ! -d "./${file_name}" ]; then
                mkdir "./${file_name}"
                echo "文件夹已创建： ./${file_name}"
            else
                echo "文件夹已存在： ./${file_name}"
            fi

            awk -F, '{ if (NR>1) { print $1 }}'  "${file_path}" > "./${file_name}/rsidlist_${chr}_${part}.txt"
            awk -F, '{ if (NR>1) { print sprintf("%02d", $2)":"$3"-"$3 }}' "${file_path}" > "./${file_name}/chrposlist_${chr}_${part}.txt"

            bgenix -g "/data/UKBioBank/bgen_Data/raw_data/ukb22828_c${chr}_b0_v3.bgen" -incl-rsids "./${file_name}/rsidlist_${chr}_${part}.txt" -incl-range "./${file_name}/chrposlist_${chr}_${part}.txt" > "./${file_name}/${chr}_${part}.bgen"
            cmd=$cmd"./${file_name}/${chr}_${part}.bgen "
        else
            echo "文件不存在"
            continue
        fi
    done
done

echo "需要合并的文件： ${cmd}"
# Combine the .bgen files for each chromosome into one
cat-bgen -g  $cmd -og initial_chr.bgen -clobber
# Write index file .bgen.bgi
bgenix -g initial_chr.bgen -index -clobber

# Remove the individual chromosome files
# for i in {1..22}
# do
#   rm chr_${i}.bgen
# done


all_file="/data/UKBioBank/FileHandler_And_Dataset/Prote_MDD_Subtype_Clustering数据/result/GWAS/${file_name}/GWAS_Betas_${file_name}.csv"
# Import the betas into the sqlite database as a table called Betas
sqlite3 initial_chr.bgen.bgi "DROP TABLE IF EXISTS Betas;"
sqlite3 -separator "," initial_chr.bgen.bgi ".import ${all_file} Betas"

sqlite3 initial_chr.bgen.bgi "DROP TABLE IF EXISTS Joined;"
# And inner join it to the index table (Variants), making a new table (Joined)
# By joining on alleles as well as chromosome and position
# we can ensure only the relevant alleles from any multi-allelic SNPs are retained
sqlite3 -header -csv initial_chr.bgen.bgi \
"CREATE TABLE Joined AS
  SELECT Variant.*, Betas.chr_name, Betas.Beta FROM Variant INNER JOIN Betas
    ON Variant.chromosome = printf('%02d', Betas.chr_name)
    AND Variant.position = Betas.chr_position
    AND Variant.allele1 = Betas.noneffect_allele
    AND Variant.allele2 = Betas.effect_allele
  UNION
  SELECT Variant.*, Betas.chr_name, -Betas.Beta FROM Variant INNER JOIN Betas
    ON Variant.chromosome = printf('%02d', Betas.chr_name)
    AND Variant.position = Betas.chr_position
    AND Variant.allele1 = Betas.effect_allele AND
    Variant.allele2 = Betas.noneffect_allele;"

# Filter the .bgen file to include only the alleles specified in the Betas for each SNP
bgenix -g initial_chr.bgen -table Joined  > single_allelic.bgen

# And produce an index file for the new .bgen
bgenix -g single_allelic.bgen -index -clobber

plink2 --bgen single_allelic.bgen ref-first \
--hard-call-threshold 0.1 \
--sample /data/UKBioBank/bgen_Data/raw_data/ukb22828_c1_b0_v3_s487150.sample \
--memory 50000 \
--set-all-var-ids @:#_\$r_\$a \
--freq \
--make-pgen \
--out raw

awk '/^[^#]/ { if( $5>0.4 && $5<0.6 && ( ($3=="A" && $4=="T") || ($4=="T" && $3=="A") || ($3=="C" && $4=="G") || ($4=="G" && $3=="C") ) ) { print $0 }}' \
raw.afreq > exclrsIDs_ambiguous.txt

########################################################
#----------------------- SNP QC -----------------------#
########################################################

for i in {1..22}
do
  awk -v chr=$i 'BEGIN {FS="\t"; OFS="\t"} { print chr,$0,chr":"$3"_"$4"_"$5 }' /data/UKBioBank/bgen_Data/raw_data/ukb_mfi_chr${i}_v3.txt
done > ukb_mfi_all_v3.tsv


plink2 --pfile raw \
--memory 50000 \
--exclude exclrsIDs_ambiguous.txt \
--extract-col-cond ukb_mfi_all_v3.tsv 9 10 --extract-col-cond-min 0.4 \
--maf 0.005 \
--write-snplist \
--make-pgen \
--out snpQC

########################################################
#--------------------- Sample QC ----------------------#
########################################################

plink2 --pfile raw \
--memory 50000 \
--extract snpQC.snplist \
--keep-fam /data/UKBioBank/bgen_Data/raw_data/usedinpca.txt \
--write-samples \
--out sampleQC

########################################################
#----------------- Calculate the PRS ------------------#
########################################################

sqlite3 -separator " " -list initial_chr.bgen.bgi \
"SELECT chr_name || ':' || position || '_' ||allele1 || '_' || allele2, allele2, Beta FROM Joined;" > "${file_name}_score.txt"

plink2 --pfile raw \
--memory 50000 \
--extract snpQC.snplist \
--keep sampleQC.id \
--score "${file_name}_score.txt" no-mean-imputation \
--out "${file_name}_PRS"
