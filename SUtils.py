import pandas as pd
import os
import numpy as np
from scipy.stats import chi2_contingency, zscore
from scipy.stats.mstats import kruskalwallis
from sklearn.feature_selection import SelectKBest, f_regression
from scipy.stats import ttest_ind
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from functools import reduce


root_path = "/data/UKBioBank/FileHandler_And_Dataset/"
path = root_path + "Prote_MDD_Subtype_Clustering数据/"
data_path = path + "dataset/"
PRS_path = root_path + "PRS_MDD_Scores/"
img_path = path +"image/"
result_path = path +"result/"
mdd_gwas_summary_file = 'daner_MDDwoBP_20201001_2015iR15iex_HRC_MDDwoBP_iPSYCH2015i_Wray_FinnGen_MVPaf_2_HRC_MAF01.gz'
bed_path = "/data/UKBioBank/bed_Data/"

NC, MDD, UNNORMAL = 0, 1, -1
cate100042_cols = ['2188-0.0','2178-0.0','2296-0.0','2306-0.0']
cate100057_cols = ['1160-0.0','1170-0.0','1180-0.0','1190-0.0','1200-0.0','1210-0.0','1220-0.0']
cate100060_cols =  [c+'-0.0' for c in ['20123','20124','20125','4609','4620','5375','5386','1920','1930','1940','1950','1960','1970','1980','1990','2000','2010','2020','2030','2040','2090','2100','4598','4631','4642','4653','4526','4548','4559','4570','4581','4537','2050','2060','2070','2080','6156','5663','5674','6145','10721','20122','20126','20127']]
cate100061_cols = [c+'-0.0' for c in ['1031','6160','2110','10740']]
cate100065_cols = [c+'-0.0' for c in ['21000']]
cate17518_cols = [c+'-0.0' for c in ['30600','30610','30620','30630','30640','30650','30660','30670','30680','30690','30700','30710','30720','30730','30740','30750','30760','30770','30780','30790','30800','30810','30820','30830','30840','30850','30860','30870','30880','30890']]
cate17518_new_cols = ['ALB','ALP','ALT','APOA1','APOB','AST','DBIL','UREA','CALC','CHOL','CREA','CRP','CYSC','GGT','GLU','HbA1c','HDL','IGF-1','LDL','LP-a','OEST','PHOS','RF','SHBG','TBIL','TEST','TP','TRIG','UA','VITD']

cate145_cols = [c+'-0.0' for c in ['20487','20488','20489','20490','20491','20521','20522','20523','20524','20525','20526','20527','20528','20529','20530','20531','20494','20495','20496','20497','20498']]

cat_cols = ['Sex_cate', 'Townsend_index_cate', 'Qualifications_cate', 'Smoking_status_cate', 'Alcohol_status_cate', 'BMI_cate', 'IPAQ_cate', 'family_history', 'Drug']
conti_cols = ['Age', 'chronic_num']
covr_cols = cat_cols + conti_cols

Prote_cols_all = ['A1BG', 'AAMDC', 'AARSD1', 'ABCA2', 'ABHD14B', 'ABL1', 'ABO', 'ABRAXAS2', 'ACAA1', 'ACADM', 'ACADSB', 'ACAN', 'ACE', 'ACE2', 'ACHE', 'ACOT13', 'ACOX1', 'ACP1', 'ACP5', 'ACP6', 'ACRBP', 'ACRV1', 'ACSL1', 'ACTA2', 'ACTN2', 'ACTN4', 'ACVRL1', 'ACY1', 'ACY3', 'ACYP1', 'ADA', 'ADA2', 'ADAM12', 'ADAM15', 'ADAM22', 'ADAM23', 'ADAM8', 'ADAM9', 'ADAMTS1', 'ADAMTS13', 'ADAMTS15', 'ADAMTS16', 'ADAMTS4', 'ADAMTS8', 'ADAMTSL2', 'ADAMTSL4', 'ADAMTSL5', 'ADCYAP1R1', 'ADD1', 'ADGRB3', 'ADGRD1', 'ADGRE1', 'ADGRE2', 'ADGRE5', 'ADGRF5', 'ADGRG1', 'ADGRG2', 'ADGRV1', 'ADH1B', 'ADH4', 'ADIPOQ', 'ADM', 'ADRA2A', 'AFAP1', 'AFM', 'AFP', 'AGBL2', 'AGER', 'AGR2', 'AGR3', 'AGRN', 'AGRP', 'AGT', 'AGXT', 'AHCY', 'AHNAK', 'AHNAK2', 'AHSA1', 'AHSG', 'AHSP', 'AIDA', 'AIF1', 'AIF1L', 'AIFM1', 'AK1', 'AK2', 'AKAP12', 'AKR1B1', 'AKR1B10', 'AKR1C4', 'AKR7L', 'AKT1S1', 'AKT2', 'AKT3', 'ALCAM', 'ALDH1A1', 'ALDH2', 'ALDH3A1', 'ALDH5A1', 'ALMS1', 'ALPI', 'ALPP', 'AMBN', 'AMBP', 'AMDHD2', 'AMFR', 'AMIGO1', 'AMIGO2', 'AMN', 'AMOT', 'AMOTL2', 'AMPD3', 'AMY2A', 'AMY2B', 'ANG', 'ANGPT1', 'ANGPT2', 'ANGPTL1', 'ANGPTL2', 'ANGPTL3', 'ANGPTL4', 'ANGPTL7', 'ANK2', 'ANKMY2', 'ANKRA2', 'ANKRD54', 'ANP32C', 'ANPEP', 'ANXA1', 'ANXA10', 'ANXA11', 'ANXA2', 'ANXA3', 'ANXA4', 'ANXA5', 'AOC1', 'AOC3', 'AP1G2', 'AP2B1', 'AP3B1', 'AP3S2', 'APBB1IP', 'APCS', 'APEX1', 'APLP1', 'APOA1', 'APOA2', 'APOA4', 'APOB', 'APOBR', 'APOC1', 'APOD', 'APOE', 'APOF', 'APOH', 'APOL1', 'APOM', 'APP', 'APPL2', 'APRT', 'ARAF', 'AREG', 'ARF6', 'ARFIP1', 'ARG1', 'ARG2', 'ARHGAP1', 'ARHGAP25', 'ARHGAP30', 'ARHGAP45', 'ARHGAP5', 'ARHGEF1', 'ARHGEF10', 'ARHGEF12', 'ARHGEF5', 'ARID3A', 'ARID4B', 'ARL13B', 'ARL2BP', 'ARMCX2', 'ARNT', 'ARNTL', 'ARSA', 'ARSB', 'ART3', 'ART5', 'ARTN', 'ASAH1', 'ASAH2', 'ASGR1', 'ASGR2', 'ASPN', 'ASPSCR1', 'ASRGL1', 'ASS1', 'ATF2', 'ATF4', 'ATG16L1', 'ATG4A', 'ATOX1', 'ATP1B1', 'ATP1B2', 'ATP1B3', 'ATP1B4', 'ATP2B4', 'ATP5F1D', 'ATP5IF1', 'ATP5PO', 'ATP6AP2', 'ATP6V1D', 'ATP6V1F', 'ATP6V1G1', 'ATP6V1G2', 'ATRAID', 'ATRN', 'ATXN10', 'ATXN2', 'ATXN2L', 'ATXN3', 'AXIN1', 'AXL', 'AZI2', 'AZU1', 'B2M', 'B3GAT3', 'B3GNT7', 'B4GALT1', 'B4GAT1', 'BABAM1', 'BACH1', 'BAG3', 'BAG4', 'BAG6', 'BAIAP2', 'BAMBI', 'BANK1', 'BAP18', 'BATF', 'BAX', 'BCAM', 'BCAN', 'BCAT1', 'BCAT2', 'BCHE', 'BCL2', 'BCL2L1', 'BCL2L11', 'BCL2L15', 'BCL7A', 'BCL7B', 'BCR', 'BDNF', 'BECN1', 'BEX3', 'BGLAP', 'BGN', 'BHLHE40', 'BHMT2', 'BID', 'BIN2', 'BIRC2', 'BLMH', 'BLNK', 'BLOC1S2', 'BLOC1S3', 'BLVRB', 'BMP10', 'BMP4', 'BMP6', 'BMPER', 'BNIP2', 'BNIP3L', 'BOC', 'BOLA1', 'BOLA2_BOLA2B', 'BPIFB1', 'BPIFB2', 'BRAP', 'BRD1', 'BRD2', 'BRD3', 'BRDT', 'BRK1', 'BRME1', 'BRSK2', 'BSG', 'BSND', 'BST1', 'BST2', 'BTC', 'BTD', 'BTLA', 'BTN1A1', 'BTN2A1', 'BTN3A2', 'BTNL10', 'C19orf12', 'C1GALT1C1', 'C1QA', 'C1QBP', 'C1QL2', 'C1QTNF1', 'C1QTNF5', 'C1QTNF6', 'C1QTNF9', 'C1R', 'C1RL', 'C1S', 'C2', 'C2CD2L', 'C2orf69', 'C3', 'C4BPB', 'C5', 'C7', 'C7orf50', 'C8B', 'C9', 'C9orf40', 'CA1', 'CA11', 'CA12', 'CA13', 'CA14', 'CA2', 'CA3', 'CA4', 'CA5A', 'CA6', 'CA7', 'CA8', 'CA9', 'CABP2', 'CACNA1C', 'CACNA1H', 'CACNB1', 'CACNB3', 'CACYBP', 'CADPS', 'CALB1', 'CALB2', 'CALCA', 'CALCB', 'CALCOCO1', 'CALCOCO2', 'CALY', 'CAMKK1', 'CAMLG', 'CAMSAP1', 'CANT1', 'CAPG', 'CAPN3', 'CAPS', 'CARHSP1', 'CASC3', 'CASP1', 'CASP10', 'CASP2', 'CASP3', 'CASP4', 'CASP7', 'CASP8', 'CASP9', 'CASQ2', 'CAT', 'CBLIF', 'CBLN1', 'CBLN4', 'CBS', 'CBX2', 'CC2D1A', 'CCAR2', 'CCDC134', 'CCDC28A', 'CCDC50', 'CCDC80', 'CCER2', 'CCL11', 'CCL13', 'CCL14', 'CCL15', 'CCL16', 'CCL17', 'CCL18', 'CCL19', 'CCL2', 'CCL20', 'CCL21', 'CCL22', 'CCL23', 'CCL24', 'CCL25', 'CCL26', 'CCL27', 'CCL28', 'CCL3', 'CCL4', 'CCL5', 'CCL7', 'CCL8', 'CCN1', 'CCN2', 'CCN3', 'CCN4', 'CCN5', 'CCND2', 'CCNE1', 'CCS', 'CCT5', 'CD101', 'CD109', 'CD14', 'CD160', 'CD163', 'CD164', 'CD164L2', 'CD177', 'CD1C', 'CD2', 'CD200', 'CD200R1', 'CD207', 'CD209', 'CD22', 'CD226', 'CD244', 'CD248', 'CD27', 'CD274', 'CD276', 'CD28', 'CD2AP', 'CD300A', 'CD300C', 'CD300E', 'CD300LF', 'CD300LG', 'CD302', 'CD33', 'CD34', 'CD36', 'CD38', 'CD3D', 'CD3E', 'CD3G', 'CD4', 'CD40', 'CD40LG', 'CD46', 'CD48', 'CD5', 'CD55', 'CD58', 'CD59', 'CD5L', 'CD6', 'CD63', 'CD69', 'CD7', 'CD70', 'CD72', 'CD74', 'CD79B', 'CD80', 'CD82', 'CD83', 'CD84', 'CD86', 'CD8A', 'CD93', 'CD99', 'CD99L2', 'CDA', 'CDAN1', 'CDC123', 'CDC25A', 'CDC26', 'CDC27', 'CDC37', 'CDC42BPB', 'CDCP1', 'CDH1', 'CDH15', 'CDH17', 'CDH2', 'CDH22', 'CDH23', 'CDH3', 'CDH4', 'CDH5', 'CDH6', 'CDHR1', 'CDHR2', 'CDHR5', 'CDK1', 'CDK5RAP3', 'CDKL5', 'CDKN1A', 'CDKN2D', 'CDNF', 'CDON', 'CDSN', 'CEACAM1', 'CEACAM16', 'CEACAM18', 'CEACAM19', 'CEACAM20', 'CEACAM21', 'CEACAM3', 'CEACAM5', 'CEACAM6', 'CEACAM8', 'CEBPA', 'CEBPB', 'CELA2A', 'CELA3A', 'CELSR2', 'CEMIP2', 'CEND1', 'CENPF', 'CENPJ', 'CEP112', 'CEP152', 'CEP164', 'CEP170', 'CEP20', 'CEP290', 'CEP350', 'CEP43', 'CEP85', 'CERT', 'CES1', 'CES2', 'CES3', 'CETN2', 'CETN3', 'CFB', 'CFC1', 'CFD', 'CFHR2', 'CFHR4', 'CFHR5', 'CFI', 'CFP', 'CGA', 'CGB3_CGB5_CGB8', 'CGN', 'CGREF1', 'CHAC2', 'CHAD', 'CHCHD10', 'CHCHD6', 'CHEK2', 'CHGA', 'CHGB', 'CHI3L1', 'CHIT1', 'CHL1', 'CHM', 'CHMP1A', 'CHMP6', 'CHP1', 'CHRDL1', 'CHRDL2', 'CHRM1', 'CIAPIN1', 'CILP', 'CINP', 'CIRBP', 'CIT', 'CKAP4', 'CKB', 'CKMT1A_CKMT1B', 'CLASP1', 'CLC', 'CLEC10A', 'CLEC11A', 'CLEC12A', 'CLEC14A', 'CLEC1A', 'CLEC1B', 'CLEC2L', 'CLEC3B', 'CLEC4A', 'CLEC4C', 'CLEC4D', 'CLEC4G', 'CLEC4M', 'CLEC5A', 'CLEC6A', 'CLEC7A', 'CLGN', 'CLIC5', 'CLINT1', 'CLIP2', 'CLMP', 'CLNS1A', 'CLPP', 'CLPS', 'CLSPN', 'CLSTN1', 'CLSTN2', 'CLSTN3', 'CLTA', 'CLU', 'CLUL1', 'CMC1', 'CMIP', 'CNDP1', 'CNGB3', 'CNP', 'CNPY2', 'CNPY4', 'CNST', 'CNTF', 'CNTN1', 'CNTN2', 'CNTN3', 'CNTN4', 'CNTN5', 'CNTNAP2', 'CNTNAP4', 'COCH', 'COL15A1', 'COL18A1', 'COL1A1', 'COL24A1', 'COL28A1', 'COL2A1', 'COL3A1', 'COL4A1', 'COL4A4', 'COL5A1', 'COL6A3', 'COL9A1', 'COL9A2', 'COLEC12', 'COMMD1', 'COMMD9', 'COMP', 'COMT', 'COPB2', 'COPE', 'COQ7', 'CORO1A', 'CORO6', 'COX5B', 'COX6B1', 'CPA1', 'CPA2', 'CPA4', 'CPB1', 'CPB2', 'CPE', 'CPLX2', 'CPM', 'CPOX', 'CPPED1', 'CPQ', 'CPTP', 'CPVL', 'CPXM1', 'CPXM2', 'CR1', 'CR2', 'CRACR2A', 'CRADD', 'CREB3', 'CREBZF', 'CREG1', 'CRELD1', 'CRELD2', 'CRH', 'CRHBP', 'CRHR1', 'CRIM1', 'CRIP2', 'CRISP2', 'CRISP3', 'CRKL', 'CRLF1', 'CRNN', 'CRTAC1', 'CRTAM', 'CRTAP', 'CRX', 'CRYBB1', 'CRYBB2', 'CRYGD', 'CRYM', 'CRYZL1', 'CSDE1', 'CSF1', 'CSF1R', 'CSF2', 'CSF2RA', 'CSF2RB', 'CSF3', 'CSF3R', 'CSH1', 'CSNK1D', 'CSNK2A1', 'CSPG4', 'CSPG5', 'CSRP3', 'CST3', 'CST5', 'CST6', 'CST7', 'CSTB', 'CTAG1A_CTAG1B', 'CTBS', 'CTF1', 'CTHRC1', 'CTLA4', 'CTNNA1', 'CTRB1', 'CTRC', 'CTRL', 'CTSB', 'CTSC', 'CTSD', 'CTSE', 'CTSF', 'CTSH', 'CTSL', 'CTSO', 'CTSV', 'CTSZ', 'CUZD1', 'CWC15', 'CX3CL1', 'CXADR', 'CXCL1', 'CXCL10', 'CXCL11', 'CXCL12', 'CXCL13', 'CXCL14', 'CXCL16', 'CXCL17', 'CXCL3', 'CXCL5', 'CXCL6', 'CXCL8', 'CXCL9', 'CYB5A', 'CYB5R2', 'CYP24A1', 'CYTH3', 'CYTL1', 'DAAM1', 'DAB2', 'DAG1', 'DAND5', 'DAPK2', 'DAPP1', 'DARS1', 'DBH', 'DBI', 'DBN1', 'DBNL', 'DCBLD2', 'DCC', 'DCDC2C', 'DCLRE1C', 'DCN', 'DCTD', 'DCTN1', 'DCTN2', 'DCTN6', 'DCTPP1', 'DCUN1D1', 'DCUN1D2', 'DCXR', 'DDA1', 'DDAH1', 'DDC', 'DDHD2', 'DDI2', 'DDR1', 'DDT', 'DDX1', 'DDX25', 'DDX39A', 'DDX4', 'DDX53', 'DDX58', 'DECR1', 'DEFA1_DEFA1B', 'DEFB103A_DEFB103B', 'DEFB104A_DEFB104B', 'DEFB116', 'DEFB118', 'DEFB4A_DEFB4B', 'DENND2B', 'DENR', 'DFFA', 'DGCR6', 'DGKA', 'DGKZ', 'DHODH', 'DHPS', 'DHRS4L2', 'DIABLO', 'DIPK1C', 'DIPK2B', 'DKK1', 'DKK3', 'DKK4', 'DKKL1', 'DLG4', 'DLGAP5', 'DLK1', 'DLL1', 'DLL4', 'DMD', 'DMP1', 'DNAJA1', 'DNAJA2', 'DNAJA4', 'DNAJB1', 'DNAJB14', 'DNAJB2', 'DNAJB6', 'DNAJB8', 'DNAJC21', 'DNAJC6', 'DNAJC9', 'DNER', 'DNLZ', 'DNM1', 'DNM3', 'DNMBP', 'DNPEP', 'DNPH1', 'DOC2B', 'DOCK9', 'DOK1', 'DOK2', 'DPEP1', 'DPEP2', 'DPP10', 'DPP4', 'DPP6', 'DPP7', 'DPT', 'DPY30', 'DRAXIN', 'DRG2', 'DSC2', 'DSCAM', 'DSG2', 'DSG3', 'DSG4', 'DTD1', 'DTNB', 'DTX2', 'DTX3', 'DTYMK', 'DUOX2', 'DUSP13', 'DUSP29', 'DUSP3', 'DUT', 'DXO', 'DYNC1H1', 'DYNLT1', 'DYNLT3', 'EBAG9', 'EBI3_IL27', 'ECE1', 'ECHDC3', 'ECHS1', 'ECI2', 'ECM1', 'ECSCR', 'EDA2R', 'EDAR', 'EDDM3B', 'EDEM2', 'EDF1', 'EDIL3', 'EDN1', 'EDNRB', 'EEF1D', 'EFCAB14', 'EFCAB2', 'EFEMP1', 'EFHD1', 'EFNA1', 'EFNA4', 'EFNB2', 'EGF', 'EGFL7', 'EGFLAM', 'EGFR', 'EGLN1', 'EHBP1', 'EHD3', 'EIF1AX', 'EIF2AK2', 'EIF2AK3', 'EIF2S2', 'EIF4B', 'EIF4E', 'EIF4EBP1', 'EIF4G1', 'EIF4G3', 'EIF5', 'EIF5A', 'ELAC1', 'ELAVL4', 'ELN', 'ELOA', 'ELOB', 'ENAH', 'ENG', 'ENO1', 'ENO2', 'ENO3', 'ENOPH1', 'ENOX2', 'ENPEP', 'ENPP2', 'ENPP5', 'ENPP6', 'ENPP7', 'ENSA', 'ENTPD2', 'ENTPD5', 'ENTPD6', 'ENTR1', 'EP300', 'EPB41L5', 'EPCAM', 'EPGN', 'EPHA1', 'EPHA10', 'EPHA2', 'EPHA4', 'EPHB4', 'EPHB6', 'EPHX2', 'EPN1', 'EPO', 'EPPK1', 'EPS8L2', 'ERBB2', 'ERBB3', 'ERBB4', 'ERBIN', 'ERC2', 'ERCC1', 'EREG', 'ERI1', 'ERMAP', 'ERN1', 'ERP29', 'ERP44', 'ERVV-1', 'ESAM', 'ESM1', 'ESPL1', 'ESR1', 'ESYT2', 'EVI2B', 'EVI5', 'EVPL', 'EXOSC10', 'EXTL1', 'EZR', 'F10', 'F11', 'F11R', 'F12', 'F13B', 'F2', 'F2R', 'F3', 'F7', 'F9', 'FABP1', 'FABP2', 'FABP3', 'FABP4', 'FABP5', 'FABP6', 'FABP9', 'FADD', 'FAM13A', 'FAM171A2', 'FAM171B', 'FAM172A', 'FAM20A', 'FAM3B', 'FAM3C', 'FAM3D', 'FAP', 'FARSA', 'FAS', 'FASLG', 'FBLN2', 'FBN2', 'FBP1', 'FCAMR', 'FCAR', 'FCER1A', 'FCER2', 'FCGR2A', 'FCGR2B', 'FCGR3B', 'FCN1', 'FCN2', 'FCRL1', 'FCRL2', 'FCRL3', 'FCRL5', 'FCRL6', 'FCRLB', 'FDX1', 'FDX2', 'FEN1', 'FES', 'FETUB', 'FGA', 'FGD3', 'FGF12', 'FGF16', 'FGF19', 'FGF2', 'FGF20', 'FGF21', 'FGF23', 'FGF3', 'FGF5', 'FGF6', 'FGF7', 'FGF9', 'FGFBP1', 'FGFBP2', 'FGFBP3', 'FGFR2', 'FGFR4', 'FGL1', 'FGR', 'FH', 'FHIP2A', 'FHIT', 'FIS1', 'FKBP14', 'FKBP1B', 'FKBP4', 'FKBP5', 'FKBP7', 'FKBPL', 'FLI1', 'FLRT2', 'FLT1', 'FLT3', 'FLT3LG', 'FLT4', 'FMNL1', 'FMR1', 'FN1', 'FNDC1', 'FNTA', 'FOLH1', 'FOLR1', 'FOLR2', 'FOLR3', 'FOS', 'FOSB', 'FOXJ3', 'FOXO1', 'FOXO3', 'FRMD4B', 'FRMD7', 'FRZB', 'FSHB', 'FST', 'FSTL1', 'FSTL3', 'FTCD', 'FUCA1', 'FUOM', 'FURIN', 'FUS', 'FUT1', 'FUT3_FUT5', 'FUT8', 'FXN', 'FXYD5', 'FYB1', 'FZD10', 'FZD8', 'GABARAP', 'GABARAPL1', 'GABRA4', 'GAD1', 'GAD2', 'GADD45B', 'GADD45GIP1', 'GAGE2A', 'GAL', 'GALNT10', 'GALNT2', 'GALNT3', 'GALNT5', 'GALNT7', 'GAMT', 'GAPDH', 'GART', 'GAS2', 'GAS6', 'GASK1A', 'GAST', 'GATA3', 'GATD3', 'GBA', 'GBP1', 'GBP2', 'GBP4', 'GBP6', 'GC', 'GCC1', 'GCG', 'GCHFR', 'GCLM', 'GCNT1', 'GDF15', 'GDF2', 'GDNF', 'GET3', 'GFAP', 'GFER', 'GFOD2', 'GFRA1', 'GFRA2', 'GFRA3', 'GFRAL', 'GGA1', 'GGACT', 'GGCT', 'GGH', 'GGT1', 'GGT5', 'GH1', 'GH2', 'GHR', 'GHRHR', 'GHRL', 'GID8', 'GIGYF2', 'GIMAP7', 'GIMAP8', 'GIP', 'GIPC2', 'GIPC3', 'GIPR', 'GIT1', 'GJA8', 'GKN1', 'GLA', 'GLB1', 'GLI2', 'GLO1', 'GLOD4', 'GLP1R', 'GLRX', 'GLRX5', 'GLT8D2', 'GLYR1', 'GM2A', 'GMFG', 'GMPR', 'GMPR2', 'GNAS', 'GNE', 'GNGT1', 'GNLY', 'GNPDA1', 'GNPDA2', 'GOLGA3', 'GOLM2', 'GOPC', 'GORASP2', 'GOT1', 'GP1BA', 'GP1BB', 'GP2', 'GP5', 'GP6', 'GPA33', 'GPC1', 'GPC5', 'GPD1', 'GPHA2', 'GPI', 'GPIHBP1', 'GPKOW', 'GPNMB', 'GPR101', 'GPR158', 'GPR15L', 'GPR37', 'GPRC5C', 'GRAP2', 'GRHPR', 'GRIK2', 'GRIN2B', 'GRK5', 'GRN', 'GRP', 'GRPEL1', 'GRSF1', 'GSAP', 'GSN', 'GSR', 'GSTA1', 'GSTA3', 'GSTM4', 'GSTP1', 'GSTT2B', 'GTF2IRD1', 'GTPBP2', 'GUCA2A', 'GUCY2C', 'GUK1', 'GUSB', 'GYS1', 'GZMA', 'GZMB', 'GZMH', 'H2AP', 'HADH', 'HAGH', 'HAO1', 'HARS1', 'HAVCR1', 'HAVCR2', 'HBEGF', 'HBQ1', 'HBZ', 'HCG22', 'HCLS1', 'HDAC8', 'HDAC9', 'HDDC2', 'HDGF', 'HDGFL2', 'HEBP1', 'HEG1', 'HEPACAM2', 'HEPH', 'HEXIM1', 'HGF', 'HGFAC', 'HGS', 'HHEX', 'HIF1A', 'HIP1', 'HIP1R', 'HJV', 'HK2', 'HLA-A', 'HLA-DRA', 'HLA-E', 'HMBS', 'HMCN2', 'HMGCL', 'HMGCS1', 'HMMR', 'HMOX1', 'HMOX2', 'HNF1A', 'HNMT', 'HNRNPK', 'HNRNPUL1', 'HPCAL1', 'HPGDS', 'HPSE', 'HRAS', 'HRC', 'HRG', 'HS1BP3', 'HS3ST3B1', 'HS6ST1', 'HS6ST2', 'HSBP1', 'HSD11B1', 'HSD17B14', 'HSD17B3', 'HSDL2', 'HSP90B1', 'HSPA1A', 'HSPA2', 'HSPB1', 'HSPB6', 'HSPG2', 'HTR1A', 'HTR1B', 'HTRA2', 'HYAL1', 'HYOU1', 'ICA1', 'ICAM1', 'ICAM2', 'ICAM3', 'ICAM4', 'ICAM5', 'ICOSLG', 'ID4', 'IDI2', 'IDO1', 'IDS', 'IDUA', 'IFI30', 'IFIT1', 'IFIT3', 'IFNAR1', 'IFNG', 'IFNGR1', 'IFNGR2', 'IFNL1', 'IFNL2', 'IFNLR1', 'IFNW1', 'IFT20', 'IGBP1', 'IGDCC3', 'IGDCC4', 'IGF1R', 'IGF2BP3', 'IGF2R', 'IGFBP1', 'IGFBP2', 'IGFBP3', 'IGFBP4', 'IGFBP6', 'IGFBP7', 'IGFBPL1', 'IGFL4', 'IGHMBP2', 'IGLC2', 'IGLON5', 'IGSF21', 'IGSF3', 'IGSF8', 'IGSF9', 'IKBKG', 'IKZF2', 'IL10', 'IL10RA', 'IL10RB', 'IL11', 'IL12A_IL12B', 'IL12B', 'IL12RB1', 'IL12RB2', 'IL13', 'IL13RA1', 'IL13RA2', 'IL15', 'IL15RA', 'IL16', 'IL17A', 'IL17C', 'IL17D', 'IL17F', 'IL17RA', 'IL17RB', 'IL18', 'IL18BP', 'IL18R1', 'IL18RAP', 'IL19', 'IL1A', 'IL1B', 'IL1R1', 'IL1R2', 'IL1RAP', 'IL1RL1', 'IL1RL2', 'IL1RN', 'IL2', 'IL20', 'IL20RA', 'IL20RB', 'IL21R', 'IL22', 'IL22RA1', 'IL24', 'IL25', 'IL2RA', 'IL2RB', 'IL2RG', 'IL3', 'IL31', 'IL31RA', 'IL32', 'IL33', 'IL34', 'IL36A', 'IL36G', 'IL3RA', 'IL4', 'IL4R', 'IL5', 'IL5RA', 'IL6', 'IL6R', 'IL6ST', 'IL7', 'IL7R', 'IL9', 'ILKAP', 'IMMT', 'IMPA1', 'IMPACT', 'IMPG1', 'ING1', 'INHBB', 'INHBC', 'INPP1', 'INPP5D', 'INPP5J', 'INPPL1', 'INSL3', 'INSL4', 'INSL5', 'INSR', 'IPCEF1', 'IQGAP2', 'IRAG2', 'IRAK1', 'IRAK4', 'ISLR2', 'ISM1', 'ISM2', 'IST1', 'ITGA11', 'ITGA2', 'ITGA5', 'ITGA6', 'ITGAL', 'ITGAM', 'ITGAV', 'ITGAX', 'ITGB1', 'ITGB1BP1', 'ITGB1BP2', 'ITGB2', 'ITGB5', 'ITGB6', 'ITGB7', 'ITGBL1', 'ITIH1', 'ITIH3', 'ITIH4', 'ITIH5', 'ITM2A', 'ITPA', 'ITPR1', 'ITPRIP', 'IVD', 'IZUMO1', 'JAM2', 'JAM3', 'JCHAIN', 'JMJD1C', 'JPT2', 'JUN', 'KAZALD1', 'KAZN', 'KCNC4', 'KCNH2', 'KCNIP4', 'KCTD5', 'KDM3A', 'KDR', 'KEL', 'KHDC3L', 'KHK', 'KIAA0319', 'KIAA1549', 'KIAA1549L', 'KIAA2013', 'KIF1C', 'KIF20B', 'KIF22', 'KIFBP', 'KIR2DL2', 'KIR2DL3', 'KIR2DS4', 'KIR3DL1', 'KIR3DL2', 'KIRREL1', 'KIRREL2', 'KIT', 'KITLG', 'KLB', 'KLF4', 'KLHL41', 'KLK1', 'KLK10', 'KLK11', 'KLK12', 'KLK13', 'KLK14', 'KLK15', 'KLK3', 'KLK4', 'KLK6', 'KLK7', 'KLK8', 'KLKB1', 'KLRB1', 'KLRC1', 'KLRD1', 'KLRF1', 'KLRK1', 'KRT14', 'KRT17', 'KRT18', 'KRT19', 'KRT5', 'KRT6C', 'KRT8', 'KYAT1', 'KYNU', 'L1CAM', 'L3HYPDH', 'LACRT', 'LACTB2', 'LAG3', 'LAIR1', 'LAIR2', 'LAMA1', 'LAMA4', 'LAMB1', 'LAMP1', 'LAMP2', 'LAMP3', 'LAMTOR5', 'LAP3', 'LARP1', 'LAT', 'LAT2', 'LATS1', 'LAYN', 'LBP', 'LBR', 'LCAT', 'LCN15', 'LCN2', 'LCP1', 'LDLR', 'LDLRAP1', 'LECT2', 'LEFTY2', 'LELP1', 'LEO1', 'LEP', 'LEPR', 'LETM1', 'LGALS1', 'LGALS3', 'LGALS3BP', 'LGALS4', 'LGALS7B', 'LGALS8', 'LGALS9', 'LGMN', 'LHB', 'LHPP', 'LIF', 'LIFR', 'LILRA2', 'LILRA3', 'LILRA4', 'LILRA5', 'LILRA6', 'LILRB1', 'LILRB2', 'LILRB4', 'LILRB5', 'LIPF', 'LMNB1', 'LMNB2', 'LMOD1', 'LMOD2', 'LONP1', 'LPA', 'LPCAT2', 'LPL', 'LPO', 'LPP', 'LRCH4', 'LRFN2', 'LRG1', 'LRIG1', 'LRIG3', 'LRP1', 'LRP11', 'LRP2', 'LRP2BP', 'LRPAP1', 'LRRC25', 'LRRC37A2', 'LRRC38', 'LRRC59', 'LRRFIP1', 'LRRN1', 'LRTM1', 'LRTM2', 'LSM1', 'LSM8', 'LSP1', 'LTA', 'LTA4H', 'LTB', 'LTBP2', 'LTBP3', 'LTBR', 'LTO1', 'LUZP2', 'LXN', 'LY6D', 'LY75', 'LY9', 'LY96', 'LYAR', 'LYN', 'LYPD1', 'LYPD3', 'LYPD8', 'LYPLA2', 'LYSMD3', 'LYVE1', 'LYZL2', 'LZTFL1', 'M6PR', 'MAD1L1', 'MAEA', 'MAG', 'MAGEA3', 'MAGED1', 'MAMDC2', 'MAMDC4', 'MAN1A2', 'MAN2B2', 'MANEAL', 'MANF', 'MANSC1', 'MANSC4', 'MAP1LC3A', 'MAP1LC3B2', 'MAP2', 'MAP2K1', 'MAP2K6', 'MAP3K5', 'MAP4K5', 'MAPK13', 'MAPK9', 'MAPKAPK2', 'MAPRE3', 'MAPT', 'MARCO', 'MARS1', 'MASP1', 'MATN2', 'MATN3', 'MAVS', 'MAX', 'MB', 'MBL2', 'MCAM', 'MCEE', 'MCEMP1', 'MCFD2', 'MCTS1', 'MDGA1', 'MDH1', 'MDK', 'MDM1', 'MECR', 'MED18', 'MED21', 'MEGF10', 'MEGF11', 'MEGF9', 'MELTF', 'MENT', 'MEP1A', 'MEP1B', 'MEPE', 'MERTK', 'MESD', 'MET', 'METAP1', 'METAP1D', 'METAP2', 'MFAP3', 'MFAP3L', 'MFAP4', 'MFAP5', 'MFGE8', 'MGLL', 'MGMT', 'MIA', 'MICALL2', 'MICB_MICA', 'MIF', 'MILR1', 'MINDY1', 'MINK1', 'MITD1', 'MKI67', 'MLLT1', 'MLN', 'MME', 'MMP1', 'MMP10', 'MMP12', 'MMP13', 'MMP15', 'MMP3', 'MMP7', 'MMP8', 'MMP9', 'MMUT', 'MN1', 'MNAT1', 'MNDA', 'MOCS2', 'MOG', 'MORC3', 'MORF4L1', 'MORF4L2', 'MORN4', 'MPHOSPH8', 'MPI', 'MPIG6B', 'MPO', 'MPRIP', 'MRC1', 'MRI1', 'MRPL24', 'MRPL28', 'MRPL46', 'MRPL52', 'MRPL58', 'MRPS16', 'MSLN', 'MSLNL', 'MSMB', 'MSR1', 'MSRA', 'MST1', 'MSTN', 'MTDH', 'MTHFD2', 'MTHFSD', 'MTIF3', 'MTPN', 'MTR', 'MTSS1', 'MTSS2', 'MTUS1', 'MUC13', 'MUC16', 'MUC2', 'MUCL3', 'MVK', 'MXRA8', 'MYBPC1', 'MYBPC2', 'MYCBP2', 'MYDGF', 'MYH4', 'MYH7B', 'MYH9', 'MYL1', 'MYL3', 'MYL4', 'MYL6B', 'MYLPF', 'MYO6', 'MYO9B', 'MYOC', 'MYOM1', 'MYOM2', 'MYOM3', 'MZB1', 'MZT1', 'NAA10', 'NAA80', 'NAAA', 'NACC1', 'NADK', 'NAGA', 'NAGK', 'NAGPA', 'NAMPT', 'NAP1L4', 'NAPRT', 'NARS1', 'NBL1', 'NBN', 'NCAM1', 'NCAM2', 'NCAN', 'NCF2', 'NCK2', 'NCLN', 'NCR1', 'NCR3LG1', 'NCS1', 'NDRG1', 'NDST1', 'NDUFA5', 'NDUFB7', 'NDUFS6', 'NEB', 'NECAP2', 'NECTIN1', 'NECTIN2', 'NECTIN4', 'NEDD4L', 'NEDD9', 'NEFL', 'NEK7', 'NELL1', 'NELL2', 'NENF', 'NEO1', 'NEXN', 'NFASC', 'NFAT5', 'NFATC1', 'NFATC3', 'NFE2', 'NFIC', 'NFKB1', 'NFKB2', 'NFKBIE', 'NFU1', 'NFX1', 'NFYA', 'NGF', 'NGFR', 'NGRN', 'NHLRC3', 'NID1', 'NID2', 'NINJ1', 'NIT1', 'NIT2', 'NLGN1', 'NLGN2', 'NME1', 'NME3', 'NMI', 'NMNAT1', 'NMRK2', 'NMT1', 'NOMO1', 'NOP56', 'NOS1', 'NOS2', 'NOS3', 'NOTCH1', 'NOTCH2', 'NOTCH3', 'NPC2', 'NPDC1', 'NPHS1', 'NPHS2', 'NPL', 'NPPB', 'NPPC', 'NPR1', 'NPTN', 'NPTX1', 'NPTX2', 'NPTXR', 'NPY', 'NRCAM', 'NRGN', 'NRN1', 'NRP1', 'NRP2', 'NRTN', 'NRXN3', 'NSFL1C', 'NT5C', 'NT5C1A', 'NT5C3A', 'NT5E', 'NTF3', 'NTF4', 'NTproBNP', 'NTRK2', 'NTRK3', 'NUB1', 'NUBP1', 'NUCB2', 'NUDC', 'NUDT10', 'NUDT15', 'NUDT16', 'NUDT2', 'NUDT5', 'NUMB', 'NUP50', 'NXPE4', 'NXPH1', 'NXPH3', 'OBP2B', 'OCLN', 'ODAM', 'OFD1', 'OGA', 'OGFR', 'OGN', 'OGT', 'OLR1', 'OMD', 'OMG', 'OMP', 'OPHN1', 'OPLAH', 'OPTC', 'ORM1', 'OSBPL2', 'OSCAR', 'OSM', 'OSMR', 'OSTN', 'OTOA', 'OTUD6B', 'OTUD7B', 'OXCT1', 'OXT', 'P4HB', 'PACS2', 'PADI2', 'PADI4', 'PAEP', 'PAFAH1B3', 'PAFAH2', 'PAG1', 'PAGR1', 'PAIP2B', 'PAK4', 'PALLD', 'PALM', 'PALM2', 'PALM3', 'PAM', 'PAMR1', 'PAPPA', 'PARD3', 'PARK7', 'PARP1', 'PAXX', 'PBK', 'PBLD', 'PBXIP1', 'PCARE', 'PCBD1', 'PCBP2', 'PCDH1', 'PCDH12', 'PCDH17', 'PCDH7', 'PCDH9', 'PCDHB15', 'PCNA', 'PCSK7', 'PCSK9', 'PCYT2', 'PDAP1', 'PDCD1', 'PDCD1LG2', 'PDCD5', 'PDCD6', 'PDCL2', 'PDE1C', 'PDE4D', 'PDE5A', 'PDGFA', 'PDGFB', 'PDGFC', 'PDGFRA', 'PDGFRB', 'PDIA2', 'PDIA3', 'PDIA4', 'PDIA5', 'PDLIM5', 'PDLIM7', 'PDP1', 'PDRG1', 'PDXDC1', 'PDZD2', 'PDZK1', 'PEAR1', 'PEBP1', 'PECAM1', 'PECR', 'PENK', 'PEPD', 'PER3', 'PF4', 'PFDN2', 'PFDN4', 'PFDN6', 'PFKFB2', 'PGA4', 'PGD', 'PGF', 'PGLYRP1', 'PGLYRP2', 'PGLYRP4', 'PGM2', 'PGR', 'PHACTR2', 'PHLDB1', 'PHLDB2', 'PHOSPHO1', 'PHYKPL', 'PI16', 'PI3', 'PIBF1', 'PIGR', 'PIK3AP1', 'PIK3IP1', 'PIKFYVE', 'PILRA', 'PILRB', 'PINLYP', 'PITHD1', 'PKD1', 'PKD2', 'PKLR', 'PKN3', 'PLA2G10', 'PLA2G15', 'PLA2G1B', 'PLA2G2A', 'PLA2G4A', 'PLA2G7', 'PLAT', 'PLAU', 'PLAUR', 'PLB1', 'PLCB1', 'PLCB2', 'PLEKHO1', 'PLG', 'PLIN1', 'PLIN3', 'PLPBP', 'PLSCR3', 'PLTP', 'PLXDC1', 'PLXDC2', 'PLXNA4', 'PLXNB2', 'PLXNB3', 'PM20D1', 'PMCH', 'PMM2', 'PMS1', 'PMVK', 'PNLIP', 'PNLIPRP1', 'PNLIPRP2', 'PNMA1', 'PNMA2', 'PNPT1', 'PODXL', 'PODXL2', 'POF1B', 'POLR2A', 'POLR2F', 'POMC', 'PON1', 'PON2', 'PON3', 'POSTN', 'PPBP', 'PPCDC', 'PPIB', 'PPIE', 'PPIF', 'PPL', 'PPM1A', 'PPM1B', 'PPM1F', 'PPME1', 'PPP1CC', 'PPP1R12A', 'PPP1R12B', 'PPP1R14A', 'PPP1R14D', 'PPP1R2', 'PPP1R9B', 'PPP2R5A', 'PPP3R1', 'PPT1', 'PPY', 'PQBP1', 'PRAME', 'PRAP1', 'PRC1', 'PRCP', 'PRDX1', 'PRDX2', 'PRDX3', 'PRDX5', 'PRDX6', 'PREB', 'PRELP', 'PRG2', 'PRG3', 'PRKAB1', 'PRKAG3', 'PRKAR1A', 'PRKAR2A', 'PRKCQ', 'PRKD2', 'PRKG1', 'PRKRA', 'PRL', 'PRND', 'PROC', 'PROCR', 'PROK1', 'PROS1', 'PRR4', 'PRR5', 'PRRT3', 'PRSS2', 'PRSS22', 'PRSS27', 'PRSS53', 'PRSS8', 'PRTFDC1', 'PRTG', 'PRTN3', 'PRUNE2', 'PSAP', 'PSAPL1', 'PSCA', 'PSG1', 'PSIP1', 'PSMA1', 'PSMC3', 'PSMD1', 'PSMD5', 'PSMD9', 'PSME1', 'PSME2', 'PSMG3', 'PSMG4', 'PSPN', 'PSRC1', 'PTEN', 'PTGDS', 'PTGES2', 'PTGR1', 'PTH', 'PTH1R', 'PTK7', 'PTN', 'PTP4A3', 'PTPN1', 'PTPN6', 'PTPN9', 'PTPRB', 'PTPRC', 'PTPRF', 'PTPRH', 'PTPRK', 'PTPRM', 'PTPRN2', 'PTPRR', 'PTPRS', 'PTPRZ1', 'PTRHD1', 'PTS', 'PTTG1', 'PTX3', 'PVALB', 'PVR', 'PXDNL', 'PXN', 'PYDC1', 'PYY', 'PZP', 'QDPR', 'QPCT', 'QSOX1', 'RAB10', 'RAB11FIP3', 'RAB27B', 'RAB2B', 'RAB33A', 'RAB37', 'RAB39B', 'RAB3GAP1', 'RAB44', 'RAB6A', 'RAB6B', 'RABEP1', 'RABEPK', 'RABGAP1L', 'RAC3', 'RAD23B', 'RAD51', 'RALB', 'RALY', 'RANBP1', 'RANBP2', 'RANGAP1', 'RAP1A', 'RAPGEF2', 'RARRES1', 'RARRES2', 'RASA1', 'RASGRF1', 'RASSF2', 'RBFOX3', 'RBKS', 'RBM17', 'RBM19', 'RBM25', 'RBP1', 'RBP2', 'RBP5', 'RBP7', 'RBPMS', 'RBPMS2', 'RCC1', 'RCOR1', 'RECK', 'REEP4', 'REG1A', 'REG1B', 'REG3A', 'REG3G', 'REG4', 'RELB', 'RELT', 'REN', 'REPS1', 'REST', 'RET', 'RETN', 'REXO2', 'RFC4', 'RGCC', 'RGL2', 'RGMA', 'RGMB', 'RGS10', 'RGS8', 'RHOC', 'RICTOR', 'RIDA', 'RILP', 'RILPL2', 'RIPK4', 'RLN1', 'RLN2', 'RNASE1', 'RNASE10', 'RNASE3', 'RNASE4', 'RNASE6', 'RNASEH2A', 'RNASET2', 'RNF149', 'RNF168', 'RNF31', 'RNF4', 'RNF41', 'RNF43', 'RNF5', 'ROBO1', 'ROBO2', 'ROBO4', 'ROR1', 'RP2', 'RPA2', 'RPE', 'RPGR', 'RPL14', 'RPS10', 'RRAS', 'RRM2', 'RRM2B', 'RRP15', 'RSPO1', 'RSPO3', 'RTBDN', 'RTKN2', 'RTN4IP1', 'RTN4R', 'RUVBL1', 'RWDD1', 'RYR1', 'S100A11', 'S100A12', 'S100A13', 'S100A14', 'S100A16', 'S100A3', 'S100A4', 'S100G', 'S100P', 'SAA4', 'SAFB2', 'SAG', 'SAMD9L', 'SAP18', 'SARG', 'SART1', 'SAT1', 'SAT2', 'SATB1', 'SBSN', 'SCAMP3', 'SCARA5', 'SCARB1', 'SCARB2', 'SCARF1', 'SCARF2', 'SCG2', 'SCG3', 'SCGB1A1', 'SCGB2A2', 'SCGB3A1', 'SCGB3A2', 'SCGN', 'SCIN', 'SCLY', 'SCN2A', 'SCN2B', 'SCN3A', 'SCN3B', 'SCN4B', 'SCP2', 'SCPEP1', 'SCRG1', 'SCRIB', 'SCRN1', 'SCT', 'SDC1', 'SDC4', 'SDCCAG8', 'SDHB', 'SDK2', 'SEC31A', 'SEL1L', 'SELE', 'SELENOP', 'SELL', 'SELP', 'SELPLG', 'SEMA3F', 'SEMA3G', 'SEMA4C', 'SEMA4D', 'SEMA6C', 'SEMA7A', 'SEPTIN3', 'SEPTIN7', 'SEPTIN8', 'SEPTIN9', 'SERPINA1', 'SERPINA11', 'SERPINA12', 'SERPINA3', 'SERPINA4', 'SERPINA5', 'SERPINA6', 'SERPINA7', 'SERPINA9', 'SERPINB1', 'SERPINB5', 'SERPINB6', 'SERPINB8', 'SERPINB9', 'SERPINC1', 'SERPIND1', 'SERPINE1', 'SERPINE2', 'SERPINF1', 'SERPINF2', 'SERPING1', 'SERPINH1', 'SERPINI1', 'SERPINI2', 'SESTD1', 'SETMAR', 'SEZ6', 'SEZ6L', 'SEZ6L2', 'SF3B4', 'SFRP1', 'SFRP4', 'SFTPA1', 'SFTPA2', 'SFTPD', 'SGSH', 'SH2B3', 'SH2D1A', 'SH3BGRL2', 'SH3BP1', 'SH3GL3', 'SH3GLB2', 'SHBG', 'SHC1', 'SHD', 'SHH', 'SHISA5', 'SHMT1', 'SHPK', 'SIAE', 'SIGLEC1', 'SIGLEC10', 'SIGLEC15', 'SIGLEC5', 'SIGLEC6', 'SIGLEC7', 'SIGLEC8', 'SIGLEC9', 'SIL1', 'SIRPA', 'SIRPB1', 'SIRT1', 'SIRT2', 'SIRT5', 'SIT1', 'SKAP1', 'SKAP2', 'SKIV2L', 'SLA2', 'SLAMF1', 'SLAMF6', 'SLAMF7', 'SLAMF8', 'SLC12A2', 'SLC13A1', 'SLC16A1', 'SLC1A4', 'SLC27A4', 'SLC28A1', 'SLC34A3', 'SLC39A14', 'SLC39A5', 'SLC44A4', 'SLC4A1', 'SLC51B', 'SLC9A3R1', 'SLC9A3R2', 'SLIRP', 'SLIT2', 'SLITRK1', 'SLITRK2', 'SLITRK6', 'SLK', 'SLMAP', 'SLURP1', 'SMAD1', 'SMAD2', 'SMAD3', 'SMAD5', 'SMARCA2', 'SMC3', 'SMNDC1', 'SMOC1', 'SMOC2', 'SMPD1', 'SMPD3', 'SMPDL3A', 'SMPDL3B', 'SMS', 'SMTN', 'SNAP23', 'SNAP25', 'SNAP29', 'SNAPIN', 'SNCA', 'SNCG', 'SNED1', 'SNRPB2', 'SNU13', 'SNX15', 'SNX18', 'SNX2', 'SNX5', 'SNX9', 'SOD1', 'SOD2', 'SOD3', 'SORBS1', 'SORCS2', 'SORD', 'SORT1', 'SOST', 'SOWAHA', 'SOX2', 'SOX9', 'SPACA5_SPACA5B', 'SPAG1', 'SPARC', 'SPARCL1', 'SPART', 'SPESP1', 'SPINK1', 'SPINK2', 'SPINK4', 'SPINK5', 'SPINK6', 'SPINK8', 'SPINT1', 'SPINT2', 'SPINT3', 'SPOCK1', 'SPON1', 'SPON2', 'SPP1', 'SPRED2', 'SPRING1', 'SPRR1B', 'SPRR3', 'SPRY2', 'SPTBN2', 'SPTLC1', 'SRC', 'SRP14', 'SRPK2', 'SRPX', 'SSB', 'SSBP1', 'SSC4D', 'SSC5D', 'SSH3', 'SSNA1', 'ST13', 'ST3GAL1', 'ST6GAL1', 'ST8SIA1', 'STAB2', 'STAM', 'STAMBP', 'STAT2', 'STAT5B', 'STAU1', 'STC1', 'STC2', 'STEAP4', 'STIP1', 'STK11', 'STK24', 'STK4', 'STOML2', 'STX16', 'STX1B', 'STX3', 'STX4', 'STX5', 'STX6', 'STX7', 'STX8', 'STXBP1', 'STXBP3', 'SUGP1', 'SUGT1', 'SULT1A1', 'SULT2A1', 'SUMF1', 'SUMF2', 'SUOX', 'SUSD1', 'SUSD2', 'SUSD4', 'SUSD5', 'SV2A', 'SWAP70', 'SYAP1', 'SYNGAP1', 'SYT1', 'SYTL4', 'TAB2', 'TACC3', 'TACSTD2', 'TADA3', 'TAFA5', 'TAGLN3', 'TALDO1', 'TANK', 'TAP1', 'TARBP2', 'TARM1', 'TARS1', 'TAX1BP1', 'TBC1D17', 'TBC1D23', 'TBC1D5', 'TBCA', 'TBCB', 'TBCC', 'TBL1X', 'TBR1', 'TCL1A', 'TCL1B', 'TCN1', 'TCN2', 'TCOF1', 'TCP11', 'TCTN3', 'TDGF1', 'TDO2', 'TDP1', 'TDRKH', 'TEF', 'TEK', 'TERF1', 'TET2', 'TEX101', 'TEX33', 'TF', 'TFAP2A', 'TFF1', 'TFF2', 'TFF3', 'TFPI', 'TFPI2', 'TFRC', 'TG', 'TGFA', 'TGFB1', 'TGFB2', 'TGFBI', 'TGFBR1', 'TGFBR2', 'TGFBR3', 'TGM2', 'TGOLN2', 'THAP12', 'THBD', 'THBS2', 'THBS4', 'THOP1', 'THPO', 'THRAP3', 'THSD1', 'THTPA', 'THY1', 'TIA1', 'TIE1', 'TIGAR', 'TIGIT', 'TIMD4', 'TIMM10', 'TIMM8A', 'TIMP1', 'TIMP2', 'TIMP3', 'TIMP4', 'TINAGL1', 'TJAP1', 'TJP3', 'TK1', 'TLR1', 'TLR2', 'TLR3', 'TLR4', 'TMCO5A', 'TMED1', 'TMED10', 'TMED4', 'TMED8', 'TMEM106A', 'TMEM132A', 'TMEM25', 'TMOD4', 'TMPRSS11B', 'TMPRSS11D', 'TMPRSS15', 'TMPRSS5', 'TMSB10', 'TNC', 'TNF', 'TNFAIP2', 'TNFAIP8', 'TNFAIP8L2', 'TNFRSF10A', 'TNFRSF10B', 'TNFRSF10C', 'TNFRSF11A', 'TNFRSF11B', 'TNFRSF12A', 'TNFRSF13B', 'TNFRSF13C', 'TNFRSF14', 'TNFRSF17', 'TNFRSF19', 'TNFRSF1A', 'TNFRSF1B', 'TNFRSF21', 'TNFRSF4', 'TNFRSF6B', 'TNFRSF8', 'TNFRSF9', 'TNFSF10', 'TNFSF11', 'TNFSF12', 'TNFSF13', 'TNFSF13B', 'TNFSF14', 'TNFSF8', 'TNIP1', 'TNN', 'TNNI3', 'TNPO1', 'TNR', 'TNXB', 'TOMM20', 'TOP1', 'TOP1MT', 'TOP2B', 'TOR1AIP1', 'TP53', 'TP53BP1', 'TP53I3', 'TP53INP1', 'TP73', 'TPBGL', 'TPD52L2', 'TPK1', 'TPM3', 'TPMT', 'TPP1', 'TPPP2', 'TPPP3', 'TPR', 'TPRKB', 'TPSAB1', 'TPSD1', 'TPSG1', 'TPT1', 'TRAF2', 'TRAF3', 'TRAF3IP2', 'TRDMT1', 'TREH', 'TREM2', 'TREML1', 'TREML2', 'TRIAP1', 'TRIM21', 'TRIM24', 'TRIM25', 'TRIM26', 'TRIM40', 'TRIM5', 'TRIM58', 'TRPV3', 'TSC1', 'TSC22D1', 'TSHB', 'TSLP', 'TSNAX', 'TSPAN1', 'TSPAN15', 'TSPAN7', 'TSPAN8', 'TSPYL1', 'TST', 'TTF2', 'TTN', 'TTR', 'TUBB3', 'TWF2', 'TXK', 'TXLNA', 'TXN', 'TXNDC15', 'TXNDC5', 'TXNDC9', 'TXNL1', 'TXNRD1', 'TYMP', 'TYRO3', 'TYRP1', 'UBAC1', 'UBE2B', 'UBE2L6', 'UBE2Z', 'UBQLN3', 'UBXN1', 'UFD1', 'UGDH', 'UHRF2', 'ULBP2', 'UMOD', 'UNC5D', 'UNC79', 'UNG', 'UPB1', 'UPK3A', 'UPK3BL1', 'UROD', 'UROS', 'USO1', 'USP25', 'USP28', 'USP47', 'USP8', 'UXS1', 'VAMP5', 'VAMP8', 'VASH1', 'VASN', 'VASP', 'VAT1', 'VAV3', 'VCAM1', 'VCAN', 'VCPKMT', 'VEGFA', 'VEGFB', 'VEGFC', 'VEGFD', 'VGF', 'VIM', 'VIPR1', 'VIT', 'VMO1', 'VNN1', 'VNN2', 'VPS28', 'VPS37A', 'VPS4B', 'VPS53', 'VSIG10', 'VSIG10L', 'VSIG2', 'VSIG4', 'VSIR', 'VSNL1', 'VSTM1', 'VSTM2B', 'VSTM2L', 'VTA1', 'VTCN1', 'VTI1A', 'VWA1', 'VWA5A', 'VWC2', 'VWC2L', 'VWF', 'WARS', 'WAS', 'WASF1', 'WASF3', 'WASHC3', 'WASL', 'WDR46', 'WFDC1', 'WFDC12', 'WFDC2', 'WFIKKN1', 'WFIKKN2', 'WIF1', 'WNT9A', 'WWP2', 'XCL1', 'XG', 'XIAP', 'XPNPEP2', 'XRCC4', 'YAP1', 'YARS1', 'YES1', 'YJU2', 'YOD1', 'YTHDF3', 'YWHAQ', 'YY1', 'ZBP1', 'ZBTB16', 'ZBTB17', 'ZCCHC8', 'ZFYVE19', 'ZHX2', 'ZNF174', 'ZNF75D', 'ZNF830', 'ZNRD2', 'ZNRF4', 'ZP3', 'ZP4', 'ZPR1']
# print("Prote_cols_all: {}, {}".format(len(Prote_cols_all), Prote_cols_all))

Prote_cols_subtype1 = ['RARRES2','IL1RN','ASGR1','PRAP1','PIGR','MVK','NTRK3','DPY30','GSTA1','HAO1','PON3','ADH4','OCLN','IGSF9','NCAM1','GUSB','GFER','CD83','LBR','FURIN']
Prote_cols_subtype2 = ['CD38','HNMT','CCER2','KHK','GZMB','FAP','TIMP1','HGF','ADH4','ADGRE2','ASGR1','TNFRSF1A','KYNU','MB','LGALS1','RBP5','PRL','RARRES2','RNASET2','ACY1']
Prote_cols_subtype1_and_2 = ['CCER2','PIGR','ADAMTS8','FAP','NTRK3','GZMB','SELENOP','CRISP2','LEP','CXCL14','SCGB1A1','CDH15','LGALS3','CST5','MSTN','CRTAC1','PRL','CPM','SUSD5','ASAH2']

cols_MRI_c192_Area = [26721, 26822, 26722, 26823, 26723, 26824, 26724, 26825, 26725, 26826, 26726, 26827, 26752, 26853, 26727, 26828, 26728, 26829, 26729, 26830, 26754, 26855, 26730, 26831, 26731, 26832, 26732, 26833, 26733, 26834, 26734, 26835, 26735, 26836, 26737, 26838, 26736, 26837, 26738, 26839, 26739, 26840, 26740, 26841, 26741, 26842, 26742, 26843, 26743, 26844, 26744, 26845, 26745, 26846, 26746, 26847, 26747, 26848, 26748, 26849, 26749, 26850, 26750, 26851, 26751, 26852, 26753, 26854]
cols_MRI_c192_Thickness = [26755, 26856, 26756, 26857, 26757, 26858, 26758, 26859, 26759, 26860, 26760, 26861, 26786, 26887, 26761, 26862, 26762, 26863, 26763, 26864, 26788, 26889, 26764, 26865, 26765, 26866, 26766, 26867, 26767, 26868, 26768, 26869, 26769, 26870, 26771, 26872, 26770, 26871, 26772, 26873, 26773, 26874, 26774, 26875, 26775, 26876, 26776, 26877, 26777, 26878, 26778, 26879, 26779, 26880, 26780, 26881, 26781, 26882, 26782, 26883, 26783, 26884, 26784, 26885, 26785, 26886, 26787, 26888]
cols_MRI_c192_Volume = [26789, 26890, 26790, 26891, 26791, 26892, 26792, 26893, 26793, 26894, 26819, 26920, 26794, 26895, 26795, 26896, 26796, 26897, 26821, 26922, 26797, 26898, 26798, 26899, 26799, 26900, 26800, 26901, 26801, 26902, 26802, 26903, 26804, 26905, 26803, 26904, 26805, 26906, 26806, 26907, 26807, 26908, 26808, 26909, 26809, 26910, 26810, 26911, 26811, 26912, 26812, 26913, 26813, 26914, 26814, 26915, 26815, 26916, 26816, 26917, 26817, 26918, 26818, 26919, 26820, 26921]

cols_MRI_c135_MD = [25515,25516,25517,25518,25519,25520,25523,25524,25525,25526,25527,25528,25529,25530,25532,25533,25531,25521,25522,25534,25535,25536,25537,25538,25539,25540,25541]
cols_MRI_c135_FA = [25488,25489,25490,25491,25492,25493,25496,25497,25498,25499,25500,25501,25502,25503,25505,25506,25504,25494,25495,25507,25508,25509,25510,25511,25512,25513,25514]
cols_MRI_c1102_subcortical_volume = [25023, 25024, 25021, 25022, 25013, 25014, 25019, 25020, 25017, 25018, 25015, 25016, 25011, 25012]
cols_MRI_field_25753 = ["col"+str(i) for i in range(0, 1485)]
# print(cols_MRI_field_25753)

# 将labels转换为numpy数组，并使用广播和向量化操作来计算相似度矩阵。这样可以避免使用双重循环，提高代码性能。
def compute_similarity_matrix(labels):
    labels = np.array(labels)
    similarity_matrix = (labels[:, None] == labels).astype(int)
    return similarity_matrix


def annot_trans(pval):
    if pval < 0.001:
        return '***'
    elif pval < 0.01:
        return '**'
    elif pval < 0.05:
        return '*'
    else:
        return ' '

def count_1feature(feature_name, df):
    print('----------打印字段: <' + feature_name + "> 的人数分组统计情况----------")
    target_feature_value_list = df.drop_duplicates(subset=[feature_name], keep='first')[feature_name].tolist()
    target_feature_value_list.sort()
    for value in target_feature_value_list:
        print("字段< %s > 类别 < %s > 人数统计: %d" % (feature_name, str(value), df[df[feature_name] == value].shape[0]))


def count_2features(feature_name1, feature_name2, df):
    print('----------打印字段: <' + feature_name1 + "> 和 字段< " + feature_name2 + " > 的分组统计情况----------")
    target_feature_value_list1 = df.drop_duplicates(subset=[feature_name1], keep='first')[feature_name1].tolist()
    target_feature_value_list1.sort()
    target_feature_value_list2 = df.drop_duplicates(subset=[feature_name2], keep='first')[feature_name2].tolist()
    target_feature_value_list2.sort()

    for value1 in target_feature_value_list1:
        for value2 in target_feature_value_list2:
            print("字段< %s > 类别 < %s > 字段< %s > 类别 < %s > 人数统计: %d" % (
            feature_name1, str(value1), feature_name2, str(value2),
            df[(df[feature_name1] == value1) & (df[feature_name2] == value2)].shape[0]))


def count_Chi2(cnt_feature, div_feature, df):
    print('卡方检验结果: <' + cnt_feature + "> on <" + div_feature + ">")
    # print(df.drop_duplicates(subset=[target_feature], keep='first')[target_feature])
    div_feature_value_list = sorted(df.drop_duplicates(subset=[div_feature], keep='first')[div_feature].tolist())
    cnt_feature_value_list = sorted(df.drop_duplicates(subset=[cnt_feature], keep='first')[cnt_feature].tolist())
    print("{}: {}".format(div_feature, div_feature_value_list))
    print("{}: {}".format(cnt_feature, cnt_feature_value_list))

    kf_data = []
    for div in div_feature_value_list:
        c_g =[]
        for value in cnt_feature_value_list:
            count = df[(df[div_feature] == div) & (df[cnt_feature] == value)].shape[0]
            c_g.append(count)
        kf_data.append(c_g)
    print(kf_data)
    kf = chi2_contingency(kf_data)
    print('chisq-statistic=%.4f, p-value=%.4f, df=%i expected_frep=%s' % kf)
    print('-'*25)
    return kf, kf_data


def kwtest(feature_name, target_feature, df):
    print('kwtest结果: <' + feature_name + "> on <" + target_feature + ">")
    # print(df.drop_duplicates(subset=[target_feature], keep='first')[target_feature])
    target_feature_value_list = df.drop_duplicates(subset=[target_feature], keep='first')[target_feature].tolist()
    print(target_feature_value_list)
    kw_data = []
    for value in target_feature_value_list:
        feature_name_value_list = df[df[target_feature] == value][feature_name].tolist()
        kw_data.append(feature_name_value_list)
    statistic, pvalue = kruskalwallis(*kw_data) #  将kw_data中的值作为参数传入kruskalwallis中, 第1个值是第1个参数， 第2个值是第2个参数
    print("column: {}, statistic: {}, pvalue: {}".format(feature_name, statistic, pvalue))
    print('-'*25)


def count_mean_std_group_by_div_feature(feature_name, div_feature_name, df):
    print(
        '----------打印字段: <' + feature_name + "> 根据字段< " + div_feature_name + " > 的分组 均值(Mean) 标准差(Std) 统计情况----------")
    print('字段< %s > 总体Mean: %f' % (feature_name, df[feature_name].mean()))
    print('字段< %s > 总体Std: %f' % (feature_name, df[feature_name].std(ddof=1)))
    print('-' * 25)
    div_feature_value_list = df.drop_duplicates(subset=[div_feature_name], keep='first')[div_feature_name].tolist()
    div_feature_value_list.sort()

    for div_value in div_feature_value_list:
        print("字段< %s > 按字段< %s > 类别 < %s > 统计的均值 (Mean): %f" % (
            feature_name, div_feature_name, str(div_value), df[df[div_feature_name] == div_value][feature_name].mean()))
        print("字段< %s > 按字段< %s > 类别 < %s > 统计的标准差 (Std): %f" % (
            feature_name, div_feature_name, str(div_value),
            df[df[div_feature_name] == div_value][feature_name].std(ddof=1)))


def count_mean_std(feature_name, df):
    print('----------打印字段: <' + feature_name + "> 的均值(Mean) 标准差(Std) 统计情况----------")
    print('字段< %s > 总体Mean: %f' % (feature_name, df[feature_name].mean()))
    print('字段< %s > 总体Std: %f' % (feature_name, df[feature_name].std(ddof=1)))
    print('-' * 25)


def count_T(feature_name, target_feature, df):
    count_mean_std_group_by_div_feature(feature_name, target_feature, df)
    print('T检验结果: <' + feature_name + "> on <" + target_feature + ">")
    # print(df.drop_duplicates(subset=[target_feature], keep='first')[target_feature])
    target_feature_value_list = df.drop_duplicates(subset=[target_feature], keep='first')[target_feature].tolist()
    c1 = target_feature_value_list[0]
    c2 = target_feature_value_list[1]
    v1_list = df[df[target_feature] == c1][feature_name].tolist()
    v2_list = df[df[target_feature] == c2][feature_name].tolist()
    v1_list = [v for v in v1_list if not pd.isnull(v)]
    v2_list = [v for v in v2_list if not pd.isnull(v)]
    print("size:{}".format(len(v1_list)))
    print("size:{}".format(len(v2_list)))

    # Run a two sample t-test to compare the two samples
    tstat, pval = ttest_ind(a=v1_list, b=v2_list, alternative="two-sided")

    # Display results
    print("t-stat: {:.2f}   pval: {:.4f}".format(tstat, pval))
    return tstat, pval


palette = {"HC": "#009432", "Subtpye1": "#F79F1F", "Subtpye2": "#EA2027", "MDD": "#EAB543"}
def plot_bar(df, file_name, feature_lst, width, high, hue_order, y_label):
    df_bar = []
    for feature in feature_lst:
        df_sub = df[[feature, 'final_cluster']].copy()
        df_sub['Features'] = feature
        df_sub.rename(columns={feature: y_label}, inplace=True)
        # print(df_sub.head())
        df_bar.append(df_sub)
    df_bar = pd.concat(df_bar, ignore_index=True)
    # print("before: {}".format(df_bar.shape))
    print(df_bar.head())
    # df_bar = df_bar[(df_bar['z-score']<=5) & (df_bar['z-score']>=-5)]
    print(" after: {}".format(df_bar.shape))

    plt.figure(figsize=(width, high))
    ax = sns.barplot(hue="final_cluster", hue_order=hue_order, y=y_label, x="Features", data=df_bar,
                     capsize=.2, errorbar=("ci", 95), err_kws={"color": ".8", "linewidth": 0.6},
                palette=palette, width=.5, gap=0.1) # errorbar=None
    ax.tick_params(axis='x', labelrotation=0, labelsize=6)
    ax.tick_params(axis='y', labelrotation=0, labelsize=6)
    plt.legend(loc='upper center', bbox_to_anchor=(1.47, 1.02))  # 使用这行代码把图例放到图外。
    plt.savefig(img_path + "Bar_{}.png".format(file_name), dpi=720)
    plt.clf()


def plot_violin(df, file_name, feature_lst, width, high, hue_order, y_label):
    df_violin = []
    for feature in feature_lst:
        df_sub = df[[feature, 'final_cluster']].copy()
        df_sub['Features'] = feature
        df_sub.rename(columns={feature: y_label}, inplace=True)
        # print(df_sub.head())
        df_violin.append(df_sub)
    df_violin = pd.concat(df_violin, ignore_index=True)
    # print("before: {}".format(df_violin.shape))
    print(df_violin.head())

    plt.figure(figsize=(width, high))
    ax = sns.violinplot(hue="final_cluster", hue_order=hue_order, y=y_label, x="Features", data=df_violin,
                palette=palette, width=.9, gap=0.1) # errorbar=None
    ax.tick_params(axis='x', labelrotation=0, labelsize=8)
    ax.tick_params(axis='y', labelrotation=0, labelsize=8)
    plt.legend(loc='upper center', bbox_to_anchor=(1.47, 1.02))  # 使用这行代码把图例放到图外。
    plt.savefig(img_path + "Violin_{}.png".format(file_name), dpi=720)
    plt.clf()


def cnt_OR_CI95(f_list):
    for f_name in f_list:
        z = norm.ppf(0.975)  # 对应于95%置信水平的z值
        df = pd.read_csv(result_path+'{}.csv'.format(f_name))
        # df['clust1_OR'] = df.apply(lambda row: np.exp(row['clust1_Estimate']), axis=1)
        df['clust1_lower95'] = df.apply(lambda row: row['clust1_Estimate']-z*row['clust1_StdError'], axis=1)
        df['clust1_upper95'] = df.apply(lambda row: row['clust1_Estimate']+z*row['clust1_StdError'], axis=1)
        # df['clust1'] = 'clust1'

        # df['clust2_OR'] = df.apply(lambda row: np.exp(row['clust2_Estimate']), axis=1)
        df['clust2_lower95'] = df.apply(lambda row: row['clust2_Estimate']-z*row['clust2_StdError'], axis=1)
        df['clust2_upper95'] = df.apply(lambda row: row['clust2_Estimate']+z*row['clust2_StdError'], axis=1)
        # df['clust2'] = 'clust2'

        # df_rst = pd.DataFrame({'Feature': df['Col'].tolist()+df['Col'].tolist(),
        #                        'OR': df['clust1_OR'].tolist()+df['clust2_OR'].tolist(),
        #                        'CI_lower': df['clust1_lower95'].tolist()+df['clust2_lower95'].tolist(),
        #                        'CI_upper': df['clust1_upper95'].tolist()+df['clust2_upper95'].tolist(),
        #                        'Clust': df['clust1'].tolist()+df['clust2'].tolist()})
        df.to_csv(result_path+'{}.csv'.format(f_name), index=False)


if __name__ == '__main__':
    # 创建一个示例dataframe

    # rst = set(pd.read_csv(path + "proteomics.csv").columns.tolist())-set(Prote_cols_all)
    # print(rst)

    # df1 = pd.read_excel(path+'protein2923.xlsx', sheet_name='Sheet1', usecols=['coding', 'meaning'])
    # df2 = pd.read_excel(path+'protein2923.xlsx', sheet_name='Sheet2', usecols=['Assay Target', 'Protein panel'])
    # df = pd.merge(df1, df2, left_on='coding', right_on='Assay Target', how='inner')
    # df.to_csv(path + "protein2923.csv", index=None)
    # print(df.shape[0])

    print(len(Prote_cols_all))
    # cnt_OR_CI95(['LR_PRS_rst', 'LR_PRS_with_Sex_Age_rst'])
    cnt_OR_CI95(['LR_MRI_MD_with_Sex_Age_rst', 'LR_MRI_Volume_with_Sex_Age_rst', 'LR_MRI_Thickness_with_Sex_Age_rst',
                 'LR_MRI_FA_with_Sex_Age_rst', 'LR_MRI_subCortex_with_Sex_Age_rst', 'LR_MRI_subCortex_rst',
                 'LR_MRI_Area_with_Sex_Age_rst', 'LR_MRI_FA_rst', 'LR_MRI_MD_rst', 'LR_MRI_Volume_rst',
                 'LR_MRI_Thickness_rst', 'LR_MRI_Area_rst'])

    pass
