# DeConveil

DeConveil is a computational method for the Differential Gene Expression testing that explicitly integrates gene Copy Number data into statistical framework making statistical adjustments to traditional methods.
This approach adjusts for dosage effects and categorizes genes as *dosage-sensitive (DSG)*, *dosage-insensitive (DIG)*, and *dosage-compensated (DCG)*.

**Pre-required installations before running DeConveil** 
Python libraries are required to be installed: *pydeseq2*

**How to install DeConveil**

`git clone https://github.com/Katerina10-cloud/DeConveil.git`


**Input data**
DeConveil requires two matched input matrices: mRNA read counts (normal and tumor samples) and absolute CN values (for normal diploid samples we assign CN 2). Example of CN data: 1,2,3,4,5,6. Each value of CN we divide by 2: CN/2. Example of input data is shown in *test_deconveil* Jupyter Notebook.


**Output data**
`res_CNnaive.csv` and `res_CNaware.csv` data frames reporting *LogFC* and *p-value* for both methods.
These data are further processed to separate gene groups using `defene_gene_groups()` function.



### References

1. Michael I Love, Wolfgang Huber, and Simon Anders. Moderated estimation of fold change and dispersion for rna-seq data with deseq2. Genome biology, 15(12):1–21, 2014. doi:10.1186/s13059-014-0550-8.

2. Boris Muzellec, Maria Telenczuk, Vincent Cabeli, and Mathieu Andreux. Pydeseq2: a python package for bulk rna-seq differential expression analysis. bioRxiv, pages 2022–12, 2022. doi:10.1101/2022.12.14.520412.

3. Anqi Zhu, Joseph G Ibrahim, and Michael I Love. Heavy-tailed prior distributions for sequence count data: removing the noise and preserving large differences. Bioinformatics, 35(12):2084–2092, 2019. doi:10.1093/bioinformatics/bty895.
