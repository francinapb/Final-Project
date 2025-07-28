# Promoter Prediction in *Pseudomonas putida* using Retrained RF-HOT Model

This repository extends the Promotech framework to retrain the RF-HOT model for improved promoter prediction adding a new dataset from *Pseudomonas putida* KT2440. It integrates curated multi-species datasets (positive dataset from validated promoter sequences extracted from the Prokaryotic Promoter Database (PPD) and negative dataset from the coding DNA sequences (CDS)), one-hot encoded training and test sequences, and evaluation scripts to produce a more accurate and interpretable model for synthetic biology and regulatory genomics applications.

## Requirements
1. Set up and activation environment as in the original Promotech repository: https://github.com/BioinformaticsLabAtMUN/Promotech

    conda env create -f promotech_env.yml for Ubuntu 20.04
   
    conda env create -f promotech_mac_env.yml for Mac OS Big Sur V11.3
   
    conda activate promotech_env

## Commands used
1. One-hot encode sequences
   
    <pre>  python sone_hot_encoding.py  </pre>

2. Retrained model
   
    <pre>  python train_rfhot.py  </pre>

3. Create test set
   
    <pre>  python create_test_set.py  </pre>

4. Compare models
   
    <pre>  python compare_models.py  </pre>

Results saved in the results/ folder

5. Sequences prediction with the new model

    <pre>  python promotech.py -s -m RF-HOT-CUSTOM -f examples/sequences/test.fasta -o results  <pre>
        

## References

Promotech original repository: https://github.com/BioinformaticsLabAtMUN/Promotech

R. Chevez-Guardado and L. Peña-Castillo, “Promotech: a general tool for bacterial promoter recognition,” Genome Biology, vol. 22, no. 1, pp. 1–16, Dec. 2021, doi: 10.1186/S13059-021-02514-9/TABLES/10.
  
W. Su et al., “PPD: A Manually Curated Database for Experimentally Verified Prokaryotic Promoters,” Journal of Molecular Biology, vol. 433, no. 11, p. 166860, May 2021, doi: 10.1016/J.JMB.2021.166860.  

Biopython: https://biopython.org/

Scikit-learn: https://scikit-learn.org/stable/

BEDTools, VSEARCH, BLASTn: used in the data generation for *Pseudomonas putida*

## Citation
Francina Parera-Bordoy. _Optimising promoter finding software for use in pathogenic bacteria_. University of Galway, 2025.


