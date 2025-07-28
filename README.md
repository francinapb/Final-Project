# Promoter Prediction in *Pseudomonas putida* using Retrained RF-HOT Model

This repository extends the Promotech framework to retrain the RF-HOT model for improved promoter prediction in *Pseudomonas putida* KT2440. It integrates curated multi-species datasets, one-hot encoded training and test sequences, and evaluation scripts to produce a more accurate and interpretable model for synthetic biology and regulatory genomics applications.

## Requirements
1. Set up and activation environment as in the original Promotech repository: https://github.com/BioinformaticsLabAtMUN/Promotech

    conda env create -f promotech_env.yml for Ubuntu 20.04
   
    conda env create -f promotech_mac_env.yml for Mac OS Big Sur V11.3
   
    conda activate promotech_env

## Commands
1. One-hot encode sequences
   
    <pre>  python sone_hot_encoding.py  </pre>

3. Retrained model
   
    <pre>  python train_rfhot.py  </pre>

5. Create test set
   
    <pre>  python create_test_set.py  </pre>

7. Compare models
   
    <pre>  python compare_models.py  </pre>

Results saved in the results/ folder


## References

Promotech original repository: https://github.com/BioinformaticsLabAtMUN/Promotech

Promotech: A general tool for bacterial promoter recognition. Ruben Chevez-Guardado and Lourdes Peña-Castillo. Genome Biol 22(1):318 (2021). PMID: 34789306. [DOI: 10.1186/s13059-021-02514-9 ] (https://doi.org/10.1186/s13059-021-02514-9)

Biopython: https://biopython.org/

Scikit-learn: https://scikit-learn.org/stable/

BEDTools, VSEARCH, BLASTn: used in the data generation for *Pseudomonas putida*

## Citation
Francina Parera-Bordoy. _Optimising promoter finding software for use in pathogenic bacteria_. University of Galway, 2025.

---

## Repository Structure

```bash
├── benchmark/                        # Scripts from original Promotech to benchmark performance
│   ├── __pycache__/
│   │   ├── ipromoter2l.cpython-36.pyc
│   │   ├── ipromoter2ldatabase.cpython-36.pyc
│   │   └── process_benchmark.cpython-36.pyc
│   ├── ipromoter2l.py
│   ├── ipromoter2ldatabase.py
│   └── process_benchmark.py

├── combined_data/                   # Encoded training and test arrays
│   ├── combined_X.npy
│   ├── combined_y.npy
│   ├── test_X.npy
│   └── test_y.npy

├── core/                            # Core Promotech framework scripts
│   ├── __pycache__/
│   │   ├── database.cpython-36.pyc
│   │   └── utils.cpython-36.pyc
│   ├── base.py
│   ├── data.py
│   ├── database.py
│   ├── main.py
│   ├── models.py
│   ├── nextflow
│   ├── nextflow.config
│   ├── pipeline.nf
│   ├── pipeline_unbalanced.nf
│   └── utils.py

├── dataset/                         # Training and validation datasets
│   ├── training/
│   │   ├── CPNEUMONIAE/
│   │   ├── ECOLI/
│   │   ├── ECOLI_2/
│   │   ├── HPYLORI/
│   │   ├── HPYLORI_2/
│   │   ├── LINTERROGANS/
│   │   ├── PPUTIDA/
│   │   ├── SCOELICOLOR/
│   │   ├── SONEIDENSIS/
│   │   ├── SPYOGENE/
│   │   └── STYPHIRMURIUM/
│   │       # Each contains negative.fasta and positive.fasta
│   └── validation/
│       ├── BACILLUS/
│       ├── CLOSTRIDIUM/
│       ├── MYCOBACTER/
│       ├── RHODOBACTER_1/
│       └── RHODOBACTER_2/
│           # Each contains negative.fasta and positive.fasta

├── examples/                        # Sample sequences and logs from Promotech
│   └── sequences/
│       ├── example.log.txt
│       └── test.fasta

├── genome/                          # Genome processing script
│   ├── __pycache__/
│   │   └── process_genome.cpython-36.pyc
│   └── process_genome.py

├── models/                          # Original and retrained models
│   ├── GRU-0.h5
│   ├── GRU-1.h5
│   ├── LSTM-3.h5
│   ├── LSTM-4.h5
│   ├── RF-HOT.model                # Original Promotech RF-HOT model
│   ├── RF-TETRA.model
│   ├── tokenizer.data
│   └── RF-HOT-CUSTOM.model         # Retrained model with P. putida

├── promotech_env.yml               # Conda environment files
├── promotech_mac_env.yml
├── promotech.py

├── results_pyfiles/                        # Output plots and tables
│   ├── ROC and PR curves
│   ├── confusion matrix plots
│   └── evaluation_metrics.csv

├── scripts/                        # Custom scripts for encoding, training, and evaluation
│   ├── compare_models.py
│   ├── create_test_set.py
│   ├── one_hot_encoding.py
│   └── train_rfhot.py

├── sequences/                      # Sequence preprocessing
│   ├── __pycache__/
│   │   └── process_sequences.cpython-36.pyc
│   └── process_sequences.py

├── ui/                             # Promotech GUI resources
│   ├── __pycache__/
│   │   └── GUI.cpython-36.pyc
│   ├── form.ui
│   ├── GUI.py
│   ├── main.pyproject
│   ├── main.pyproject.user
│   ├── main.qml
│   └── ui_form-h

---
