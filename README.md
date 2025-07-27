# Promoter Prediction in *Pseudomonas putida* using Retrained RF-HOT Model

This repository contains all scripts, data references, and instructions required to retrain and evaluate the **RF-HOT** model from the [Promotech framework](https://github.com/Gabaldonlab/promotech) for promoter prediction in *Pseudomonas putida* KT2440.

## Overview

The objective of this project is to refine promoter prediction by integrating validated transcription start sites (TSS) from *P. putida* into an existing multi-species training set. The model is retrained using one-hot encoded sequences and evaluated on an external validation set to assess performance.

---

## Repository Structure

```bash
├── data/
│   ├── positive/                # FASTA files of known promoters (40 bp upstream of TSS)
│   ├── negative/                # FASTA files of negative samples (from CDS regions)
│   ├── test_set/                # One-hot encoded test data (test_X.npy, test_y.npy)
├── scripts/
│   ├── one_hot_encoding.py      # Script to convert sequences into one-hot encoded NumPy arrays
│   ├── create_test_set.py       # Script to generate and encode the independent test set
│   ├── compare_models.py        # Script to evaluate and compare original and retrained models
│   └── rfhot_train.py           # (Optional) Standalone training script for RF-HOT retraining
├── models/
│   ├── RF-HOT.model             # Original Promotech model
│   ├── RF-HOT-CUSTOM.model      # Retrained model including *P. putida*
├── results/
│   ├── rfhot_10fold_results.txt # Cross-validation metrics for retrained model
│   ├── evaluation_metrics.csv   # Final test set performance metrics
│   ├── plots/                   # ROC, PR curves, and confusion matrices
└── README.md
