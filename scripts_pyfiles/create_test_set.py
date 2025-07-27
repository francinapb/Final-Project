import numpy as np
from Bio import SeqIO
import os

def one_hot_encode(seq):
    mapping = {'A': [1,0,0,0], 
    	       'C': [0,1,0,0], 
    	       'G': [0,0,1,0], 
    	       'T': [0,0,0,1], 
    	       'N': [0,0,0,0]}
    return np.array([mapping.get(base.upper(), [0,0,0,0]) for base in seq])

def process_fasta_dir(fasta_dir):
    X, y = [], []

    for species in os.listdir(fasta_dir):
        sp_dir = os.path.join(fasta_dir, species)
        pos_path = os.path.join(sp_dir, "positive.fasta")
        neg_path = os.path.join(sp_dir, "negative.fasta")

        if os.path.exists(pos_path):
            for record in SeqIO.parse(pos_path, "fasta"):
                if len(record.seq) == 40:
                    X.append(one_hot_encode(record.seq))
                    y.append(1)

        if os.path.exists(neg_path):
            for record in SeqIO.parse(neg_path, "fasta"):
                if len(record.seq) == 40:
                    X.append(one_hot_encode(record.seq))
                    y.append(0)

    X = np.array(X).reshape(len(X), -1)
    y = np.array(y)

    return X, y


X_test, y_test = process_fasta_dir("../datasets/validation")


out_dir = "../combined_data"
os.makedirs(out_dir, exist_ok=True)


np.save(os.path.join(out_dir, "test_X.npy"), X_test)
np.save(os.path.join(out_dir, "test_y.npy"), y_test)

print(f"Test set created: {X_test.shape[0]} samples, {X_test.shape[1]} features")
print(f"Files saved to: {out_dir}/test_X.npy and test_y.npy")

