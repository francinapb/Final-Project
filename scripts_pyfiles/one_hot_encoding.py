import os
import numpy as np
from Bio import SeqIO

def one_hot_encode(seq):
    mapping = {'A':[1,0,0,0], 
    	       'C':[0,1,0,0], 
    	       'G':[0,0,1,0], 
    	       'T':[0,0,0,1], 
    	       'N':[0,0,0,0]}
    return np.array([mapping.get(base.upper(), [0,0,0,0]) for base in seq])

def collect_and_encode_sequences(dataset_root, out_dir="../combined_data"):
    X, y = [], []

    species_dirs = sorted([
        os.path.join(dataset_root, d)
        for d in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, d))
    ])

    for species in species_dirs:
        pos_path = os.path.join(species, "positive.fasta")
        neg_path = os.path.join(species, "negative.fasta")

        if os.path.exists(pos_path):
            for record in SeqIO.parse(pos_path, "fasta"):
                if len(record.seq) == 40:
                    X.append(one_hot_encode(str(record.seq)))
                    y.append(1)

        if os.path.exists(neg_path):
            for record in SeqIO.parse(neg_path, "fasta"):
                if len(record.seq) == 40:
                    X.append(one_hot_encode(str(record.seq)))
                    y.append(0)

    X = np.array(X).reshape(len(X), -1)
    y = np.array(y)

    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "combined_X.npy"), X)
    np.save(os.path.join(out_dir, "combined_y.npy"), y)

    print(f"Combined dataset created: {X.shape[0]} sequences, {X.shape[1]} features.")
    print(f"Saved to {out_dir}/combined_X.npy and combined_y.npy")

if __name__ == "__main__":
    collect_and_encode_sequences(dataset_root="../datasets/training", out_dir="../combined_data")

