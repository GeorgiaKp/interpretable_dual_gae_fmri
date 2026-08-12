import os
import numpy as np

# --- Settings ---
embedding_dir = '/content/drive/MyDrive/abide_dataset/ABIDE_pcp/ccs/filt_noglobal/roi_embeddings'            # Directory with raw embeddings (.npy files)
output_dir = '/content/drive/MyDrive/abide_dataset/ABIDE_pcp/ccs/filt_noglobal/roi_embeddings_normalized_datawide'    # Directory to save normalized embeddings
os.makedirs(output_dir, exist_ok=True)

# --- Step 1: Load all embeddings ---
embeddings = []
subject_ids = []

print(f"Loading embeddings from {embedding_dir}...")
for file in os.listdir(embedding_dir):
    if file.endswith('.npy'):
        filepath = os.path.join(embedding_dir, file)
        emb = np.load(filepath)               # Expected shape: [111, D]  (e.g., [111, 96])
        embeddings.append(emb)
        subject_ids.append(file)

embeddings = np.stack(embeddings, axis=0)     # Shape: [N, 111, D]
print(f"✔ Loaded {len(embeddings)} embeddings. Shape: {embeddings.shape}")

# --- Step 2: Dataset-wide Normalization ---
print("⚙️ Performing dataset-wide normalization (z-score)...")
mean = embeddings.mean(axis=(0, 1), keepdims=True)   # Shape: [1, 1, D]
std = embeddings.std(axis=(0, 1), keepdims=True) + 1e-8

normalized_embeddings = (embeddings - mean) / std

# --- Step 3: Save normalized embeddings ---
print(f"Saving normalized embeddings to {output_dir}...")
for subj_id, norm_emb in zip(subject_ids, normalized_embeddings):
    output_path = os.path.join(output_dir, subj_id)
    np.save(output_path, norm_emb)

print(f"\nDone. Normalized embeddings saved to: {output_dir}")
