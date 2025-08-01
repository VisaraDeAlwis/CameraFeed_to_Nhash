import numpy as np

def load_projection_matrix(filepath):

    data = np.fromfile(filepath, dtype=np.float32)

    # Skip first 32 metadata values
    matrix_data = data[32:32 + 128 * 96]

    # Reshape to 128x96
    return matrix_data.reshape((128, 96))

def compute_hash_from_embedding(embedding_128, projection_matrix):
    projected = np.dot(embedding_128, projection_matrix)  # Result is (96,)
    binary_hash = (projected > 0).astype(int)              # Convert to 0/1
    return binary_hash
