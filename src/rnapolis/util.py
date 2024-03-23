import gzip
import os
import tempfile
import numpy as np
from pandas import DataFrame
from typing import IO


def handle_input_file(path) -> IO[str]:
    root, ext = os.path.splitext(path)

    if ext == ".gz":
        root, ext = os.path.splitext(root)
        file = tempfile.NamedTemporaryFile("wt+", suffix=ext)
        with gzip.open(path, "rt") as f:
            file.write(f.read())
            file.seek(0)
    else:
        file = tempfile.NamedTemporaryFile("wt+", suffix=ext)
        with open(path) as f:
            file.write(f.read())
            file.seek(0)
    return file


def pairwise_distances(df: DataFrame, num_bins: int) -> list:
    coords = df[['x', 'y', 'z']].values
    distances = np.sqrt(np.sum((coords[:, np.newaxis, :] - coords[np.newaxis, :, :]) ** 2, axis=-1))
    upper_triangle_indices = np.triu_indices_from(distances, k=1)
    flattened_distances = distances[upper_triangle_indices]
    histogram, bin_edges = np.histogram(flattened_distances, bins=num_bins, density=True)
    normalized_histogram = histogram / np.sum(histogram)
    return normalized_histogram, bin_edges

def plannar_angles(num_bins: int) -> list:
    pass

def torsion_angles(num_bins: int) -> list:
    pass

def is_correct_according_to_rnaview(c2 = None, c6 = None, n1 = None):
    # Convert tuples to numpy arrays for vector operations
    c2 = np.array(c2)
    c6 = np.array(c6)
    n1 = np.array(n1)

    # Calculate distances using numpy's linear algebra norm function
    d1 = np.linalg.norm(c2 - c6)
    d2 = np.linalg.norm(n1 - c6)
    d3 = np.linalg.norm(n1 - c2)

    return (d1 <= 3.0 and
            d2 <= 2.0 and
            d3 <= 2.0)