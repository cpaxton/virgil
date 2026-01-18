# Copyright 2024 Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
from typing import List, Tuple, Optional
from sklearn.decomposition import PCA


def cosine_distance(emb1: torch.Tensor, emb2: torch.Tensor) -> float:
    """
    Compute cosine distance between two embeddings.

    Cosine distance = 1 - cosine similarity
    For normalized embeddings, this is equivalent to 1 - dot product.

    Args:
        emb1: First embedding tensor (normalized)
        emb2: Second embedding tensor (normalized)

    Returns:
        float: Cosine distance between embeddings
    """
    if isinstance(emb1, torch.Tensor):
        emb1 = emb1.cpu().numpy()
    if isinstance(emb2, torch.Tensor):
        emb2 = emb2.cpu().numpy()

    # For normalized embeddings, cosine similarity is just dot product
    similarity = np.dot(emb1.flatten(), emb2.flatten())
    distance = 1.0 - similarity
    return float(distance)


def compute_pairwise_distances(
    embeddings: torch.Tensor, labels: List[str]
) -> Tuple[np.ndarray, List[Tuple[str, str]]]:
    """
    Compute pairwise cosine distances between all embeddings.

    Args:
        embeddings: Tensor of shape (n, embedding_dim) with normalized embeddings
        labels: List of n labels corresponding to each embedding

    Returns:
        Tuple of (distance_matrix, label_pairs) where:
        - distance_matrix: (n, n) array of distances
        - label_pairs: List of (label1, label2) tuples for each pair
    """
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()

    n = len(embeddings)
    distance_matrix = np.zeros((n, n))
    label_pairs = []

    for i in range(n):
        for j in range(n):
            if i == j:
                distance = 0.0
            else:
                # Cosine distance for normalized embeddings
                similarity = np.dot(embeddings[i], embeddings[j])
                distance = 1.0 - similarity
            distance_matrix[i, j] = distance
            label_pairs.append((labels[i], labels[j]))

    return distance_matrix, label_pairs


def compute_pca(
    embeddings: torch.Tensor, n_components: int = 2
) -> Tuple[np.ndarray, PCA]:
    """
    Compute PCA reduction of embeddings.

    Args:
        embeddings: Tensor of shape (n, embedding_dim)
        n_components: Number of principal components to compute

    Returns:
        Tuple of (reduced_embeddings, pca_model) where:
        - reduced_embeddings: (n, n_components) array
        - pca_model: Fitted PCA model
    """
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()

    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(embeddings)

    return reduced, pca


def compute_distances_to_reference(
    embeddings: torch.Tensor,
    labels: List[str],
    reference_labels: List[str],
    reference_indices: Optional[List[int]] = None,
) -> dict:
    """
    Compute distances from all embeddings to a set of reference embeddings.

    Args:
        embeddings: Tensor of shape (n, embedding_dim)
        labels: List of n labels
        reference_labels: List of reference label names
        reference_indices: Optional list of indices for reference embeddings.
                         If None, will find indices by matching labels.

    Returns:
        Dictionary mapping each label to its distances to reference labels
    """
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()

    # Find reference indices if not provided
    if reference_indices is None:
        reference_indices = [labels.index(ref) for ref in reference_labels]

    # Compute distances
    results = {}
    for i, label in enumerate(labels):
        distances = {}
        for ref_idx, ref_label in zip(reference_indices, reference_labels):
            similarity = np.dot(embeddings[i], embeddings[ref_idx])
            distance = 1.0 - similarity
            distances[ref_label] = float(distance)
        results[label] = distances

    return results
