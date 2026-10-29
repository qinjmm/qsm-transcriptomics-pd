"""Cell-type and cortical-layer marker gene analysis."""

import numpy as np
import pandas as pd
from tqdm import tqdm


thresholds = [2.5, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
n_permutations = 1000


weights = ...
expression = ...
marker_data = ...

weights.index = weights.index.str.upper()
expression.columns = expression.columns.str.upper()
marker_data["gene.symbol"] = marker_data["gene.symbol"].str.upper()

marker_columns = [
    "Layer 1",
    "Layer 2",
    "Layer 3",
    "Layer 4",
    "Layer 5",
    "Layer 6",
    "Cell Ex",
    "Cell In",
    "Cell Ast",
    "Cell End",
    "Cell Mic",
    "Cell OPC",
    "Cell Oli",
]


marker_sets = {}
for category in marker_columns:
    marker_sets[category] = marker_data.loc[
        marker_data[category] == True, "gene.symbol"
    ].dropna()


marker_expression = {}
for category, marker_genes in marker_sets.items():
    genes = expression.columns.intersection(marker_genes)
    marker_expression[category] = expression[genes].mean(axis=1)

marker_expression = pd.DataFrame(marker_expression)

all_genes = weights.index.to_numpy()
rng = np.random.default_rng()
results = []


for threshold in tqdm(thresholds, desc="Permutation by threshold"):
    n_selected = int(len(weights) * threshold / 100)
    positive_genes = weights.nlargest(n_selected).index.to_numpy()
    negative_genes = weights.nsmallest(n_selected).index.to_numpy()

    for direction, selected_genes in (
        ("positive", positive_genes),
        ("negative", negative_genes),
    ):
        for category, marker_genes in marker_sets.items():
            marker_genes = np.intersect1d(marker_genes, all_genes)

            observed_ratio = np.isin(selected_genes, marker_genes).mean()

            null_ratios = []
            for _ in range(n_permutations):
                random_genes = rng.choice(all_genes, n_selected, replace=False)
                random_ratio = np.isin(random_genes, marker_genes).mean()
                null_ratios.append(random_ratio)

            mean_null_ratio = np.mean(null_ratios)
            enrichment = observed_ratio - mean_null_ratio

            results.append(
                {
                    "threshold": threshold,
                    "category": category,
                    "direction": direction,
                    "enrichment": enrichment,
                }
            )


results = pd.DataFrame(results)
