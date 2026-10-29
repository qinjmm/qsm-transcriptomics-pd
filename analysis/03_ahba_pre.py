"""AHBA preprocessing"""

import abagen
import pandas as pd
from abagen.correct import keep_stable_genes


IBF_THRESHOLD = 0.5
DS_THRESHOLD = 0.1
TOLERANCE_MM = 2


atlas = abagen.fetch_desikan_killiany(native=False, surface=False)
atlas_info = pd.read_csv(atlas["info"])


expression = abagen.get_expression_data(
    atlas=atlas["image"],
    atlas_info=atlas["info"],
    reannotated=True,
    ibf_threshold=IBF_THRESHOLD,
    probe_selection="diff_stability",
    donor_probes="aggregate",
    lr_mirror=None,
    tolerance=TOLERANCE_MM,
    missing=None,
    sample_norm="srs",
    gene_norm="srs",
    norm_matched=True,
    norm_structures=False,
    region_agg="donors",
    agg_metric="mean",
    return_donors=True,
)


# Retain the 34 left cortical parcels
left_ids = atlas_info.loc[
    (atlas_info["hemisphere"] == "L") & (atlas_info["structure"] == "cortex"),
    "id",
].tolist()

expression_left = []
for donor_expression in expression.values():
    donor_left = donor_expression.loc[donor_expression.index.intersection(left_ids)]
    expression_left.append(donor_left)


# Retain genes with differential stability greater than 0.1
expression_stable, differential_stability = keep_stable_genes(
    expression_left,
    threshold=DS_THRESHOLD,
    percentile=False,
    return_stability=True,
)


regional_expression = pd.concat(expression_stable).groupby("label").mean()


# Save
regional_expression.to_csv("DK34_left_gene_expression_DS0.1.csv")
differential_stability.to_csv("AHBA_gene_differential_stability.csv")

print(
    f"Final matrix: {regional_expression.shape[0]} regions x "
    f"{regional_expression.shape[1]} genes"
)
