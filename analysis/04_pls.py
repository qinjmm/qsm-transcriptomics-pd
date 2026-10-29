"""PLS analysis linking AHBA expression to cortical QSM alterations."""

import numpy as np
import pandas as pd
from neuromaps.images import relabel_gifti
from neuromaps.nulls import alexander_bloch
from pyls import behavioral_pls
from scipy.stats import spearmanr, zscore


N_BOOTSTRAP = 10000
N_SPIN = 1000


def spin_vector(vector, parcellation, n_spin=N_SPIN):
    """Generate spatial null maps."""
    parcellation = relabel_gifti(parcellation, background=None)

    nulls = alexander_bloch(
        vector,
        n_perm=n_spin,
        parcellation=parcellation,
    )

    if nulls.shape[0] != n_spin:
        nulls = nulls.T

    return nulls


def run_pls(
    gene_expression,
    qsm_effect,
    gene_names,
    parcellation,
    n_bootstrap=N_BOOTSTRAP,
    n_spin=N_SPIN,
):
    X = zscore(np.asarray(gene_expression), axis=0)
    Y = zscore(np.asarray(qsm_effect).reshape(-1, 1), axis=0)

    pls = behavioral_pls(
        X,
        Y,
        n_perm=0,
        n_boot=n_bootstrap,
        seed=SEED,
    )

    brain_scores = pls["x_scores"][:, 0]
    bootstrap_ratio = pls["bootres"]["x_weights_normed"][:, 0]

    if spearmanr(brain_scores, Y.ravel()).statistic < 0:
        brain_scores = -brain_scores
        bootstrap_ratio = -bootstrap_ratio

    gene_results = pd.DataFrame(
        {
            "gene": gene_names,
            "bootstrap_ratio": bootstrap_ratio,
        }
    ).sort_values("bootstrap_ratio", ascending=False)

    qsm_nulls = spin_vector(Y.ravel(), parcellation, n_spin)
    observed_variance = pls["varexp"][0]
    null_variance = []

    for permuted_qsm in qsm_nulls:
        null_pls = behavioral_pls(
            X,
            zscore(permuted_qsm).reshape(-1, 1),
            n_perm=0,
            n_boot=0,
            seed=SEED,
        )
        null_variance.append(null_pls["varexp"][0])

    p_variance = (np.sum(np.asarray(null_variance) >= observed_variance) + 1) / (
        n_spin + 1
    )

    observed_rho = spearmanr(brain_scores, Y.ravel()).statistic
    null_rho = [
        spearmanr(brain_scores, permuted_qsm).statistic for permuted_qsm in qsm_nulls
    ]
    p_correlation = (np.sum(np.abs(null_rho) >= abs(observed_rho)) + 1) / (n_spin + 1)

    return {
        "pls": pls,
        "brain_scores": brain_scores,
        "gene_results": gene_results,
        "pls1_variance": observed_variance,
        "pls1_variance_p_spin": p_variance,
        "score_qsm_rho": observed_rho,
        "score_qsm_p_spin": p_correlation,
    }
