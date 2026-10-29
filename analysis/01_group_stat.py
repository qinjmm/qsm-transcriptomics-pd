"""Regional group comparisons for one imaging measurement.

For every selected region, the script fits:

    imaging measurement ~ group + covariates

Group is coded as 0 = healthy control and 1 = Parkinson's disease.
FDR correction is applied across the selected regions.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


def cohen_d(data, outcome, group_col="group"):
    """Calculate unadjusted Cohen's d as PD minus HC."""
    hc = data.loc[data[group_col] == 0, outcome].dropna()
    pd_group = data.loc[data[group_col] == 1, outcome].dropna()

    if len(hc) < 2 or len(pd_group) < 2:
        return np.nan

    pooled_sd = np.sqrt(
        ((len(hc) - 1) * hc.var() + (len(pd_group) - 1) * pd_group.var())
        / (len(hc) + len(pd_group) - 2)
    )
    if pooled_sd == 0:
        return np.nan

    return (pd_group.mean() - hc.mean()) / pooled_sd


def analyze_regions(
    data,
    region_columns,
    measurement,
    covariates=("age", "sex"),
    group_col="group",
):
    """Fit one OLS model for each imaging."""
    formula_terms = " + ".join([group_col, *covariates])
    results = []

    for region in region_columns:
        if region not in data.columns or data[region].notna().sum() < 3:
            continue

        try:
            model = smf.ols(f"Q({region!r}) ~ {formula_terms}", data=data).fit()
            results.append(
                {
                    "region": region,
                    "measurement": measurement,
                    "n": int(model.nobs),
                    "group_beta": model.params[group_col],
                    "group_t": model.tvalues[group_col],
                    "group_p": model.pvalues[group_col],
                    "cohen_d": cohen_d(data, region, group_col),
                }
            )
        except Exception as error:
            print(f"Model failed for {region}: {error}")

    return pd.DataFrame(results)


def correct_p_values(results):
    output = results.copy()
    valid = output["group_p"].notna()
    p_values = output.loc[valid, "group_p"]

    reject_fdr, p_fdr, _, _ = multipletests(p_values, method="fdr_bh")

    output["p_fdr"] = np.nan
    output["fdr_significant"] = False

    output.loc[valid, "p_fdr"] = p_fdr
    output.loc[valid, "fdr_significant"] = reject_fdr
    return output


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--measurement",
        required=True,
        help="Name used to identify and label the imaging measurement",
    )
    parser.add_argument("--group-col", default="group")
    parser.add_argument("--covariates", nargs="+", default=["age", "sex"])
    parser.add_argument("--encoding", default="utf-8")
    return parser.parse_args()


def main():
    args = parse_args()
    data = pd.read_csv(args.input, encoding=args.encoding)
    regions = [column for column in data.columns if args.measurement in column]

    if not regions:
        raise ValueError(f"No columns contain measurement: {args.measurement}")

    results = analyze_regions(
        data,
        regions,
        measurement=args.measurement,
        covariates=args.covariates,
        group_col=args.group_col,
    )
    if results.empty:
        raise ValueError("No regional model could be fitted")
    results = correct_p_values(results)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    print(f"Analyzed {len(results)} regions for {args.measurement}")


if __name__ == "__main__":
    main()
