import os
import warnings
from math import ceil, floor
from pathlib import Path

import numpy as np
import pandas as pd
import deconveil

from typing import List, Literal, Optional, Dict, Any, cast



def load_test_data(
    modality: Literal["rna", "cnv", "metadata", "cnv_tumor"] = "rna",
    dataset: Literal["tcga_brca"] = "tcga_brca",
    debug: bool = False,
    debug_seed: int = 42,
) -> pd.DataFrame:
    """Load TCGA-BRCA example data from the DeConveil package.

    Parameters
    ----------
    modality : {"rna", "cnv", "metadata", "cnv_tumor"}
        Type of data to load.

    dataset : {"tcga_brca"}
        Dataset name. Only "tcga_brca" is currently supported.

    debug : bool, optional
        If True, randomly subsample 10 samples and 100 features (if applicable).
        Default is False.

    debug_seed : int, optional
        Random seed for reproducibility of debug subsampling. Default is 42.

    Returns
    -------
    pandas.DataFrame
        The requested data modality as a DataFrame.
    """
    assert modality in ["rna", "cnv", "metadata", "cnv_tumor"], (
        "modality must be one of: 'rna', 'cnv', 'metadata', 'cnv_tumor'"
    )
    assert dataset in ["tcga_brca"], (
        "dataset must be one of: 'tcga_brca'"
    )

    # Locate data within the package
    datasets_path = Path(__file__).resolve().parent.parent / "datasets" / dataset

    # Construct file paths
    file_map = {
        "rna": datasets_path / "rna.csv",
        "cnv": datasets_path / "cnv.csv",
        "metadata": datasets_path / "metadata.csv",
        "cnv_tumor": datasets_path / "cnv_tumor.csv",
    }

    data_path = file_map[modality]
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # Load the CSV
    df = pd.read_csv(data_path, index_col=0)

    # Apply debug mode subsampling
    if debug:
        df = df.sample(n=min(10, df.shape[0]), random_state=debug_seed)
        if modality in ["rna", "cnv"]:
            df = df.sample(n=min(100, df.shape[1]), axis=1, random_state=debug_seed)

    return df


def replace_underscores(factors: List[str]):
    """Replace all underscores from strings in a list by hyphens.

    To be used on design factors to avoid bugs due to the reliance on
    ``str.split("_")`` in parts of the code.

    Parameters
    ----------
    factors : list
        A list of strings which may contain underscores.

    Returns
    -------
    list
        A list of strings in which underscores were replaced by hyphens.
    """
    return [factor.replace("_", "-") for factor in factors]


def filter_low_count_genes(
    df: pd.DataFrame,
    other_dfs: Optional[List[pd.DataFrame]] = None,
    min_count: int = 10,
    min_samples: Optional[int] = 3,
    min_frac: Optional[float] = None,
    return_mask: bool = False
) -> Dict[str, Any]:
    """
    Filter genes (columns) by expression thresholds.

    Parameters
    ----------
    df : pd.DataFrame
        Main dataframe (genes as columns, samples as rows).
    other_dfs : list of pd.DataFrame, optional
        Other dataframes with the same columns to filter in parallel.
    min_count : int, default=10
        Minimum expression/count threshold.
    min_samples : int, default=3
        Minimum number of samples meeting the threshold.
    min_frac : float, optional
        Fraction of samples that must meet the threshold.
        If provided, overrides min_samples.
    return_mask : bool, default=False
        If True, also return the boolean mask of kept genes.

    Returns
    -------
    result : dict
        {
            "filtered_df": pd.DataFrame,
            "other_filtered": list[pd.DataFrame] or None,
            "mask": pd.Series (if return_mask),
            "stats": dict with counts
        }
    """
    # compute required min_samples
    if min_frac is not None:
        min_samples = max(1, int(round(min_frac * df.shape[0])))

    # gene-wise filter mask
    mask = (df >= min_count).sum(axis=0) >= min_samples

    # apply mask
    filtered_df = df.loc[:, mask]
    filtered_others = [odf.loc[:, mask] for odf in other_dfs] if other_dfs else None

    # collect stats
    stats = {
        "n_total": df.shape[1],
        "n_kept": int(mask.sum()),
        "n_removed": int((~mask).sum()),
        "min_count": min_count,
        "min_samples": min_samples,
    }

    result = {
        "filtered_df": filtered_df,
        "other_filtered": filtered_others,
        "stats": stats,
    }
    if return_mask:
        result["mask"] = mask

    return result


def process_results(file_path, method, lfc_cut = 1.0, pval_cut = 0.05):
    df = pd.read_csv(file_path, index_col=0)
    df['isDE'] = (np.abs(df['log2FoldChange']) >= lfc_cut) & (df['padj'] <= pval_cut)
    df['DEtype'] = np.where(
        ~df['isDE'], 
        "n.s.", 
        np.where(df['log2FoldChange'] > 0, "Up-reg", "Down-reg")
    )
    df['method'] = method
    return df[['log2FoldChange', 'padj', 'isDE', 'DEtype', 'method']]
    

def define_gene_groups(res_joint):
    DSGs = res_joint[
        ((res_joint['DEtype_naive'] == "Up-reg") & (res_joint['DEtype_aware'] == "n.s.")) |
        ((res_joint['DEtype_naive'] == "Down-reg") & (res_joint['DEtype_aware'] == "n.s."))
    ].assign(gene_category='DSGs')
    
    DIGs = res_joint[
        ((res_joint['DEtype_naive'] == "Up-reg") & (res_joint['DEtype_aware'] == "Up-reg")) |
        ((res_joint['DEtype_naive'] == "Down-reg") & (res_joint['DEtype_aware'] == "Down-reg"))
    ].assign(gene_category='DIGs')
             
    DCGs = res_joint[
        ((res_joint['DEtype_naive'] == "n.s.") & (res_joint['DEtype_aware'] == "Up-reg")) |
        ((res_joint['DEtype_naive'] == "n.s.") & (res_joint['DEtype_aware'] == "Down-reg"))
    ].assign(gene_category='DCGs')
             
    non_DEGs = res_joint[
        (res_joint['DEtype_naive'] == "n.s.") & (res_joint['DEtype_aware'] == "n.s.")
    ].assign(gene_category='non-DEGs')
             
    return {
        "DSGs": DSGs,
        "DIGs": DIGs,
        "DCGs": DCGs,
        "non_DEGs": non_DEGs
    }


def clean_gene_group(df, mode="naive"):
    """Rename and subset a gene group dataframe for a given mode."""
    suffix = f"_{mode}"
    rename_map = {
        f"logFC{suffix}": "log2FC",
        f"padj{suffix}": "padj",
        f"isDE{suffix}": "isDE",
        f"DEtype{suffix}": "DEtype",
        f"method{suffix}": "method",
        "gene_category": "gene_group"
    }
    return df.rename(columns=rename_map)[["log2FC", "padj", "isDE", "DEtype", "method", "gene_group"]]