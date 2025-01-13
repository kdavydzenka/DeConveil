import os
import multiprocessing
import warnings
from math import ceil
from math import floor
from pathlib import Path
from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import Union
from typing import cast

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.linalg import solve  # type: ignore
from scipy.optimize import minimize  # type: ignore
from scipy.special import gammaln  # type: ignore
from scipy.special import polygamma  # type: ignore
from scipy.stats import norm  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore
import matplotlib.pyplot as plt
import seaborn as sns

from deconveil.grid_search import grid_fit_beta

from pydeseq2.utils import fit_alpha_mle
from pydeseq2.utils import get_num_processes
from pydeseq2.grid_search import grid_fit_alpha
from pydeseq2.grid_search import grid_fit_shrink_beta


def irls_glm(
    counts: np.ndarray,
    cnv: np.ndarray,
    size_factors: np.ndarray,
    design_matrix: np.ndarray,
    disp: float,
    min_mu: float = 0.5,
    beta_tol: float = 1e-8,
    min_beta: float = -30,
    max_beta: float = 30,
    optimizer: Literal["BFGS", "L-BFGS-B"] = "L-BFGS-B",
    maxiter: int = 250,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:

    assert optimizer in ["BFGS", "L-BFGS-B"]
    
    num_vars = design_matrix.shape[1]
    X = design_matrix
    
    # if full rank, estimate initial betas for IRLS below
    if np.linalg.matrix_rank(X) == num_vars:
        Q, R = np.linalg.qr(X)
        y = np.log((counts / cnv) / size_factors + 0.1)
        beta_init = solve(R, Q.T @ y)
        beta = beta_init

    else:  # Initialise intercept with log base mean
        beta_init = np.zeros(num_vars)
        beta_init[0] = np.log((counts / cnv) / size_factors).mean()
        beta = beta_init
        
    dev = 1000.0
    dev_ratio = 1.0

    ridge_factor = np.diag(np.repeat(1e-6, num_vars))
    #mu = np.maximum(cnv * size_factors * np.exp(X @ beta), min_mu)
    mu = np.maximum(cnv * size_factors * np.exp(np.clip(X @ beta, -30, 30)), min_mu)
    
    converged = True
    i = 0
    while dev_ratio > beta_tol:
        W = mu / (1.0 + mu * disp)
        z = np.log((mu / cnv) / size_factors) + (counts - mu) / mu
        H = (X.T * W) @ X + ridge_factor
        beta_hat = solve(H, X.T @ (W * z), assume_a="pos")
        i += 1

        if sum(np.abs(beta_hat) > max_beta) > 0 or i >= maxiter:
            # If IRLS starts diverging, use L-BFGS-B
            def f(beta: np.ndarray) -> float:
                # closure to minimize
                #mu_ = np.maximum(cnv * size_factors * np.exp(X @ beta), min_mu)
                mu_ = np.maximum(cnv * size_factors * np.exp(np.clip(X @ beta, -30, 30)), min_mu)
                
                return nb_nll(counts, mu_, disp) + 0.5 * (ridge_factor @ beta**2).sum()

            def df(beta: np.ndarray) -> np.ndarray:
                #mu_ = np.maximum(cnv * size_factors * np.exp(X @ beta), min_mu)
                mu_ = np.maximum(cnv * size_factors * np.exp(np.clip(X @ beta, -30, 30)), min_mu)
                return (
                    -X.T @ counts
                    + ((1 / disp + counts) * mu_ / (1 / disp + mu_)) @ X
                    + ridge_factor @ beta
                )

            res = minimize(
                f,
                beta_init,
                jac=df,
                method=optimizer,
                bounds=(
                    [(min_beta, max_beta)] * num_vars
                    if optimizer == "L-BFGS-B"
                    else None
                ),
            )
            
            beta = res.x
            #mu = np.maximum(cnv * size_factors * np.exp(X @ beta), min_mu)
            mu = np.maximum(cnv * size_factors * np.exp(np.clip(X @ beta, -30, 30)), min_mu)
            converged = res.success

            #if not res.success and num_vars <= 2:
                #beta = grid_fit_beta(
                    #counts,
                    #size_factors,
                    #cnv,
                    #X,
                    #disp,
                #)
                #mu = np.maximum(cnv * size_factors * np.exp(X @ beta), min_mu)
            #break

        beta = beta_hat
        #mu = np.maximum(cnv * size_factors * np.exp(X @ beta), min_mu)
        mu = np.maximum(cnv * size_factors * np.exp(np.clip(X @ beta, -30, 30)), min_mu)
        
        # Compute deviation
        old_dev = dev
        # Replaced deviation with -2 * nll, as in the R code
        dev = -2 * nb_nll(counts, mu, disp)
        dev_ratio = np.abs(dev - old_dev) / (np.abs(dev) + 0.1)

    # Compute H diagonal (useful for Cook distance outlier filtering)
    W = mu / (1.0 + mu * disp)
    W_sq = np.sqrt(W)
    XtWX = (X.T * W) @ X + ridge_factor
    H = W_sq * np.diag(X @ np.linalg.inv(XtWX) @ X.T) * W_sq
    
    # Return an UNthresholded mu 
    # Previous quantities are estimated with a threshold though
    #mu = cnv * size_factors * np.exp(X @ beta)
    mu = np.maximum(cnv * size_factors * np.exp(np.clip(X @ beta, -30, 30)), min_mu)
    
    return beta, mu, H, converged


def fit_lin_mu(
    counts: np.ndarray,
    size_factors: np.ndarray,
    design_matrix: np.ndarray,
    min_mu: float = 0.5,
) -> np.ndarray:
    """Estimate mean of negative binomial model using a linear regression.

    Used to initialize genewise dispersion models.

    Parameters
    ----------
    counts : ndarray
        Raw counts for a given gene.

    size_factors : ndarray
        Sample-wise scaling factors (obtained from median-of-ratios).

    design_matrix : ndarray
        Design matrix.

    min_mu : float
        Lower threshold for fitted means, for numerical stability. (default: ``0.5``).

    Returns
    -------
    ndarray
        Estimated mean.
    """
    reg = LinearRegression(fit_intercept=False)
    reg.fit(design_matrix, counts / size_factors)
    mu_hat = size_factors * reg.predict(design_matrix)
    # Threshold mu_hat as 1/mu_hat will be used later on.
    return np.maximum(mu_hat, min_mu)


def fit_rough_dispersions(
    normed_counts: np.ndarray, design_matrix: pd.DataFrame
) -> np.ndarray:
    """Rough dispersion estimates from linear model, as per the R code.

    Used as initial estimates in :meth:`DeseqDataSet.fit_genewise_dispersions()
    <pydeseq2.dds.DeseqDataSet.fit_genewise_dispersions>`.

    Parameters
    ----------
    normed_counts : ndarray
        Array of deseq2-normalized read counts. Rows: samples, columns: genes.

    design_matrix : pandas.DataFrame
        A DataFrame with experiment design information (to split cohorts).
        Indexed by sample barcodes. Unexpanded, *with* intercept.

    Returns
    -------
    ndarray
        Estimated dispersion parameter for each gene.
    """
    num_samples, num_vars = design_matrix.shape
    # This method is only possible when num_samples > num_vars.
    # If this is not the case, throw an error.
    if num_samples == num_vars:
        raise ValueError(
            "The number of samples and the number of design variables are "
            "equal, i.e., there are no replicates to estimate the "
            "dispersion. Please use a design with fewer variables."
        )

    reg = LinearRegression(fit_intercept=False)
    reg.fit(design_matrix, normed_counts)
    y_hat = reg.predict(design_matrix)
    y_hat = np.maximum(y_hat, 1)
    alpha_rde = (
        ((normed_counts - y_hat) ** 2 - y_hat) / ((num_samples - num_vars) * y_hat**2)
    ).sum(0)
    return np.maximum(alpha_rde, 0)



def fit_moments_dispersions2(
    normed_counts: np.ndarray, size_factors: np.ndarray
) -> np.ndarray:
    """Dispersion estimates based on moments, as per the R code.

    Used as initial estimates in :meth:`DeseqDataSet.fit_genewise_dispersions()
    <pydeseq2.dds.DeseqDataSet.fit_genewise_dispersions>`.

    Parameters
    ----------
    normed_counts : ndarray
        Array of deseq2-normalized read counts. Rows: samples, columns: genes.

    size_factors : ndarray
        DESeq2 normalization factors.

    Returns
    -------
    ndarray
        Estimated dispersion parameter for each gene.
    """
    # Exclude genes with all zeroes
    #normed_counts = normed_counts[:, ~(normed_counts == 0).all(axis=0)]
    # mean inverse size factor
    s_mean_inv = (1 /size_factors).mean()
    mu = normed_counts.mean(0)
    sigma = normed_counts.var(0, ddof=1)
    # ddof=1 is to use an unbiased estimator, as in R
    # NaN (variance = 0) are replaced with 0s
    return np.nan_to_num((sigma - s_mean_inv * mu) / mu**2)


def nb_nll(
    counts: np.ndarray, mu: np.ndarray, alpha: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    r"""Neg log-likelihood of a negative binomial of parameters ``mu`` and ``alpha``.

    Mathematically, if ``counts`` is a vector of counting entries :math:`y_i`
    then the likelihood of each entry :math:`y_i` to be drawn from a negative
    binomial :math:`NB(\mu, \alpha)` is [1]

    .. math::
        p(y_i | \mu, \alpha) = \frac{\Gamma(y_i + \alpha^{-1})}{
            \Gamma(y_i + 1)\Gamma(\alpha^{-1})
        }
        \left(\frac{1}{1 + \alpha \mu} \right)^{1/\alpha}
        \left(\frac{\mu}{\alpha^{-1} + \mu} \right)^{y_i}

    As a consequence, assuming there are :math:`n` entries,
    the total negative log-likelihood for ``counts`` is

    .. math::
        \ell(\mu, \alpha) = \frac{n}{\alpha} \log(\alpha) +
            \sum_i \left \lbrace
            - \log \left( \frac{\Gamma(y_i + \alpha^{-1})}{
            \Gamma(y_i + 1)\Gamma(\alpha^{-1})
        } \right)
        + (\alpha^{-1} + y_i) \log (\alpha^{-1} + \mu)
        - y_i \log \mu
            \right \rbrace

    This is implemented in this function.

    Parameters
    ----------
    counts : ndarray
        Observations.

    mu : ndarray
        Mean of the distribution :math:`\mu`.

    alpha : float or ndarray
        Dispersion of the distribution :math:`\alpha`,
        s.t. the variance is :math:`\mu + \alpha \mu^2`.

    Returns
    -------
    float or ndarray
        Negative log likelihood of the observations counts
        following :math:`NB(\mu, \alpha)`.

    Notes
    -----
    [1] https://en.wikipedia.org/wiki/Negative_binomial_distribution
    """
    n = len(counts)
    alpha_neg1 = 1 / alpha
    logbinom = gammaln(counts + alpha_neg1) - gammaln(counts + 1) - gammaln(alpha_neg1)
    if hasattr(alpha, "__len__") and len(alpha) > 1:
        return (
            alpha_neg1 * np.log(alpha)
            - logbinom
            + (counts + alpha_neg1) * np.log(mu + alpha_neg1)
            - (counts * np.log(mu))
        ).sum(0)
    else:
        return (
            n * alpha_neg1 * np.log(alpha)
            + (
                -logbinom
                + (counts + alpha_neg1) * np.log(alpha_neg1 + mu)
                - counts * np.log(mu)
            ).sum()
        )


def nbinomGLM(
    design_matrix: np.ndarray,
    counts: np.ndarray,
    cnv: np.ndarray,
    size: np.ndarray,
    offset: np.ndarray,
    prior_no_shrink_scale: float,
    prior_scale: float,
    optimizer="L-BFGS-B",
    shrink_index: int = 1,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Fit a negative binomial MAP LFC using an apeGLM prior.

    Only the LFC is shrinked, and not the intercept.

    Parameters
    ----------
    design_matrix : ndarray
        Design matrix.

    counts : ndarray
        Raw counts.

    size : ndarray
        Size parameter of NB family (inverse of dispersion).

    offset : ndarray
        Natural logarithm of size factor.

    prior_no_shrink_scale : float
        Prior variance for the intercept.

    prior_scale : float
        Prior variance for the LFC parameter.

    optimizer : str
        Optimizing method to use in case IRLS starts diverging.
        Accepted values: 'L-BFGS-B', 'BFGS' or 'Newton-CG'. (default: ``'Newton-CG'``).

    shrink_index : int
        Index of the LFC coordinate to shrink. (default: ``1``).

    Returns
    -------
    beta: ndarray
        2-element array, containing the intercept (first) and the LFC (second).

    inv_hessian: ndarray
        Inverse of the Hessian of the objective at the estimated MAP LFC.

    converged: bool
        Whether L-BFGS-B converged.
    """
    num_vars = design_matrix.shape[-1]

    shrink_mask = np.zeros(num_vars)
    shrink_mask[shrink_index] = 1
    no_shrink_mask = np.ones(num_vars) - shrink_mask

    beta_init = np.ones(num_vars) * 0.1 * (-1) ** (np.arange(num_vars))

    # Set optimization scale
    scale_cnst = nbinomFn(
        np.zeros(num_vars),
        design_matrix,
        counts,
        cnv,
        size,
        offset,
        prior_no_shrink_scale,
        prior_scale,
        shrink_index,
    )
    scale_cnst = np.maximum(scale_cnst, 1)

    def f(beta: np.ndarray, cnst: float = scale_cnst) -> float:
        # Function to optimize
        return (
            nbinomFn(
                beta,
                design_matrix,
                counts,
                cnv,
                size,
                offset,
                prior_no_shrink_scale,
                prior_scale,
                shrink_index,
            )
            / cnst
        )

    def df(beta: np.ndarray, cnst: float = scale_cnst) -> np.ndarray:
        # Gradient of the function to optimize
        xbeta = design_matrix @ beta
        d_neg_prior = (
            beta * no_shrink_mask / prior_no_shrink_scale**2
            + 2 * beta * shrink_mask / (prior_scale**2 + beta[shrink_index] ** 2),
        )
        d_nll = (
            counts - (counts + size) / (1 + size * np.exp(-xbeta - offset - cnv))
        ) @ design_matrix

        return (d_neg_prior - d_nll) / cnst

    def ddf(beta: np.ndarray, cnst: float = scale_cnst) -> np.ndarray:
        # Hessian of the function to optimize
        # Note: will only work if there is a single shrink index
        xbeta = design_matrix @ beta
        exp_xbeta_off = np.exp(xbeta + offset + cnv)
        frac = (counts + size) * size * exp_xbeta_off / (size + exp_xbeta_off) ** 2
        # Build diagonal
        h11 = 1 / prior_no_shrink_scale**2
        h22 = (
            2
            * (prior_scale**2 - beta[shrink_index] ** 2)
            / (prior_scale**2 + beta[shrink_index] ** 2) ** 2
        )

        h = np.diag(no_shrink_mask * h11 + shrink_mask * h22)

        return 1 / cnst * ((design_matrix.T * frac) @ design_matrix + np.diag(h))

    res = minimize(
        f,
        beta_init,
        jac=df,
        hess=ddf if optimizer == "Newton-CG" else None,
        method=optimizer,
    )

    beta = res.x
    converged = res.success

    if not converged and num_vars == 2:
        # If the solver failed, fit using grid search (slow)
        # Only for single-factor analysis
        beta = grid_fit_shrink_beta(
            counts,
            cnv,
            offset,
            design_matrix,
            size,
            prior_no_shrink_scale,
            prior_scale,
            scale_cnst,
            grid_length=60,
            min_beta=-30,
            max_beta=30,
        )

    inv_hessian = np.linalg.inv(ddf(beta, 1))

    return beta, inv_hessian, converged

def nbinomFn(
    beta: np.ndarray,
    design_matrix: np.ndarray,
    counts: np.ndarray,
    cnv: np.ndarray,
    size: np.ndarray,
    offset: np.ndarray,
    prior_no_shrink_scale: float,
    prior_scale: float,
    shrink_index: int = 1,
) -> float:
    """Return the NB negative likelihood with apeGLM prior.

    Use for LFC shrinkage.

    Parameters
    ----------
    beta : ndarray
        2-element array: intercept and LFC coefficients.

    design_matrix : ndarray
        Design matrix.

    counts : ndarray
        Raw counts.

    size : ndarray
        Size parameter of NB family (inverse of dispersion).

    offset : ndarray
        Natural logarithm of size factor.

    prior_no_shrink_scale : float
        Prior variance for the intercept.

    prior_scale : float
        Prior variance for the intercept.

    shrink_index : int
        Index of the LFC coordinate to shrink. (default: ``1``).

    Returns
    -------
    float
        Sum of the NB negative likelihood and apeGLM prior.
    """
    num_vars = design_matrix.shape[-1]

    shrink_mask = np.zeros(num_vars)
    shrink_mask[shrink_index] = 1
    no_shrink_mask = np.ones(num_vars) - shrink_mask

    xbeta = design_matrix @ beta
    prior = (
        (beta * no_shrink_mask) ** 2 / (2 * prior_no_shrink_scale**2)
    ).sum() + np.log1p((beta[shrink_index] / prior_scale) ** 2)

    nll = (
        counts * xbeta - (counts + size) * np.logaddexp(xbeta + offset + cnv, np.log(size))
    ).sum(0)

    return prior - nll



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


def generate_volcano_plot(plot_data, lfc_cut=1.0, pval_cut=0.05, xlim=None, ylim=None):
    plot_data['gene_group'] = plot_data['gene_group'].astype('category')
    
    # Define gene group colors
    gene_group_colors = {
        "DIGs": "#8F3931FF",
        "DSGs": "#FFB977",
        "DCGs": "#FFC300"
    }

    # Create a FacetGrid for faceted plots
    g = sns.FacetGrid(
        plot_data, 
        col="method", 
        margin_titles=True, 
        hue="gene_group", 
        palette=gene_group_colors, 
        sharey=False, 
        sharex=True
    )

    
    # Add points for "DIGs" 
    g.map_dataframe(
        sns.scatterplot, 
        x="log2FC", 
        y="-log10(padj)", 
        alpha=0.1, 
        size=0.5, 
        legend=False, 
        data=plot_data[plot_data['gene_group'].isin(["DIGs"])]
    )

    # Add points for "DSGs" and "DCGs"
    g.map_dataframe(
        sns.scatterplot, 
        x="log2FC", 
        y="-log10(padj)", 
        alpha=1.0, 
        size=3.0, 
        legend=False, 
        data=plot_data[plot_data['gene_group'].isin(["DSGs", "DCGs"])]
    )
    
    # Add vertical and horizontal dashed lines
    for ax in g.axes.flat:
        ax.axvline(x=-lfc_cut, color="gray", linestyle="dashed")
        ax.axvline(x=lfc_cut, color="gray", linestyle="dashed")
        ax.axhline(y=-np.log10(pval_cut), color="gray", linestyle="dashed")
        
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
    
    # Set axis labels
    g.set_axis_labels("Log2 FC", "-Log10 P-value")
    
    # Add titles, legends, and customize
    g.add_legend(title="Gene category")
    g.set_titles(row_template="{row_name}", col_template="{col_name}")
    g.tight_layout()
    
    # Adjust font sizes for better readability
    for ax in g.axes.flat:
        ax.tick_params(axis='both', labelsize=12)
        ax.set_xlabel("Log2 FC", fontsize=14)
        ax.set_ylabel("-Log10 P-value", fontsize=14)
    
    # Save or display the plot
    plt.show()


def plot_cnv_hist(cnv_mean, binwidth=0.2):
    """
    Plots a histogram of the CNV mean distribution.

    Parameters:
        cnv_mean (pd.Series or list): The CNV mean values to plot.
        binwidth (float): The bin width for the histogram.
    """
    # Convert to a DataFrame if it's not already
    if isinstance(cnv_mean, list):
        cnv_mean = pd.DataFrame({'cnv_mean': cnv_mean})
    elif isinstance(cnv_mean, pd.Series):
        cnv_mean = cnv_mean.to_frame(name='cnv_mean')

    # Create the histogram plot
    plt.figure(figsize=(5, 4))
    sns.histplot(
        cnv_mean['cnv_mean'],
        bins=int((cnv_mean['cnv_mean'].max() - cnv_mean['cnv_mean'].min()) / binwidth),
        kde=False,
        color="#F39B7F",
        edgecolor="black",
        alpha=0.7
    )

    # Add labels and titles
    plt.title("", fontsize=14)
    plt.xlabel("CN state", fontsize=14, labelpad=8)
    plt.ylabel("Frequency", fontsize=14, labelpad=8)

    # Customize the appearance of axes
    plt.xticks(fontsize=12, color="black", rotation=45, ha="right")
    plt.yticks(fontsize=12, color="black")
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["left"].set_linewidth(1)
    plt.gca().spines["bottom"].set_linewidth(1)

    # Add a grid
    plt.grid(visible=False)

    # Show the plot
    plt.tight_layout()
    plt.show()