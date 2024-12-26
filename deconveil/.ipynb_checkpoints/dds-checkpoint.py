import sys
import time
import warnings
from typing import List
from typing import Literal
from typing import Optional
from typing import Union
from typing import cast

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import polygamma  # type: ignore
from scipy.stats import f  # type: ignore
from scipy.stats import trim_mean  # type: ignore

from def_inference import DefInference                              
from inference import Inference                                     
import utils_CNaware
from utils_CNaware import fit_rough_dispersions
from utils_CNaware import fit_moments_dispersions2
from utils_CNaware import grid_fit_beta
from utils_CNaware import irls_glm

from pydeseq2.preprocessing import deseq2_norm_fit
from pydeseq2.preprocessing import deseq2_norm_transform
from pydeseq2.utils import build_design_matrix
from pydeseq2.utils import dispersion_trend
from pydeseq2.utils import mean_absolute_deviation
from pydeseq2.utils import n_or_more_replicates
from pydeseq2.utils import nb_nll
from pydeseq2.utils import replace_underscores
from pydeseq2.utils import robust_method_of_moments_disp
from pydeseq2.utils import test_valid_counts
from pydeseq2.utils import trimmed_mean


class pydeseq2CN_data:
    r"""A class to implement dispersion and log fold-change (LFC) estimation.
    Dispersions and LFCs are estimated following the DESeq2 pipeline.
    
    Parameters
    ----------
    counts : pandas.DataFrame
        Raw counts. One column per gene, rows are indexed by sample barcodes.
    
    cnv : pandas.DataFrame
        Discrete numbres. One column per gene, rows are indexed by sample barcodes.
    

    metadata : pandas.DataFrame
        DataFrame containing sample metadata.
        Must be indexed by sample barcodes.

    design_factors : str or list
        Name of the columns of metadata to be used as design variables.
        (default: ``'condition'``).
    
    continuous_factors : list or None
        An optional list of continuous (as opposed to categorical) factors. Any factor
        not in ``continuous_factors`` will be considered categorical (default: ``None``).

    ref_level : list or None
        An optional list of two strings of the form ``["factor", "test_level"]``
        specifying the factor of interest and the reference (control) level against which
        we're testing, e.g. ``["condition", "A"]``. (default: ``None``).

    fit_type: str
        Either ``"parametric"`` or ``"mean"`` for the type of fitting of dispersions to
        the mean intensity. ``"parametric"``: fit a dispersion-mean relation via a
        robust gamma-family GLM. ``"mean"``: use the mean of gene-wise dispersion
        estimates. Will set the fit type for the DEA and the vst transformation. If
        needed, it can be set separately for each method.(default: ``"parametric"``).
        
    min_mu : float
        Threshold for mean estimates. (default: ``0.5``).

    min_disp : float
        Lower threshold for dispersion parameters. (default: ``1e-8``).

    max_disp : float
        Upper threshold for dispersion parameters.
        Note: The threshold that is actually enforced is max(max_disp, len(counts)).
        (default: ``10``).
        
    refit_cooks : bool
        Whether to refit cooks outliers. (default: ``True``).
    
    min_replicates : int
        Minimum number of replicates a condition should have
        to allow refitting its samples. (default: ``7``).

    beta_tol : float
        Stopping criterion for IRWLS. (default: ``1e-8``).

        .. math:: \vert dev_t - dev_{t+1}\vert / (\vert dev \vert + 0.1) < \beta_{tol}.

    n_cpus : int
        Number of cpus to use.  If ``None`` and if ``inference`` is not provided, all
        available cpus will be used by the ``DefaultInference``. If both are specified
        (i.e., ``n_cpus`` and ``inference`` are not ``None``), it will try to override
        the ``n_cpus`` attribute of the ``inference`` object. (default: ``None``).

    inference : Inference
        Implementation of inference routines object instance.
        (default:
        :class:`DefaultInference <pydeseq2.default_inference.DefaultInference>`).  

    Attributes
    ----------
    n_processes : int
        Number of cpus to use for multiprocessing.

    non_zero_idx : ndarray
        Indices of genes that have non-uniformly zero counts.

     non_zero_genes : pandas.Index
        Index of genes that have non-uniformly zero counts.

    logmeans: numpy.ndarray
        Gene-wise mean log counts, computed in ``preprocessing.deseq2_norm_fit()``.

    filtered_genes: numpy.ndarray
        Genes whose log means are different from -∞, computed in
        preprocessing.deseq2_norm_fit().
    
    """
    
    def __init__(
        self,
        *,
        counts: Optional[pd.DataFrame] = None,
        cnv: Optional[pd.DataFrame] = None,
        metadata: Optional[pd.DataFrame] = None,
        design_factors: Union[str, List[str]] = "condition",
        continuous_factors: Optional[List[str]] = None,
        ref_level: Optional[List[str]] = None,
        fit_type: Literal["parametric", "mean"] = "parametric",
        min_mu: float = 0.5,
        min_disp: float = 1e-8,
        max_disp: float = 10.0,
        refit_cooks: bool = True,
        min_replicates: int = 7,
        beta_tol: float = 1e-8,
        n_cpus: Optional[int] = None,
        inference: Optional[Inference] = None,
        quiet: bool = False,
    ) -> None:
        
        """
        Initialize object
        """
        self.data={}
        self.data["counts"] = counts
        self.data["counts"] = self.data["counts"].astype(int)
        
        self.data["cnv"] = cnv

        if self.data["counts"].shape[0] != self.data["cnv"].shape[0] and self.data["counts"].shape[1] != self.data["cnv"].shape[1]:
            raise ValueError("Matrices must have the same dimensions for element-wise operations.")

        # Test counts before going further
        test_valid_counts(counts)

        self.metadata = metadata
        self.fit_type = fit_type

        # Convert design_factors to list if a single string was provided.
        self.design_factors = (
            [design_factors] if isinstance(design_factors, str) else design_factors
        )

        self.continuous_factors = continuous_factors


        # Build the design matrix
        self.design_matrix = build_design_matrix(
            metadata=self.metadata,
            design_factors=self.design_factors,
            continuous_factors=self.continuous_factors,
            ref_level=ref_level,
            expanded=False,
            intercept=True,
        )
        
        self.obsm={}
        self.obsm["design_matrix"] = self.design_matrix 

        # Check that the design matrix has full rank
        #self._check_full_rank_design()
        
        self.min_mu = min_mu
        self.min_disp = min_disp
        self.n_obs=self.data["counts"].shape[0]
        self.n_vars=self.data["counts"].shape[1]
        self.var_names=self.data["counts"].columns
        self.max_disp = np.maximum(max_disp, self.n_obs)
        self.refit_cooks = refit_cooks
        self.ref_level = ref_level
        self.min_replicates = min_replicates
        self.beta_tol = beta_tol
        self.quiet = quiet
        self.logmeans = None
        self.filtered_genes = None
        self.uns={}
        self.varm={}
        self.layers={}
        
        
        if inference:
            if hasattr(inference, "n_cpus"):
                if n_cpus:
                    inference.n_cpus = n_cpus
            else:
                warnings.warn(
                    "The provided inference object does not have an n_cpus "
                    "attribute, cannot override `n_cpus`.",
                    UserWarning,
                    stacklevel=2,
                )
        # Initialize the inference object.
        self.inference = inference or DefInference(n_cpus=n_cpus)
        

    def vst(
        self,
        use_design: bool = False,
        fit_type: Optional[Literal["parametric", "mean"]] = None,
    ) -> None:
        
        """Fit a variance stabilizing transformation, and apply it to normalized counts.
        Results are stored in ``vst_counts"``.
        """
        
        if fit_type is not None:
            self.vst_fit_type = fit_type
        else:
            self.vst_fit_type = self.fit_type

        print(f"Fit type used for VST : {self.vst_fit_type}")
        self.vst_fit(use_design=use_design)
        self.layers["vst_counts"] = self.vst_transform()
        

    def vst_fit(
        self,
        use_design: bool = False,
    ) -> None:
        """Fit a variance stabilizing transformation.

        This method should be called before `vst_transform`.

        Results are stored in ``dds.vst_counts``.

        Parameters
        ----------
        use_design : bool
            Whether to use the full design matrix to fit dispersions and the trend curve.
            If False, only an intercept is used.
            Only useful if ``fit_type = "parametric"`.
            (default: ``False``).
        """
        # Start by fitting median-of-ratio size factors if not already present,
        # or if they were computed iteratively
        if "size_factors" not in self.obsm or self.logmeans is None:
            self.fit_size_factors()  # by default, fit_type != "iterative"

        if not hasattr(self, "vst_fit_type"):
            self.vst_fit_type = self.fit_type

        if use_design:
            if self.vst_fit_type == "parametric":
                self._fit_parametric_dispersion_trend(vst=True)
            else:
                warnings.warn(
                    "use_design=True is only useful when fit_type='parametric'. ",
                    UserWarning,
                    stacklevel=2,
                )
                self.fit_genewise_dispersions(vst=True)

        else:
            # Reduce the design matrix to an intercept and reconstruct at the end
            self.obsm["design_matrix_buffer"] = self.obsm["design_matrix"].copy()
            self.obsm["design_matrix"] = pd.DataFrame(
                1, index=self.obs_names, columns=[["intercept"]]
            )
            # Fit the trend curve with an intercept design
            self.fit_genewise_dispersions(vst=True)
            if self.vst_fit_type == "parametric":
                self._fit_parametric_dispersion_trend(vst=True)

            # Restore the design matrix and free buffer
            self.obsm["design_matrix"] = self.obsm["design_matrix_buffer"].copy()
            del self.obsm["design_matrix_buffer"]
            
        
    def vst_transform(self, counts: Optional[np.ndarray] = None) -> np.ndarray:
        
        """Apply the variance stabilizing transformation.
        Uses the results from the ``vst_fit`` method.
        Returns
        -------
        numpy.ndarray
            Variance stabilized counts.
        """
        
        if "size_factors" not in self.obsm:
            raise RuntimeError(
                "The vst_fit method should be called prior to vst_transform."
            )

        if counts is None:
            # the transformed counts will be the current ones
            normed_counts = self.layers["normed_counts"]
        else:
            if self.logmeans is None:
                # the size factors were still computed iteratively
                warnings.warn(
                    "The size factors were fitted iteratively. They will "
                    "be re-computed with the counts to be transformed. In a train/test "
                    "setting with a downstream task, this would result in a leak of "
                    "data from test to train set.",
                    UserWarning,
                    stacklevel=2,
                )
                logmeans, filtered_genes = deseq2_norm_fit(counts)
            else:
                logmeans, filtered_genes = self.logmeans, self.filtered_genes

            normed_counts, _ = deseq2_norm_transform(counts, logmeans, filtered_genes)
            
        if self.vst_fit_type == "parametric":
            if "vst_trend_coeffs" not in self.uns:
                raise RuntimeError("Fit the dispersion curve prior to applying VST.")

            a0, a1 = self.uns["vst_trend_coeffs"]
            return np.log2(
                (
                    1
                    + a1
                    + 2 * a0 * normed_counts
                    + 2 * np.sqrt(a0 * normed_counts * (1 + a1 + a0 * normed_counts))
                )
                / (4 * a0)
            )
            
        elif self.vst_fit_type == "mean":
            gene_dispersions = self.varm["vst_genewise_dispersions"]
            use_for_mean = gene_dispersions > 10 * self.min_disp
            mean_disp = trim_mean(gene_dispersions[use_for_mean], proportiontocut=0.001)
            return (
                2 * np.arcsinh(np.sqrt(mean_disp * normed_counts))
                - np.log(mean_disp)
                - np.log(4)
            ) / np.log(2)
        else:
            raise NotImplementedError(
                f"Found fit_type '{self.vst_fit_type}'. Expected 'parametric' or 'mean'."
            )
            
    
    def deseq2(self, fit_type: Optional[Literal["parametric", "mean"]] = None) -> None:
        
        """Perform dispersion and log fold-change (LFC) estimation.

        Wrapper for the first part of the PyDESeq2 pipeline.

        Parameters
        ----------
        fit_type : str
            Either None, ``"parametric"`` or ``"mean"`` for the type of fitting of
            dispersions to the mean intensity.``"parametric"``: fit a dispersion-mean
            relation via a robust gamma-family GLM. ``"mean"``: use the mean of
            gene-wise dispersion estimates.

            If None, the fit_type provided at class initialization is used.
            (default: ``None``).
        """
        
        if fit_type is not None:
            self.fit_type = fit_type
            print(f"Using {self.fit_type} fit type.")
        # Compute DESeq2 normalization factors using the Median-of-ratios method
        self.fit_size_factors()
        # Fit an independent negative binomial model per gene
        self.fit_genewise_dispersions()
        # Fit a parameterized trend curve for dispersions, of the form
        # f(\mu) = \alpha_1/\mu + a_0
        self.fit_dispersion_trend()
        # Compute prior dispersion variance
        self.fit_dispersion_prior()
        # Refit genewise dispersions a posteriori (shrinks estimates towards trend curve)
        self.fit_MAP_dispersions()
        # Fit log-fold changes (in natural log scale)
        self.fit_LFC()
         # Compute Cooks distances to find outliers
        self.calculate_cooks()

        if self.refit_cooks:
            # Replace outlier counts, and refit dispersions and LFCs
            # for genes that had outliers replaced
            self.refit()

    def fit_size_factors(
        self,
        fit_type: Literal["ratio", "poscounts", "iterative"] = "ratio",
        control_genes: Optional[
            Union[np.ndarray, List[str], List[int], pd.Index]
        ] = None,
    ) -> None:
        """Fit sample-wise deseq2 normalization (size) factors.
        Parameters
        ----------
        fit_type : str
            The normalization method to use: "ratio", "poscounts" or "iterative".
            (default: ``"ratio"``).
        control_genes : ndarray, list, pandas.Index, or None
            Genes to use as control genes for size factor fitting. If None, all genes
            are used. (default: ``None``).
        """
        
        if not self.quiet:
            print("Fitting size factors...", file=sys.stderr)

        start = time.time()

        # If control genes are provided, set a mask where those genes are True
        if control_genes is not None:
            _control_mask = np.zeros(self.data["counts"].shape[1], dtype=bool)

            # Use AnnData internal indexing to get gene index array
            # Allows bool/int/var_name to be provided
            _control_mask[self._normalize_indices((slice(None), control_genes))[1]] = (
                True
            )

        # Otherwise mask all genes to be True
        else:
            _control_mask = np.ones(self.data["counts"].shape[1], dtype=bool)

        if fit_type == "iterative":
            self._fit_iterate_size_factors()

        elif fit_type == "poscounts":

            # Calculate logcounts for x > 0 and take the mean for each gene
            log_counts = np.zeros_like(self.data["counts"], dtype=float)
            np.log(self.data["counts"], out=log_counts, where=self.data["counts"] != 0)
            logmeans = log_counts.mean(0)

            # Determine which genes are usable (finite logmeans)
            self.filtered_genes = (~np.isinf(logmeans)) & (logmeans > 0)
            _control_mask &= self.filtered_genes

            # Calculate size factor per sample
            def sizeFactor(x):
                _mask = np.logical_and(_control_mask, x > 0)
                return np.exp(np.median(np.log(x[_mask]) - logmeans[_mask]))

            sf = np.apply_along_axis(sizeFactor, 1, self.data["counts"])
            del log_counts

            # Normalize size factors to a geometric mean of 1 to match DESeq
            self.obsm["size_factors"] = sf / (np.exp(np.mean(np.log(sf))))
            self.layers["normed_counts"] = self.data["counts"] / self.obsm["size_factors"][:, None]
            self.logmeans = logmeans

        # Test whether it is possible to use median-of-ratios.
        elif (self.data["counts"] == 0).any().all():
            # There is at least a zero for each gene
            warnings.warn(
                "Every gene contains at least one zero, "
                "cannot compute log geometric means. Switching to iterative mode.",
                UserWarning,
                stacklevel=2,
            )
            self._fit_iterate_size_factors()

        else:
            self.logmeans, self.filtered_genes = deseq2_norm_fit(self.data["counts"])
            _control_mask &= self.filtered_genes

            (
                self.layers["normed_counts"],
                self.obsm["size_factors"],
            ) = deseq2_norm_transform(self.data["counts"], self.logmeans, _control_mask)

        end = time.time()
        self.varm["_normed_means"] = self.layers["normed_counts"].mean(0)

        if not self.quiet:
            print(f"... done in {end - start:.2f} seconds.\n", file=sys.stderr)
            
    
    def fit_genewise_dispersions(self, vst=False) -> None:
        
        """Fit gene-wise dispersion estimates.

        Fits a negative binomial per gene, independently.

        Parameters
        ----------
        vst : bool
            Whether the dispersion estimates are being fitted as part of the VST
            pipeline. (default: ``False``).
        """
        
        # Check that size factors are available. If not, compute them.
        if "size_factors" not in self.obsm:
            self.fit_size_factors()
            
        # Exclude genes with all zeroes
        self.varm["non_zero"] = ~(self.data["counts"] == 0).all(axis=0)
        self.non_zero_idx = np.arange(self.n_vars)[self.varm["non_zero"]]
        self.non_zero_genes = self.var_names[self.varm["non_zero"]]

        #if isinstance(self.non_zero_genes, pd.MultiIndex):
            #raise ValueError("non_zero_genes should not be a MultiIndex")

        # Fit "method of moments" dispersion estimates
        self._fit_MoM_dispersions()

        # Convert to numpy for speed
        design_matrix = self.obsm["design_matrix"].values
        counts=self.data["counts"].to_numpy()
        cnv=self.data["cnv"].to_numpy()
       
         # with a GLM (using rough dispersion estimates).
        if (
            len(self.obsm["design_matrix"].value_counts())
            == self.obsm["design_matrix"].shape[-1]
        ):
            mu_hat_ = self.inference.lin_reg_mu(
                counts=counts[:, self.non_zero_idx],
                size_factors=self.obsm["size_factors"],
                design_matrix=design_matrix,
                min_mu=self.min_mu,
            )
        else:
            _, mu_hat_, _, _ = self.inference.irls_glm(
                counts=counts[:, self.non_zero_idx],
                cnv=cnv[:, self.non_zero_idx],
                size_factors=self.obsm["size_factors"],
                design_matrix=design_matrix,
                disp=self.varm["_MoM_dispersions"][self.non_zero_idx],
                min_mu=self.min_mu,
                beta_tol=self.beta_tol,
            )
        mu_param_name = "_vst_mu_hat" if vst else "_mu_hat"
        #disp_param_name = "vst_genewise_dispersions" if vst else "genewise_dispersions"
        disp_param_name = "genewise_dispersions"

        self.layers[mu_param_name] = np.full((self.n_obs, self.n_vars), np.nan)
        self.layers[mu_param_name][:, self.varm["non_zero"]] = mu_hat_

        if not self.quiet:
            print("Fitting dispersions...", file=sys.stderr)
        start = time.time()
        dispersions_, l_bfgs_b_converged_ = self.inference.alpha_mle(
            counts=counts[:, self.non_zero_idx],
            design_matrix=design_matrix,
            mu=self.layers[mu_param_name][:, self.non_zero_idx],
            alpha_hat=self.varm["_MoM_dispersions"][self.non_zero_idx],
            min_disp=self.min_disp,
            max_disp=self.max_disp,
        )
        end = time.time()

        if not self.quiet:
            print(f"... done in {end - start:.2f} seconds.\n", file=sys.stderr)

        self.varm[disp_param_name] = np.full(self.n_vars, np.nan)
        self.varm[disp_param_name][self.varm["non_zero"]] = np.clip(
            dispersions_, self.min_disp, self.max_disp
        )

        self.varm["_genewise_converged"] = np.full(self.n_vars, np.nan)
        self.varm["_genewise_converged"][self.varm["non_zero"]] = l_bfgs_b_converged_


    def fit_dispersion_trend(self, vst: bool = False) -> None:
        
        """Fit the dispersion trend curve.

        Parameters
        ----------
        vst : bool
            Whether the dispersion trend curve is being fitted as part of the VST
            pipeline. (default: ``False``).
        """
        #disp_param_name = "vst_genewise_dispersions" if vst else "genewise_dispersions"
        disp_param_name = "genewise_dispersions"
        fit_type = self.vst_fit_type if vst else self.fit_type

         # Check that genewise dispersions are available. If not, compute them.
        if disp_param_name not in self.varm:
            self.fit_genewise_dispersions(vst)
        
        #if isinstance(self.varm["disp_param_name"], list):
            #self.varm["disp_param_name"] = np.array(self.varm["disp_param_name"])

        if not self.quiet:
            print("Fitting dispersion trend curve...", file=sys.stderr)
        start = time.time()

        if fit_type == "parametric":
            self._fit_parametric_dispersion_trend(vst)
        elif fit_type == "mean":
            self._fit_mean_dispersion_trend(vst)
        else:
            raise NotImplementedError(
                f"Expected 'parametric' or 'mean' trend curve fit "
                f"types, received {fit_type}"
            )
        end = time.time()

        if not self.quiet:
            print(f"... done in {end - start:.2f} seconds.\n", file=sys.stderr)

    def disp_function(self, x):
        """Return the dispersion trend function at x."""
        if self.uns["disp_function_type"] == "parametric":
            return dispersion_trend(x, self.uns["trend_coeffs"])
        elif self.disp_function_type == "mean":
            return np.full_like(x, self.uns["mean_disp"])
            

    def fit_dispersion_prior(self) -> None:
        """Fit dispersion variance priors and standard deviation of log-residuals.

        The computation is based on genes whose dispersions are above 100 * min_disp.

        Note: when the design matrix has fewer than 3 degrees of freedom, the
        estimate of log dispersions is likely to be imprecise.
        """

        # Check that the dispersion trend curve was fitted. If not, fit it.
        if "fitted_dispersions" not in self.varm:
            self.fit_dispersion_trend()
        
        # Exclude genes with all zeroes
        num_samples = self.n_obs
        num_vars = self.obsm["design_matrix"].shape[-1]

        # Check the degrees of freedom
        if (num_samples - num_vars) <= 3:
            warnings.warn(
                "As the residual degrees of freedom is less than 3, the distribution "
                "of log dispersions is especially asymmetric and likely to be poorly "
                "estimated by the MAD.",
                UserWarning,
                stacklevel=2,
            )
         
        # Fit dispersions to the curve, and compute log residuals
        gene_labels = self.non_zero_genes.to_numpy()
        position_map = {label: idx for idx, label in enumerate(gene_labels)}
        gene_index = self.non_zero_genes.map(lambda x: position_map[x])
        
        disp_residuals = np.log(
            self.varm["genewise_dispersions"][gene_index]
        ) - np.log(self.varm["fitted_dispersions"][gene_index])

        # Compute squared log-residuals and prior variance based on genes whose
        # dispersions are above 100 * min_disp. This is to reproduce DESeq2's behaviour.
        above_min_disp = self.varm["genewise_dispersions"][gene_index] >= (
            100 * self.min_disp
        )

        self.uns["_squared_logres"] = (
            mean_absolute_deviation(disp_residuals[above_min_disp]) ** 2
        )

        self.uns["prior_disp_var"] = np.maximum(
            self.uns["_squared_logres"] - polygamma(1, (num_samples - num_vars) / 2),
            0.25,
        )

    def fit_MAP_dispersions(self) -> None:
        """Fit Maximum a Posteriori dispersion estimates.

        After MAP dispersions are fit, filter genes for which we don't apply shrinkage.
        """

        # Check that the dispersion prior variance is available. If not, compute it.
        if "prior_disp_var" not in self.uns:
            self.fit_dispersion_prior()
        
        # Convert to numpy for speed
        design_matrix = self.obsm["design_matrix"].values
        counts=self.data["counts"].to_numpy()

        if not self.quiet:
            print("Fitting MAP dispersions...", file=sys.stderr)
        start = time.time()
        dispersions_, l_bfgs_b_converged_ = self.inference.alpha_mle(
            counts=counts[:, self.non_zero_idx],
            design_matrix=design_matrix,
            mu=self.layers["_mu_hat"][:, self.non_zero_idx],
            alpha_hat=self.varm["fitted_dispersions"][self.non_zero_idx],
            min_disp=self.min_disp,
            max_disp=self.max_disp,
            prior_disp_var=self.uns["prior_disp_var"].item(),
            cr_reg=True,
            prior_reg=True,
        )
        end = time.time()

        if not self.quiet:
            print(f"... done in {end-start:.2f} seconds.\n", file=sys.stderr)

        self.varm["MAP_dispersions"] = np.full(self.n_vars, np.nan)
        self.varm["MAP_dispersions"][self.varm["non_zero"]] = np.clip(
            dispersions_, self.min_disp, self.max_disp
        )

        self.varm["_MAP_converged"] = np.full(self.n_vars, np.nan)
        self.varm["_MAP_converged"][self.varm["non_zero"]] = l_bfgs_b_converged_

        # Filter outlier genes for which we won't apply shrinkage
        self.varm["dispersions"] = self.varm["MAP_dispersions"].copy()
        self.varm["_outlier_genes"] = np.log(self.varm["genewise_dispersions"]) > np.log(
            self.varm["fitted_dispersions"]
        ) + 2 * np.sqrt(self.uns["_squared_logres"])

        self.varm["dispersions"][self.varm["_outlier_genes"]] = self.varm["genewise_dispersions"][
        self.varm["_outlier_genes"]
        ]
        
   
    
    def fit_LFC(self) -> None:
        """Fit log fold change (LFC) coefficients.

        In the 2-level setting, the intercept corresponds to the base mean,
        while the second is the actual LFC coefficient, in natural log scale.
        """
        
         # Check that MAP dispersions are available. If not, compute them.
        if "dispersions" not in self.varm:
            self.fit_MAP_dispersions()
            
         # Convert to numpy for speed
        design_matrix = self.obsm["design_matrix"].values
        counts=self.data["counts"].to_numpy()
        cnv=self.data["cnv"].to_numpy()
        cnv = cnv + 0.1

        if not self.quiet:
            print("Fitting LFCs...", file=sys.stderr)
        start = time.time()
        mle_lfcs_, mu_, hat_diagonals_, converged_ = self.inference.irls_glm(
            counts=counts[:, self.non_zero_idx],
            cnv=cnv[:, self.non_zero_idx],
            size_factors=self.obsm["size_factors"],
            design_matrix=design_matrix,
            disp=self.varm["dispersions"][self.non_zero_idx],
            min_mu=self.min_mu,
            beta_tol=self.beta_tol,
        )
        end = time.time()

        if not self.quiet:
            print(f"... done in {end-start:.2f} seconds.\n", file=sys.stderr)

        self.varm["LFC"] = pd.DataFrame(
            np.nan,
            index=self.var_names,
            columns=self.obsm["design_matrix"].columns,
        )

        self.varm["LFC"].update(
            pd.DataFrame(
                mle_lfcs_,
                index=self.non_zero_genes,
                columns=self.obsm["design_matrix"].columns,
            )
        )

        self.obsm["_mu_LFC"] = mu_
        self.obsm["_hat_diagonals"] = hat_diagonals_

        self.varm["_LFC_converged"] = np.full(self.n_vars, np.nan)
        self.varm["_LFC_converged"][self.varm["non_zero"]] = converged_
        

    def calculate_cooks(self) -> None:
        """Compute Cook's distance for outlier detection.

        Measures the contribution of a single entry to the output of LFC estimation.
        """
        # Check that MAP dispersions are available. If not, compute them.
        if "dispersions" not in self.varm:
            self.fit_MAP_dispersions()

        if not self.quiet:
            print("Calculating cook's distance...", file=sys.stderr)

        start = time.time()
        num_vars = self.obsm["design_matrix"].shape[-1]

        #self.layers["normed_counts"] = self.layers["normed_counts"].to_numpy()
        non_zero_mask = self.varm["non_zero"].to_numpy()
        self.layers["normed_counts"] = self.layers["normed_counts"].to_numpy()
        
        # Calculate dispersion
        dispersions = robust_method_of_moments_disp(
            self.layers["normed_counts"][:, non_zero_mask],
            self.obsm["design_matrix"],
        )

        self.data["counts"] = self.data["counts"].to_numpy()
        
        # Calculate the squared pearson residuals for non-zero features
        squared_pearson_res = self.data["counts"][:, non_zero_mask] - self.obsm["_mu_LFC"]
        squared_pearson_res **= 2

        # Calculate the overdispersion parameter tau
        V = self.obsm["_mu_LFC"] ** 2
        V *= dispersions[None, :]
        V += self.obsm["_mu_LFC"]

        # Calculate r^2 / (tau * num_vars)
        squared_pearson_res /= V
        squared_pearson_res /= num_vars

        del V

        # Calculate leverage modifier H / (1 - H)^2
        diag_mul = 1 - self.obsm["_hat_diagonals"]
        diag_mul **= 2
        diag_mul = self.obsm["_hat_diagonals"] / diag_mul

        # Multiply r^2 / (tau * num_vars) by H / (1 - H)^2 to get cook's distance
        squared_pearson_res *= diag_mul

        del diag_mul

        self.layers["cooks"] = np.full((self.n_obs, self.n_vars), np.nan)
        self.layers["cooks"][:, non_zero_mask] = squared_pearson_res

        if not self.quiet:
            print(f"... done in {time.time()-start:.2f} seconds.\n", file=sys.stderr)

    def refit(self) -> None:
        """Refit Cook outliers.

        Replace values that are filtered out based on the Cooks distance with imputed
        values, and then re-run the whole DESeq2 pipeline on replaced values.
        """
        # Replace outlier counts
        self._replace_outliers()
        if not self.quiet:
            print(
                f"Replacing {sum(self.varm['replaced']) } outlier genes.\n",
                file=sys.stderr,
            )

        if sum(self.varm["replaced"]) > 0:
            # Refit dispersions and LFCs for genes that had outliers replaced
            self._refit_without_outliers()
        else:
            # Store the fact that no sample was refitted
            self.varm["refitted"] = np.full(
                self.n_vars,
                False,
            )
        

    def _fit_MoM_dispersions(self) -> None:
        
        """Rough method of moments initial dispersions fit.

        Estimates are the max of "robust" and "method of moments" estimates.
        """
        # Check that size_factors are available. If not, compute them.
        if "normed_counts" not in self.layers:
            self.fit_size_factors()

        normed_counts = self.layers["normed_counts"]
        rde = self.inference.fit_rough_dispersions(
            normed_counts,
            self.obsm["design_matrix"].values,
        )
        mde = self.inference.fit_moments_dispersions2(
            normed_counts, 
            self.obsm["size_factors"]
        )
        alpha_hat = np.minimum(rde, mde)

        self.varm["_MoM_dispersions"] = np.full(self.n_vars, np.nan)
        self.varm["_MoM_dispersions"][self.varm["non_zero"]] = np.clip(
            alpha_hat, self.min_disp, self.max_disp
        )
        

    def _fit_parametric_dispersion_trend(self, vst: bool = False):
        
        r"""Fit the dispersion curve according to a parametric model.

        :math:`f(\mu) = \alpha_1/\mu + a_0`.

        Parameters
        ----------
        vst : bool
            Whether the dispersion trend curve is being fitted as part of the VST
            pipeline. (default: ``False``).
        """
        #disp_param_name = "vst_genewise_dispersions" if vst else "genewise_dispersions"
        disp_param_name = "genewise_dispersions"

        if disp_param_name not in self.varm:
            self.fit_genewise_dispersions(vst)

        #disp_param_name = "disp_param_name"

        # Exclude all-zero counts
        targets = pd.Series(
            self.varm[disp_param_name][self.non_zero_idx].copy(),
            index=self.non_zero_genes,
        )
        covariates = pd.Series(
            1 / self.varm["_normed_means"][self.non_zero_idx],
            index=self.non_zero_genes,
        )

        for gene in self.non_zero_genes:
            if (
                np.isinf(covariates.loc[gene]).any()
                or np.isnan(covariates.loc[gene]).any()
            ):
                targets.drop(labels=[gene], inplace=True)
                covariates.drop(labels=[gene], inplace=True)

        # Initialize coefficients
        old_coeffs = pd.Series([0.1, 0.1])
        coeffs = pd.Series([1.0, 1.0])
        while (coeffs > 1e-10).all() and (
            np.log(np.abs(coeffs / old_coeffs)) ** 2
        ).sum() >= 1e-6:
            old_coeffs = coeffs
            coeffs, predictions, converged = self.inference.dispersion_trend_gamma_glm(
                covariates, targets
            )

            if not converged or (coeffs <= 1e-10).any():
                warnings.warn(
                    "The dispersion trend curve fitting did not converge. "
                    "Switching to a mean-based dispersion trend.",
                    UserWarning,
                    stacklevel=2,
                )

                self._fit_mean_dispersion_trend(vst)
                return

            # Filter out genes that are too far away from the curve before refitting
            gene_labels = self.non_zero_genes.to_numpy()
            position_map = {label: idx for idx, label in enumerate(gene_labels)}
            covariates_index = covariates.index.map(lambda x: position_map[x])
            
            
            pred_ratios = self.varm[disp_param_name][covariates_index] / predictions

            targets.drop(
                targets[(pred_ratios < 1e-4) | (pred_ratios >= 15)].index,
                inplace=True,
            )
            covariates.drop(
                covariates[(pred_ratios < 1e-4) | (pred_ratios >= 15)].index,
                inplace=True,
            )

        if vst:
            self.uns["vst_trend_coeffs"] = pd.Series(coeffs, index=["a0", "a1"])
        else:
            self.uns["trend_coeffs"] = pd.Series(coeffs, index=["a0", "a1"])

            self.varm["fitted_dispersions"] = np.full(self.n_vars, np.nan)
            self.uns["disp_function_type"] = "parametric"
            self.varm["fitted_dispersions"][self.varm["non_zero"]] = self.disp_function(
                self.varm["_normed_means"][self.varm["non_zero"]]
            )

    def _fit_mean_dispersion_trend(self, vst: bool = False):
        """Use the mean of dispersions as trend curve.

        Parameters
        ----------
        vst : bool
            Whether the dispersion trend curve is being fitted as part of the VST
            pipeline. (default: ``False``).
        """
        #disp_param_name = "vst_genewise_dispersions" if vst else "genewise_dispersions"
        disp_param_name = "genewise_dispersions"

        self.uns["mean_disp"] = trim_mean(
            self.varm[disp_param_name][self.varm[disp_param_name] > 10 * self.min_disp],
            proportiontocut=0.001,
        )

        self.uns["disp_function_type"] = "mean"
        self.varm["fitted_dispersions"] = np.full(self.n_vars, self.uns["mean_disp"])


    def _replace_outliers(self) -> None:
        """Replace values that are filtered out (based on Cooks) with imputed values."""
        # Check that cooks distances are available. If not, compute them.
        if "cooks" not in self.layers:
            self.calculate_cooks()

        num_samples = self.n_obs
        num_vars = self.obsm["design_matrix"].shape[1]

        # Check whether cohorts have enough samples to allow refitting
        self.obsm["replaceable"] = n_or_more_replicates(
            self.obsm["design_matrix"], self.min_replicates
        ).values

        if self.obsm["replaceable"].sum() == 0:
            # No sample can be replaced. Set self.replaced to False and exit.
            self.varm["replaced"] = np.full(
                self.n_vars,
                False,
            )
            return

        # Get positions of counts with cooks above threshold
        cooks_cutoff = f.ppf(0.99, num_vars, num_samples - num_vars)
        idx = self.layers["cooks"] > cooks_cutoff
        self.varm["replaced"] = idx.any(axis=0)

        if sum(self.varm["replaced"] > 0):
            # Compute replacement counts: trimmed means * size_factors
            #self.counts_to_refit = self[:, self.varm["replaced"]].copy()
            #self.counts_to_refit = self.data["counts"][:, self.varm["replaced"]].copy()

            self.counts_to_refit = pd.DataFrame(
                self.data["counts"][:, self.varm["replaced"]], 
                columns=self.var_names[self.varm["replaced"]]
            )

            #trim_base_mean = pd.DataFrame(
                #cast(
                    #np.ndarray,
                    #trimmed_mean(
                        #self.counts_to_refit / self.obsm["size_factors"][:, None],
                        #trim=0.2,
                        #axis=0,
                    #),
                #),
                #index=self.counts_to_refit.var_names,
            #)

            trim_base_mean = pd.DataFrame(
                cast(
                    np.ndarray,
                    trimmed_mean(
                        self.counts_to_refit / self.obsm["size_factors"][:, None],
                        trim=0.2,
                        axis=0,
                    ),
                ),
                index=self.counts_to_refit.columns  # Use .columns instead of .var_names
            )

            #replacement_counts = (
                #pd.DataFrame(
                    #trim_base_mean.values * self.obsm["size_factors"],
                    #index=self.counts_to_refit.var_names,
                    #columns=self.counts_to_refit.obs_names,
                #)
                #.astype(int)
                #.T
            #)

            #self.counts_to_refit.data["counts"][
                #self.obsm["replaceable"][:, None] & idx[:, self.varm["replaced"]]
            #] = replacement_counts.values[
                #self.obsm["replaceable"][:, None] & idx[:, self.varm["replaced"]]
            #]

            replacement_counts = (
                pd.DataFrame(
                    (trim_base_mean.values * self.obsm["size_factors"]).T,  # Ensure this is 2D
                    index=self.counts_to_refit.index, 
                    columns=self.counts_to_refit.columns  
                )
                .astype(int)
            )

            replace_mask = self.obsm["replaceable"][:, None] & idx[:, self.varm["replaced"]]
            
            print(f"replace_mask before filtering: {replace_mask.shape}")
            print(f"Number of True values in replace_mask: {np.sum(replace_mask)}")

            # Ensure that replacement_counts and the indexing result are compatible
            replacement_counts_trimmed = replacement_counts.loc[replace_mask.any(axis=1)] 
            print(f"replacement_counts_trimmed shape: {replacement_counts_trimmed.shape}")

            # Apply the mask to refit only the matching rows, ensuring consistent dimensions
            assert replacement_counts_trimmed.shape == self.counts_to_refit.loc[replace_mask.any(axis=1)].shape, \
                f"Shape mismatch: replacement_counts_trimmed {replacement_counts_trimmed.shape} vs refit slice  {self.counts_to_refit.loc[replace_mask.any(axis=1)].shape}"

            # Replace counts in self.counts_to_refit for the rows matching the mask
            self.counts_to_refit.loc[replace_mask.any(axis=1)] = replacement_counts_trimmed.values


    def _refit_without_outliers(
        self,
    ) -> None:
        """Re-run the whole DESeq2 pipeline with replaced outliers."""
        assert (
            self.refit_cooks
        ), "Trying to refit Cooks outliers but the 'refit_cooks' flag is set to False"

        # Check that _replace_outliers() was previously run.
        if "replaced" not in self.varm:
            self._replace_outliers()

        # Only refit genes for which replacing outliers hasn't resulted in all zeroes
        new_all_zeroes = (self.counts_to_refit == 0).all(axis=0)
        self.new_all_zeroes_genes = self.counts_to_refit.columns[new_all_zeroes]

        self.varm["refitted"] = self.varm["replaced"].copy()
        # Only replace if genes are not all zeroes after outlier replacement
        self.varm["refitted"][self.varm["refitted"]] = ~new_all_zeroes

        # Take into account new all-zero genes
        if new_all_zeroes.sum() > 0:
            self.varm["_normed_means"][
                self.var_names.get_indexer(self.new_all_zeroes_genes)
            ] = 0
            self.varm["LFC"].loc[self.new_all_zeroes_genes, :] = 0

        if self.varm["refitted"].sum() == 0:  # if no gene can be refitted, we can skip
            return

        self.counts_to_refit = self.counts_to_refit.loc[:, ~new_all_zeroes].copy()
        if isinstance(self.new_all_zeroes_genes, pd.MultiIndex):
            raise ValueError

        sub_dds = pydeseq2CN_data(
            counts=pd.DataFrame(
                self.counts_to_refit,
                index=self.counts_to_refit.index,
                columns=self.counts_to_refit.columns,
            ),
            cnv=self.data["cnv"],
            metadata=self.metadata,
            design_factors=self.design_factors,
            continuous_factors=self.continuous_factors,
            ref_level=self.ref_level,
            min_mu=self.min_mu,
            min_disp=self.min_disp,
            max_disp=self.max_disp,
            refit_cooks=self.refit_cooks,
            min_replicates=self.min_replicates,
            beta_tol=self.beta_tol,
            inference=self.inference,
        )

        # Use the same size factors
        sub_dds.obsm["size_factors"] = self.obsm["size_factors"]
        sub_dds.layers["normed_counts"] = (
            sub_dds.data["counts"] / sub_dds.obsm["size_factors"][:, None]
        )

        # Estimate gene-wise dispersions.
        sub_dds.fit_genewise_dispersions()

        # Compute trend dispersions.
        # Note: the trend curve is not refitted.
        sub_dds.uns["disp_function_type"] = self.uns["disp_function_type"]
        if sub_dds.uns["disp_function_type"] == "parametric":
            sub_dds.uns["trend_coeffs"] = self.uns["trend_coeffs"]
        elif sub_dds.uns["disp_function_type"] == "mean":
            sub_dds.uns["mean_disp"] = self.uns["mean_disp"]
        sub_dds.varm["_normed_means"] = sub_dds.layers["normed_counts"].mean(0)
        # Reshape in case there's a single gene to refit
        sub_dds.varm["fitted_dispersions"] = sub_dds.disp_function(
            sub_dds.varm["_normed_means"]
        )

        # Estimate MAP dispersions.
        # Note: the prior variance is not recomputed.
        sub_dds.uns["_squared_logres"] = self.uns["_squared_logres"]
        sub_dds.uns["prior_disp_var"] = self.uns["prior_disp_var"]

        sub_dds.fit_MAP_dispersions()

        # Estimate log-fold changes (in natural log scale)
        sub_dds.fit_LFC()

        # Replace values in main object
        self.varm["_normed_means"][self.varm["refitted"]] = sub_dds.varm["_normed_means"]
        self.varm["LFC"][self.varm["refitted"]] = sub_dds.varm["LFC"]
        self.varm["genewise_dispersions"][self.varm["refitted"]] = sub_dds.varm[
            "genewise_dispersions"
        ]
        self.varm["fitted_dispersions"][self.varm["refitted"]] = sub_dds.varm[
            "fitted_dispersions"
        ]
        self.varm["dispersions"][self.varm["refitted"]] = sub_dds.varm["dispersions"]

        replace_cooks = pd.DataFrame(self.layers["cooks"].copy())
        replace_cooks.loc[self.obsm["replaceable"], self.varm["refitted"]] = 0.0

        self.layers["replace_cooks"] = replace_cooks
        

    def _fit_iterate_size_factors(self, niter: int = 10, quant: float = 0.95) -> None:
        """
        Fit size factors using the ``iterative`` method.

        Used when each gene has at least one zero.

        Parameters
        ----------
        niter : int
            Maximum number of iterations to perform (default: ``10``).

        quant : float
            Quantile value at which negative likelihood is cut in the optimization
            (default: ``0.95``).

        """
        # Initialize size factors and normed counts fields
        self.obsm["size_factors"] = np.ones(self.n_obs)
        self.layers["normed_counts"] = self.data["counts"]

        # Reduce the design matrix to an intercept and reconstruct at the end
        self.obsm["design_matrix_buffer"] = self.obsm["design_matrix"].copy()
        self.obsm["design_matrix"] = pd.DataFrame(
            1, index=self.obs_names, columns=["intercept"]
        )

        # Fit size factors using MLE
        def objective(p):
            sf = np.exp(p - np.mean(p))
            nll = nb_nll(
                counts=self[:, self.non_zero_genes].data["counts"],
                mu=self[:, self.non_zero_genes].layers["_mu_hat"]
                / self.obsm["size_factors"][:, None]
                * sf[:, None],
                alpha=self[:, self.non_zero_genes].varm["dispersions"],
            )
            # Take out the lowest likelihoods (highest neg) from the sum
            return np.sum(nll[nll < np.quantile(nll, quant)])

        for i in range(niter):
            # Estimate dispersions based on current size factors
            self.fit_genewise_dispersions()

            # Use a mean trend curve
            use_for_mean_genes = self.var_names[
                (self.varm["genewise_dispersions"] > 10 * self.min_disp)
                & self.varm["non_zero"]
            ]

            if len(use_for_mean_genes) == 0:
                print(
                    "No genes have a dispersion above 10 * min_disp in "
                    "_fit_iterate_size_factors."
                )
                break

            mean_disp = trim_mean(
                self[:, use_for_mean_genes].varm["genewise_dispersions"],
                proportiontocut=0.001,
            )

            self.varm["fitted_dispersions"] = np.ones(self.n_vars) * mean_disp
            self.fit_dispersion_prior()
            self.fit_MAP_dispersions()
            old_sf = self.obsm["size_factors"].copy()

            # Fit size factors using MLE
            res = minimize(objective, np.log(old_sf), method="Powell")

            self.obsm["size_factors"] = np.exp(res.x - np.mean(res.x))

            if not res.success:
                print("A size factor fitting iteration failed.", file=sys.stderr)
                break

            if (i > 1) and np.sum(
                (np.log(old_sf) - np.log(self.obsm["size_factors"])) ** 2
            ) < 1e-4:
                break
            elif i == niter - 1:
                print("Iterative size factor fitting did not converge.", file=sys.stderr)

        # Restore the design matrix and free buffer
        self.obsm["design_matrix"] = self.obsm["design_matrix_buffer"].copy()
        del self.obsm["design_matrix_buffer"]

        # Store normalized counts
        self.layers["normed_counts"] = self.data["counts"] / self.obsm["size_factors"][:, None]

    
    def _check_full_rank_design(self):
        """Check that the design matrix has full column rank."""
        rank = np.linalg.matrix_rank(self.obsm["design_matrix"])
        num_vars = self.obsm["design_matrix"].shape[1]

        if rank < num_vars:
            warnings.warn(
                "The design matrix is not full rank, so the model cannot be "
                "fitted, but some operations like design-free VST remain possible. "
                "To perform differential expression analysis, please remove the design "
                "variables that are linear combinations of others.",
                UserWarning,
                stacklevel=2,
            )
     
    
