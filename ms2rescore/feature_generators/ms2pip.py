"""
MS²PIP fragmentation intensity-based feature generator.

MS²PIP is a machine learning tool that predicts the MS2 spectrum of a peptide given its sequence.
It is previously identified MS2 spectra and their corresponding peptide sequences. Because MS²PIP
uses the highly performant - but traditional - machine learning approach XGBoost, it can already
produce accurate predictions even if trained on smaller spectral libraries. This makes MS²PIP a
very flexible platform to train new models on custom datasets. Nevertheless, MS²PIP comes with
several pre-trained models. See
`github.com/compomics/ms2pip <https://github.com/compomics/ms2pip>`_ for more information.

Because traditional proteomics search engines do not fully consider MS2 peak intensities in their
scoring functions, adding rescoring features derived from spectrum prediction tools has proved to
be a very effective way to further improve the sensitivity of peptide-spectrum matching.

If you use MS²PIP through MS²Rescore, please cite:

.. epigraph::
    Declercq, A., Bouwmeester, R., Chiva, C., Sabidó, E., Hirschler, A., Carapito, C., Martens, L.,
    Degroeve, S., Gabriels, R. Updated MS²PIP web server supports cutting-edge proteomics
    applications. *Nucleic Acids Research* (2023)
    `doi:10.1093/nar/gkad335 <https://doi.org/10.1093/nar/gkad335>`_

"""

import logging
import multiprocessing
import os
import warnings
from itertools import chain
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from ms2pip import correlate
from ms2pip.exceptions import NoMatchingSpectraFound
from ms2pip.result import ProcessingResult
from psm_utils import PSMList
from rich.progress import track

from ms2rescore.feature_generators.base import FeatureGeneratorBase, FeatureGeneratorException
from ms2rescore.parse_spectra import MSDataType
from ms2rescore.utils import infer_spectrum_path

logger = logging.getLogger(__name__)


class MS2PIPFeatureGenerator(FeatureGeneratorBase):
    """Generate MS²PIP-based features."""

    required_ms_data = {MSDataType.ms2_spectra}

    def __init__(
        self,
        *args,
        model: str = "HCD",
        ms2_tolerance: float = 0.02,
        spectrum_path: Optional[str] = None,
        spectrum_id_pattern: str = "(.*)",
        model_dir: Optional[str] = None,
        processes: 1,
        **kwargs,
    ) -> None:
        """
        Generate MS²PIP-based features.

        Parameters
        ----------
        model
            MS²PIP prediction model to use. Defaults to :py:const:`HCD`.
        ms2_tolerance
            MS2 mass tolerance in Da. Defaults to :py:const:`0.02`.
        spectrum_path
            Path to spectrum file or directory with spectrum files. If None, inferred from ``run``
            field in PSMs. Defaults to :py:const:`None`.
        spectrum_id_pattern : str, optional
            Regular expression pattern to extract spectrum ID from spectrum file. Defaults to
            :py:const:`.*`.
        model_dir
            Directory containing MS²PIP models. Defaults to :py:const:`None` (use MS²PIP default).
        processes : int, optional
            Number of processes to use. Defaults to 1.

        Attributes
        ----------
        feature_names: list[str]
            Names of the features that will be added to the PSMs.

        """
        super().__init__(*args, **kwargs)
        self.model = model
        self.ms2_tolerance = ms2_tolerance
        self.spectrum_path = spectrum_path
        self.spectrum_id_pattern = spectrum_id_pattern
        self.model_dir = model_dir
        self.processes = processes

    @property
    def feature_names(self):
        return [
            "spec_pearson_norm",
            "ionb_pearson_norm",
            "iony_pearson_norm",
            "spec_mse_norm",
            "ionb_mse_norm",
            "iony_mse_norm",
            "min_abs_diff_norm",
            "max_abs_diff_norm",
            "abs_diff_Q1_norm",
            "abs_diff_Q2_norm",
            "abs_diff_Q3_norm",
            "mean_abs_diff_norm",
            "std_abs_diff_norm",
            "ionb_min_abs_diff_norm",
            "ionb_max_abs_diff_norm",
            "ionb_abs_diff_Q1_norm",
            "ionb_abs_diff_Q2_norm",
            "ionb_abs_diff_Q3_norm",
            "ionb_mean_abs_diff_norm",
            "ionb_std_abs_diff_norm",
            "iony_min_abs_diff_norm",
            "iony_max_abs_diff_norm",
            "iony_abs_diff_Q1_norm",
            "iony_abs_diff_Q2_norm",
            "iony_abs_diff_Q3_norm",
            "iony_mean_abs_diff_norm",
            "iony_std_abs_diff_norm",
            "dotprod_norm",
            "dotprod_ionb_norm",
            "dotprod_iony_norm",
            "cos_norm",
            "cos_ionb_norm",
            "cos_iony_norm",
            "weighted_dotprod_norm",
            "weighted_dotprod_ionb_norm",
            "weighted_dotprod_iony_norm",
            "spectrast_norm",
            "spectrast_ionb_norm",
            "spectrast_iony_norm",
            "spectra_angle_norm",
            "spectra_angle_ionb_norm",
            "spectra_angle_iony_norm",
            "spec_pearson",
            "ionb_pearson",
            "iony_pearson",
            "spec_spearman",
            "ionb_spearman",
            "iony_spearman",
            "spec_mse",
            "ionb_mse",
            "iony_mse",
            "min_abs_diff_iontype",
            "max_abs_diff_iontype",
            "min_abs_diff",
            "max_abs_diff",
            "abs_diff_Q1",
            "abs_diff_Q2",
            "abs_diff_Q3",
            "mean_abs_diff",
            "std_abs_diff",
            "ionb_min_abs_diff",
            "ionb_max_abs_diff",
            "ionb_abs_diff_Q1",
            "ionb_abs_diff_Q2",
            "ionb_abs_diff_Q3",
            "ionb_mean_abs_diff",
            "ionb_std_abs_diff",
            "iony_min_abs_diff",
            "iony_max_abs_diff",
            "iony_abs_diff_Q1",
            "iony_abs_diff_Q2",
            "iony_abs_diff_Q3",
            "iony_mean_abs_diff",
            "iony_std_abs_diff",
            "dotprod",
            "dotprod_ionb",
            "dotprod_iony",
            "cos",
            "cos_ionb",
            "cos_iony",
            "weighted_dotprod",
            "weighted_dotprod_ionb",
            "weighted_dotprod_iony",
            "spectrast",
            "spectrast_ionb",
            "spectrast_iony",
            "spectra_angle",
            "spectra_angle_ionb",
            "spectra_angle_iony",
            "nist_match_factor"
        ]

    def add_features(self, psm_list: PSMList) -> None:
        """
        Add MS²PIP-derived features to PSMs.

        Parameters
        ----------
        psm_list
            PSMs to add features to.

        """
        logger.info("Adding MS²PIP-derived features to PSMs.")
        psm_dict = psm_list.get_psm_dict()
        current_run = 1
        total_runs = sum(len(runs) for runs in psm_dict.values())

        for runs in psm_dict.values():
            for run, psms in runs.items():
                logger.info(
                    f"Running MS²PIP for PSMs from run ({current_run}/{total_runs}) `{run}`..."
                )
                psm_list_run = PSMList(psm_list=list(chain.from_iterable(psms.values())))
                spectrum_filename = infer_spectrum_path(self.spectrum_path, run)
                logger.debug(f"Using spectrum file `{spectrum_filename}`")
                try:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                    ms2pip_results = correlate(
                        psms=psm_list_run,
                        spectrum_file=str(spectrum_filename),
                        spectrum_id_pattern=self.spectrum_id_pattern,
                        model=self.model,
                        ms2_tolerance=self.ms2_tolerance,
                        compute_correlations=False,
                        model_dir=self.model_dir,
                        processes=self.processes,
                    )
                except NoMatchingSpectraFound as e:
                    raise FeatureGeneratorException(
                        f"Could not find any matching spectra for PSMs from run `{run}`. "
                        "Please check that the `spectrum_id_pattern` and `psm_id_pattern` "
                        "options are configured correctly. See "
                        "https://ms2rescore.readthedocs.io/en/latest/userguide/configuration/#mapping-psms-to-spectra"
                        " for more information."
                    ) from e
                self._calculate_features(psm_list_run, ms2pip_results)
                current_run += 1

    def _calculate_features(
        self, psm_list: PSMList, ms2pip_results: List[ProcessingResult]
    ) -> None:
        """Calculate features from all MS²PIP results and add to PSMs."""
        logger.debug("Calculating features from predicted spectra")
        with multiprocessing.Pool(int(self.processes)) as pool:
            # Use imap, so we can use a progress bar
            counts_failed = 0
            for result, features in zip(
                ms2pip_results,
                track(
                    pool.imap(self._calculate_features_single, ms2pip_results, chunksize=1000),
                    total=len(ms2pip_results),
                    description="Calculating features...",
                    transient=True,
                ),
            ):
                if features:
                    # Cannot use result.psm directly, as it is a copy from MS²PIP multiprocessing
                    try:
                        psm_list[result.psm_index]["rescoring_features"].update(features)
                    except (AttributeError, TypeError):
                        psm_list[result.psm_index]["rescoring_features"] = features
                else:
                    counts_failed += 1

        if counts_failed > 0:
            logger.warning(f"Failed to calculate features for {counts_failed} PSMs")

    def _calculate_features_single(self, processing_result: ProcessingResult) -> Union[dict, None]:
        """Calculate MS²PIP-based features for single PSM."""
        if (
            processing_result.observed_intensity is None
            or processing_result.predicted_intensity is None
        ):
            return None

        # Suppress RuntimeWarnings about invalid values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Convert intensities to arrays
            target_b = processing_result.predicted_intensity["b"].clip(np.log2(0.001))
            target_y = processing_result.predicted_intensity["y"].clip(np.log2(0.001))
            target_all = np.concatenate([target_b, target_y])
            prediction_b = processing_result.observed_intensity["b"].clip(np.log2(0.001))
            prediction_y = processing_result.observed_intensity["y"].clip(np.log2(0.001))
            prediction_all = np.concatenate([prediction_b, prediction_y])
            # convert m/z values to arrays
            mz_b = processing_result.theoretical_mz["b"]
            mz_y = processing_result.theoretical_mz["y"]
            mz_all = np.concatenate([mz_b, mz_y])

            # Prepare 'unlogged' intensity arrays
            target_b_unlog = 2**target_b - 0.001
            target_y_unlog = 2**target_y - 0.001
            target_all_unlog = 2**target_all - 0.001
            prediction_b_unlog = 2**prediction_b - 0.001
            prediction_y_unlog = 2**prediction_y - 0.001
            prediction_all_unlog = 2**prediction_all - 0.001

            # Calculate absolute differences
            abs_diff_b = np.abs(target_b - prediction_b)
            abs_diff_y = np.abs(target_y - prediction_y)
            abs_diff_all = np.abs(target_all - prediction_all)
            abs_diff_b_unlog = np.abs(target_b_unlog - prediction_b_unlog)
            abs_diff_y_unlog = np.abs(target_y_unlog - prediction_y_unlog)
            abs_diff_all_unlog = np.abs(target_all_unlog - prediction_all_unlog)

            # Compute features
            feature_values = [
                # Features between spectra in log space
                np.corrcoef(target_all, prediction_all)[0][1],  # Pearson all ions
                np.corrcoef(target_b, prediction_b)[0][1],  # Pearson b ions
                np.corrcoef(target_y, prediction_y)[0][1],  # Pearson y ions
                _mse(target_all, prediction_all),  # MSE all ions
                _mse(target_b, prediction_b),  # MSE b ions
                _mse(target_y, prediction_y),  # MSE y ions
                np.min(abs_diff_all),  # min_abs_diff_norm
                np.max(abs_diff_all),  # max_abs_diff_norm
                np.quantile(abs_diff_all, 0.25),  # abs_diff_Q1_norm
                np.quantile(abs_diff_all, 0.5),  # abs_diff_Q2_norm
                np.quantile(abs_diff_all, 0.75),  # abs_diff_Q3_norm
                np.mean(abs_diff_all),  # mean_abs_diff_norm
                np.std(abs_diff_all),  # std_abs_diff_norm
                np.min(abs_diff_b),  # ionb_min_abs_diff_norm
                np.max(abs_diff_b),  # ionb_max_abs_diff_norm
                np.quantile(abs_diff_b, 0.25),  # ionb_abs_diff_Q1_norm
                np.quantile(abs_diff_b, 0.5),  # ionb_abs_diff_Q2_norm
                np.quantile(abs_diff_b, 0.75),  # ionb_abs_diff_Q3_norm
                np.mean(abs_diff_b),  # ionb_mean_abs_diff_norm
                np.std(abs_diff_b),  # ionb_std_abs_diff_norm
                np.min(abs_diff_y),  # iony_min_abs_diff_norm
                np.max(abs_diff_y),  # iony_max_abs_diff_norm
                np.quantile(abs_diff_y, 0.25),  # iony_abs_diff_Q1_norm
                np.quantile(abs_diff_y, 0.5),  # iony_abs_diff_Q2_norm
                np.quantile(abs_diff_y, 0.75),  # iony_abs_diff_Q3_norm
                np.mean(abs_diff_y),  # iony_mean_abs_diff_norm
                np.std(abs_diff_y),  # iony_std_abs_diff_norm
                np.dot(target_all, prediction_all),  # Dot product all ions
                np.dot(target_b, prediction_b),  # Dot product b ions
                np.dot(target_y, prediction_y),  # Dot product y ions
                _cosine_similarity(target_all, prediction_all),  # Cos similarity all ions
                _cosine_similarity(target_b, prediction_b),  # Cos similarity b ions
                _cosine_similarity(target_y, prediction_y),  # Cos similarity y ions
                _weighted_dot_product(mz_all, target_all, prediction_all,
                                      1.0, 0.5, True),  # sokalow weighted dot product all
                _weighted_dot_product(mz_b, target_b, prediction_b,
                                      1.0, 0.5, True),  # sokalow weighted dot product b ions
                _weighted_dot_product(mz_y, target_y, prediction_y,
                                      1.0, 0.5, True),  # sokalow weighted dot product y ions
                _spectrast_match(target_all, prediction_all),  # SpectraST all ions
                _spectrast_match(target_b, prediction_b),  # SpectraST b ions
                _spectrast_match(target_y, prediction_y),  # SpectraST y ions
                _spectra_angle_calc(target_all, prediction_all),  # spectral angle similarity all ions
                _spectra_angle_calc(target_b, prediction_b),  # spectral angle similarity b ions
                _spectra_angle_calc(target_y, prediction_y),  # spectral angle similarity y ions
                # Same features in normal space
                np.corrcoef(target_all_unlog, prediction_all_unlog)[0][1],  # Pearson all
                np.corrcoef(target_b_unlog, prediction_b_unlog)[0][1],  # Pearson b
                np.corrcoef(target_y_unlog, prediction_y_unlog)[0][1],  # Pearson y
                _spearman(target_all_unlog, prediction_all_unlog),  # Spearman all ions
                _spearman(target_b_unlog, prediction_b_unlog),  # Spearman b ions
                _spearman(target_y_unlog, prediction_y_unlog),  # Spearman y ions
                _mse(target_all_unlog, prediction_all_unlog),  # MSE all ions
                _mse(target_b_unlog, prediction_b_unlog),  # MSE b ions
                _mse(target_y_unlog, prediction_y_unlog),  # MSE y ions,
                # Ion type with min absolute difference
                0 if np.min(abs_diff_b_unlog) <= np.min(abs_diff_y_unlog) else 1,
                # Ion type with max absolute difference
                0 if np.max(abs_diff_b_unlog) >= np.max(abs_diff_y_unlog) else 1,
                np.min(abs_diff_all_unlog),  # min_abs_diff
                np.max(abs_diff_all_unlog),  # max_abs_diff
                np.quantile(abs_diff_all_unlog, 0.25),  # abs_diff_Q1
                np.quantile(abs_diff_all_unlog, 0.5),  # abs_diff_Q2
                np.quantile(abs_diff_all_unlog, 0.75),  # abs_diff_Q3
                np.mean(abs_diff_all_unlog),  # mean_abs_diff
                np.std(abs_diff_all_unlog),  # std_abs_diff
                np.min(abs_diff_b_unlog),  # ionb_min_abs_diff
                np.max(abs_diff_b_unlog),  # ionb_max_abs_diff_norm
                np.quantile(abs_diff_b_unlog, 0.25),  # ionb_abs_diff_Q1
                np.quantile(abs_diff_b_unlog, 0.5),  # ionb_abs_diff_Q2
                np.quantile(abs_diff_b_unlog, 0.75),  # ionb_abs_diff_Q3
                np.mean(abs_diff_b_unlog),  # ionb_mean_abs_diff
                np.std(abs_diff_b_unlog),  # ionb_std_abs_diff
                np.min(abs_diff_y_unlog),  # iony_min_abs_diff
                np.max(abs_diff_y_unlog),  # iony_max_abs_diff
                np.quantile(abs_diff_y_unlog, 0.25),  # iony_abs_diff_Q1
                np.quantile(abs_diff_y_unlog, 0.5),  # iony_abs_diff_Q2
                np.quantile(abs_diff_y_unlog, 0.75),  # iony_abs_diff_Q3
                np.mean(abs_diff_y_unlog),  # iony_mean_abs_diff
                np.std(abs_diff_y_unlog),  # iony_std_abs_diff
                np.dot(target_all_unlog, prediction_all_unlog),  # Dot product all ions
                np.dot(target_b_unlog, prediction_b_unlog),  # Dot product b ions
                np.dot(target_y_unlog, prediction_y_unlog),  # Dot product y ions
                _cosine_similarity(target_all_unlog, prediction_all_unlog),  # Cos similarity all
                _cosine_similarity(target_b_unlog, prediction_b_unlog),  # Cos similarity b ions
                _cosine_similarity(target_y_unlog, prediction_y_unlog),  # Cos similarity y ions
                _weighted_dot_product(mz_all, target_all_unlog, prediction_all_unlog,
                                      1.0, 0.5, True),  # sokalow weighted dot product all
                _weighted_dot_product(mz_b, target_b_unlog, prediction_b_unlog,
                                      1.0, 0.5, True),  # sokalow weighted dot product b ions
                _weighted_dot_product(mz_y, target_y_unlog, prediction_y_unlog,
                                      1.0, 0.5, True),  # sokalow weighted dot product y ions
                _spectrast_match(target_all_unlog, prediction_all_unlog),  # SpectraST all ions
                _spectrast_match(target_b_unlog, prediction_b_unlog),  # SpectraST b ions
                _spectrast_match(target_y_unlog, prediction_y_unlog),  # SpectraST y ions
                _spectra_angle_calc(target_all_unlog, prediction_all_unlog),  # spectral angle similarity all ions
                _spectra_angle_calc(target_b_unlog, prediction_b_unlog),  # spectral angle similarity b ions
                _spectra_angle_calc(target_y_unlog, prediction_y_unlog),  # spectral angle similarity y ions
                _nist_ms_match(mz_all, target_all_unlog, prediction_all_unlog,
                               (1.0, 0.0), (0.5, 1.0), 0.0),  # nist match factor (only unlog space, only all ions)
            ]

        features = dict(
            zip(
                self.feature_names,
                [0.0 if np.isnan(ft) else ft for ft in feature_values],
            )
        )

        return features


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation."""
    x = np.array(x)
    y = np.array(y)
    x_rank = pd.Series(x).rank()
    y_rank = pd.Series(y).rank()
    return np.corrcoef(x_rank, y_rank)[0][1]


def _mse(x: np.ndarray, y: np.ndarray) -> float:
    """Mean squared error"""
    x = np.array(x)
    y = np.array(y)
    return np.mean((x - y) ** 2)


def _cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    """Cosine similarity"""
    x = np.array(x)
    y = np.array(y)
    return np.dot(x, y) / (np.linalg.norm(x, 2) * np.linalg.norm(y, 2))


def _weighted_dot_product(mz: np.ndarray, 
                          intens_exp: np.ndarray, intens_lib: np.ndarray, 
                          mz_weight: float = 0.0, intens_weight: float = 1.0,
                          normalize_results: bool = True) -> float:
    """
    Compute a weighted dot product similarity between experimental and library spectra.

    Parameters
    ----------
    mz : np.ndarray
        m/z values of the experimental spectrum.
    intens_exp : np.ndarray
        intensity values of the experimental spectrum.
    intens_lib : np.ndarray
        intensity values of the library spectrum.
    mz_weight : float, optional
        exponent applied to m/z weighting (default is 0.0). Set to 1.0 for 
        Sokalow weighting.
    intens_weight : float, optional
        exponent applied to intensity weighting (default is 1.0). Set to 0.5 for
        Sokalow weighting.
    normalize_results : bool, optional
        whether to normalize the result to the cosine similarity form (default is True).

    Returns
    -------
    float
        Weighted (optionally normalized) dot product similarity between the spectra.
    """
    intens_lib = np.array(intens_lib)
    intens_exp = np.array(intens_exp)
    mz = np.array(mz)
    # compute weighted components
    alpha = (intens_exp ** intens_weight) * (mz ** mz_weight)
    beta = (intens_lib ** intens_weight) * (mz ** mz_weight)

    # compute raw dot product
    dot_product = np.dot(alpha, beta)

    # optionally normalize to unit length
    if normalize_results:
        dot_product /= (np.linalg.norm(alpha, ord=2) * np.linalg.norm(beta, ord=2))

    return dot_product


def _spectrast_match(x: np.ndarray, y: np.ndarray) -> float:
    """SpectraST Match Factor
    Lam, H.; Deutsch, E. W.; Eddes, J. S.; Eng, J. K.; King, N.; Stein, S. E.; 
    Aebersold, R. Development and Validation of a Spectral Library Searching Method 
    for Peptide Identification from MS/MS. Proteomics 2007, 7, 655-667.
    """
    x = np.array(x)
    y = np.array(y)
    x_norm = np.sqrt(np.sum(x ** 2))
    y_norm = np.sqrt(np.sum(y ** 2))
    x /= x_norm
    y /= y_norm

    # compute SpectraST match factor
    return (np.dot(x, y) ** 2) / (np.sum(x ** 2) * np.sum(y ** 2))


def _spectra_angle_calc(exp: np.ndarray, lib: np.ndarray, eps: float = 1e-7) -> float:
    """
    Compute the spectral angle similarity between two spectra (Spectral Angle Mapper).

    References
    ----------
    Toprak, U. H.; Gillet, L. C.; Maiolica, A.; Navarro, P.; Leitner, A.; 
    Aebersold, R. Conserved Peptide Fragmentation as a Benchmarking Tool 
    for Mass Spectrometers and a Discriminating Feature for Targeted 
    Proteomics. Molecular & Cellular Proteomics 2014, 13, 2056–2071.
    https://doi.org/10.1074/mcp.O113.036475

    Parameters
    ----------
    exp : np.ndarray
        Experimental spectrum intensities.
    lib : np.ndarray
        Library spectrum intensities.
    eps : float, optional
        Small constant to prevent division by zero (default is 1e-7).

    Returns
    -------
    float
        Spectral angle similarity score in [0, 1], where 1 indicates identical spectra.
    """
    # normalize both spectra to unit length with epsilon safeguard
    exp_norm = exp / np.sqrt(np.maximum(np.sum(exp ** 2), eps))
    lib_norm = lib / np.sqrt(np.maximum(np.sum(lib ** 2), eps))

    # compute cosine similarity (clipped for numerical safety)
    cos_sim = np.clip(np.dot(exp_norm, lib_norm), -1.0, 1.0)

    # convert to spectral angle similarity as in Toprak et al. (2014)
    return 1 - (2 * np.arccos(cos_sim) / np.pi)


def _nist_weights(mz: np.ndarray, intens: np.ndarray, mz_weight: float, intens_weight: float) -> np.ndarray:
    """
    Apply NIST-style weighting to a spectrum with a shared m/z axis.

    Parameters
    ----------
    mz : np.ndarray
        Shared m/z values.
    intens : np.ndarray
        Intensity values corresponding to m/z.
    mz_weight : float
        Exponent weight applied to m/z values.
    intens_weight : float
        Exponent weight applied to intensity values.

    Returns
    -------
    np.ndarray
        Weighted intensities for NIST dot product and RPP computations.
    """
    return (intens ** intens_weight) * (mz ** mz_weight)


def _nist_ms_match(mz: np.ndarray, intens_exp: np.ndarray, intens_lib: np.ndarray, 
                   mz_weights: tuple[float, float] = (1.0, 0.0),
                   intens_weights: tuple[float, float] = (0.5, 1.0),
                   noise_thres: float = 0.0) -> float:
    """
    Compute the NIST MS² match factor between experimental and library spectra.

    References
    ----------
    Stein, S. E.; Scott, D. R. (1994). Optimization and testing of mass spectral
    library search algorithms for compound identification.
    Journal of the American Society for Mass Spectrometry, 5, 859–866.

    Parameters
    ----------
    mz : np.ndarray
        Shared m/z axis for both spectra (unsorted).
    intens_exp : np.ndarray
        Experimental spectrum intensities aligned to m/z.
    intens_lib : np.ndarray
        Library spectrum intensities aligned to m/z.
    mz_weights : tuple[float, float], optional
        Exponential weights for m/z in (DPC, RPP) components (default is (1.0, 0.0)).
    intens_weights : tuple[float, float], optional
        Exponential weights for intensity in (DPC, RPP) components (default is (0.5, 1.0)).
    noise_thres : float, optional
        Threshold below which peaks are treated as noise (default is 0.0).

    Returns
    -------
    float
        NIST MS² match factor (0-1 scale), combining DPC and RPP similarity components.
    """
    # sort all arrays by increasing m/z
    sort_idx = np.argsort(mz)
    mz = mz[sort_idx]
    intens_exp = intens_exp[sort_idx]
    intens_lib = intens_lib[sort_idx]

    # weighted dot product cosine (DPC)
    w_exp_dpc = _nist_weights(mz, intens_exp, mz_weights[0], intens_weights[0])
    w_lib_dpc = _nist_weights(mz, intens_lib, mz_weights[0], intens_weights[0])

    numerator = np.dot(w_exp_dpc, w_lib_dpc) ** 2
    denominator = np.sum(w_exp_dpc ** 2) * np.sum(w_lib_dpc ** 2)
    dpc = numerator / denominator if denominator > 0 else 0.0

    # weighted ratio of peak pairs (RPP)
    w_exp_rpp = _nist_weights(mz, intens_exp, mz_weights[1], intens_weights[1])
    w_lib_rpp = _nist_weights(mz, intens_lib, mz_weights[1], intens_weights[1])

    # identify non-noise peaks
    valid_mask = (intens_exp > noise_thres) & (intens_lib > noise_thres)
    idx = np.nonzero(valid_mask)[0]

    if idx.size > 1:
        n_exp = np.count_nonzero(intens_exp > noise_thres)

        # compute RPP ratios between consecutive common peaks
        ratios = ((w_lib_rpp[idx[1:n_exp]] / w_lib_rpp[idx[0: n_exp - 1]]) * 
                  (w_exp_rpp[idx[0: n_exp - 1]] / w_exp_rpp[idx[1:n_exp]]))
        signs = np.where(ratios < 1, 1, -1)
        rpp = np.sum(ratios ** signs) / n_exp

        # final NIST match factor
        match_factor = (n_exp * dpc + (idx.size * rpp)) / (n_exp + idx.size)
    else:
        match_factor = 0.0

    return match_factor
