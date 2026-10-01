from types import SimpleNamespace

import numpy as np
import pytest
from ms2rescore_rs import MS2Spectrum, Precursor
from psm_utils import PSM, PSMList

from ms2rescore.feature_generators.base import FeatureGeneratorException
from ms2rescore.feature_generators.ms2pip import MS2PIPFeatureGenerator


def _make_ms2_spectrum(identifier: str = "scan=1") -> MS2Spectrum:
    return MS2Spectrum(
        identifier=identifier,
        mz=[100.0, 200.0],
        intensity=[1000.0, 500.0],
        precursor=Precursor(mz=475.14, charge=2, rt=51.2),
    )


def _make_psm_list(with_spectrum: bool = True) -> PSMList:
    psm = PSM(peptidoform="PEPTIDE/2", spectrum_id="scan=1", run="run1")
    psm.rescoring_features = {}
    if with_spectrum:
        psm.spectrum = _make_ms2_spectrum()
    return PSMList(psm_list=[psm])


def test_ms2pip_feature_generator_uses_unified_correlate(monkeypatch):
    psm_list = _make_psm_list()
    captured = {}

    def fake_correlate(psms, **kwargs):
        captured["psms"] = psms
        captured["kwargs"] = kwargs
        return [
            SimpleNamespace(
                psm_index=0,
                predicted_intensity={"b": np.array([1.0]), "y": np.array([2.0])},
                observed_intensity={"b": np.array([3.0]), "y": np.array([4.0])},
            )
        ]

    def fake_feature_calculation(idx, pred_b, pred_y, obs_b, obs_y):
        captured["feature_inputs"] = (idx, pred_b, pred_y, obs_b, obs_y)
        return [(0, {"spec_pearson_norm": 0.91})]

    monkeypatch.setattr("ms2rescore.feature_generators.ms2pip.correlate", fake_correlate)
    monkeypatch.setattr(
        "ms2rescore.feature_generators.ms2pip.ms2pip_features_from_prediction_peak_arrays",
        fake_feature_calculation,
    )

    feature_generator = MS2PIPFeatureGenerator(model="HCD2021", processes=4)
    feature_generator.add_features(psm_list)

    assert captured["psms"] is psm_list
    assert "spectrum_file" not in captured["kwargs"]
    assert captured["kwargs"]["compute_correlations"] is False
    assert captured["kwargs"]["model"] == "HCD2021"
    assert "ms2_tolerance" not in captured["kwargs"]
    assert captured["kwargs"]["processes"] == 4
    assert psm_list[0].rescoring_features["spec_pearson_norm"] == 0.91


def test_ms2pip_feature_generator_requires_preloaded_spectra(monkeypatch):
    psm_list = _make_psm_list(with_spectrum=False)

    with pytest.raises(FeatureGeneratorException, match="preloaded on `psm.spectrum`"):
        MS2PIPFeatureGenerator().add_features(psm_list)


def _fake_result(with_mz: bool = True):
    return SimpleNamespace(
        psm_index=0,
        theoretical_mz=(
            {"b": np.array([98.06, 227.10]), "y": np.array([148.06, 263.09])} if with_mz else None
        ),
        predicted_intensity={"b": np.log2(np.array([0.1, 0.2]) + 0.001),
                             "y": np.log2(np.array([0.3, 0.4]) + 0.001)},
        observed_intensity={"b": np.log2(np.array([0.15, 0.1]) + 0.001),
                            "y": np.log2(np.array([0.35, 0.4]) + 0.001)},
    )


def test_ms2pip_feature_generator_adds_similarity_features(monkeypatch):
    psm_list = _make_psm_list()
    monkeypatch.setattr(
        "ms2rescore.feature_generators.ms2pip.correlate", lambda psms, **kwargs: [_fake_result()]
    )
    feature_generator = MS2PIPFeatureGenerator()
    feature_generator.add_features(psm_list)
    features = psm_list[0].rescoring_features
    for name in ["spectral_angle", "spectrast", "weighted_dotprod", "nist_match_factor"]:
        assert name in features and f"{name}_norm" in features
        assert 0 <= features[name] <= 1
    assert set(features) == set(feature_generator.feature_names)


def test_ms2pip_feature_generator_without_theoretical_mz(monkeypatch):
    psm_list = _make_psm_list()
    monkeypatch.setattr(
        "ms2rescore.feature_generators.ms2pip.correlate",
        lambda psms, **kwargs: [_fake_result(with_mz=False)],
    )
    MS2PIPFeatureGenerator().add_features(psm_list)
    features = psm_list[0].rescoring_features
    assert "spectral_angle" in features
    assert "nist_match_factor" not in features


@pytest.mark.parametrize("keep", [True, False])
def test_ms2pip_feature_generator_keep_predictions(monkeypatch, keep):
    psm_list = _make_psm_list()
    result = _fake_result()
    monkeypatch.setattr(
        "ms2rescore.feature_generators.ms2pip.correlate", lambda psms, **kwargs: [result]
    )
    feature_generator = MS2PIPFeatureGenerator(keep_predictions=keep)
    feature_generator.add_features(psm_list)
    if keep:
        assert list(feature_generator.predictions) == ["PEPTIDE/2"]
        prediction = feature_generator.predictions["PEPTIDE/2"]
        np.testing.assert_array_equal(prediction["theoretical_mz"]["b"], result.theoretical_mz["b"])
        np.testing.assert_array_equal(
            prediction["predicted_intensity"]["y"], result.predicted_intensity["y"]
        )
    else:
        assert feature_generator.predictions == {}
