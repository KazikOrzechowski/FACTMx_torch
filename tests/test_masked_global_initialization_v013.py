"""Regression tests against the archived real-cohort masked adapter.

The frozen reference initializer is copied verbatim from the previously used
TCGA/IMMUCAN runner module, not independently reimplemented for these tests.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from factmx_torch import (
    FACTMxModel,
    MissingViewFACTMx,
    __version__,
    initialize_global_covariance,
    initialize_masked_global_covariance,
)

sys.path.insert(0, str(Path(__file__).parent))
from reference_masked_initializer_v012 import initialize_masked_global_covariance as reference_initializer


def make_case(seed: int = 7):
    torch.manual_seed(seed)
    model = MissingViewFACTMx(
        dim_latent=3,
        heads_config=[
            {"head_type": "MultiNormal", "head_name": "rna", "dim": 5},
            {"head_type": "MultiNormal", "head_name": "methylation", "dim": 4},
            {
                "head_type": "GMM", "head_name": "he", "dim": 3,
                "dim_normal": 2, "assignment_estimator": "mean",
                "objective": "marginal", "covariance_parameterization": "softplus",
            },
        ],
        encoder_config={"encoder_type": "Linear", "scale_parameterization": "softplus"},
        availability_to_scale=True,
    )
    with torch.no_grad():
        model.heads[2].mixture_logits.weight.normal_(0.0, 0.02)
        model.heads[2].mixture_logits.bias.zero_()
    rng = np.random.default_rng(seed)
    n = 30
    mask = rng.random((n, 3)) > np.asarray([.13, .23, .31])
    # Independent per-head encoder summaries; missing rows may be arbitrary.
    summaries = [
        rng.normal(size=(n, 5)).astype('float32'),
        rng.normal(size=(n, 4)).astype('float32'),
        rng.normal(size=(n, 3)).astype('float32'),
    ]
    targets = {
        'rna': (rng.normal(size=(n, 5)) * 2.0 + 1.0).astype('float32'),
        'methylation': (rng.normal(size=(n, 4)) * 0.6 - 2.0).astype('float32'),
    }
    for j, name in enumerate(('rna', 'methylation')):
        targets[name][~mask[:, j]] = np.nan
    # The centers are estimated from observed training heads only, as runners do.
    centers = np.concatenate([x[mask[:, j]].mean(axis=0) for j, x in enumerate(summaries)])
    model.set_encoder_centers(torch.from_numpy(centers))
    probs = {'he': np.asarray([0.7, 0.2, 0.1])}
    return model, summaries, mask, targets, probs


@pytest.mark.parametrize('weight_mode,bias_mode', [
    ('zero', 'population'),
    ('preserve_random', 'population'),
    ('preserve_random', 'zero'),
    ('zero', 'zero'),
])
def test_native_masked_matches_frozen_cohort_adapter(weight_mode, bias_mode):
    model, summaries, mask, targets, probs = make_case()
    reference = copy.deepcopy(model)
    native = copy.deepcopy(model)
    common = dict(
        simple_targets=targets,
        population_gate_probs=probs,
        ridge_alpha=1e-4,
        min_relative_singular=1e-5,
        max_whitening_gain=100.,
        gate_weight_mode=weight_mode,
        gate_bias_mode=bias_mode,
    )
    old_info = reference_initializer(reference, summaries, mask, **common)
    new_info = initialize_masked_global_covariance(native, summaries, mask, **common)
    assert old_info.keys() == new_info.keys()
    assert old_info['effective_initialized_rank'] == new_info['effective_initialized_rank']
    assert old_info['gate_bias_mode'] == new_info['gate_bias_mode']
    for key, old_val in reference.state_dict().items():
        torch.testing.assert_close(native.state_dict()[key], old_val,
                                   atol=0, rtol=0, msg=lambda msg: f'{key}: {msg}')
    assert new_info['method'] == old_info['method']
    assert new_info['gate_sources'] == old_info['gate_sources']
    assert new_info['latent_score_sd'] == pytest.approx(old_info['latent_score_sd'], rel=0, abs=0)
    assert new_info['gate_weight_diagnostics'] == old_info['gate_weight_diagnostics']
    # Model-level convenience method must apply exactly the same operation.
    method_model = copy.deepcopy(model)
    method_info = method_model.initialize_masked_global_covariance(summaries, mask, **common)
    assert method_info == new_info
    for key, val in native.state_dict().items():
        torch.testing.assert_close(method_model.state_dict()[key], val, atol=0, rtol=0)


def test_gate_policies_are_independent_and_do_not_consume_rng():
    initial, summaries, mask, targets, probs = make_case(10)
    final = {}
    for w in ('zero','preserve_random'):
        for b in ('zero','population'):
            model = copy.deepcopy(initial)
            before = torch.random.get_rng_state().clone()
            initialize_masked_global_covariance(
                model, summaries, mask,
                simple_targets=targets, population_gate_probs=probs,
                gate_weight_mode=w, gate_bias_mode=b,
            )
            torch.testing.assert_close(torch.random.get_rng_state(), before, atol=0, rtol=0)
            final[w, b] = model
    old_weight = initial.heads[2].mixture_logits.weight
    assert old_weight.abs().max().item() > 0
    for (w,b),model in final.items():
        gate = model.heads[2].mixture_logits
        if w == 'zero':
            assert torch.count_nonzero(gate.weight) == 0
        else:
            torch.testing.assert_close(gate.weight, old_weight, atol=0, rtol=0)
        if b == 'zero':
            assert torch.count_nonzero(gate.bias) == 0
        else:
            torch.testing.assert_close(gate.bias.softmax(0), torch.tensor([.7,.2,.1]), atol=1e-6, rtol=1e-6)
        for key in ('encoder.loc.weight', 'encoder.loc.bias', 'heads.0.loc.weight', 'heads.1.loc.bias'):
            torch.testing.assert_close(model.state_dict()[key], final['zero','population'].state_dict()[key], atol=0, rtol=0)


@pytest.mark.parametrize('bad,choice', [('weight','random_new'),('bias','preserve')])
def test_invalid_gate_mode_rejected_without_changing_model(bad, choice):
    model, summaries, mask, targets, probs = make_case()
    before = copy.deepcopy(model.state_dict())
    kwargs = {('gate_weight_mode' if bad == 'weight' else 'gate_bias_mode'):choice}
    with pytest.raises(ValueError, match='gate_'):
        initialize_masked_global_covariance(model, summaries, mask,
            simple_targets=targets, population_gate_probs=probs, **kwargs)
    for key in before:
        torch.testing.assert_close(model.state_dict()[key], before[key], atol=0, rtol=0)


def test_complete_data_gate_modes_and_fit_api():
    from test_initialization import _model, _data
    base = _model()
    data = _data()
    results = {}
    for w in ('zero','preserve_random'):
        for b in ('zero','population'):
            model = copy.deepcopy(base)
            gate_before = model.heads[2].mixture_logits.weight.detach().clone()
            init = model.initialize_global_covariance(data,
                population_gate_probs={'structured':[0.8,0.2]},
                gate_weight_mode=w, gate_bias_mode=b)
            assert init['gate_weight_mode'] == w
            assert init['gate_bias_mode'] == b
            if w=='preserve_random':
                torch.testing.assert_close(model.heads[2].mixture_logits.weight, gate_before, atol=0, rtol=0)
            else:
                assert torch.count_nonzero(model.heads[2].mixture_logits.weight) == 0
            if b == 'zero':
                assert torch.count_nonzero(model.heads[2].mixture_logits.bias) == 0
            else:
                torch.testing.assert_close(model.heads[2].mixture_logits.bias.softmax(0),
                                           torch.tensor([.8,.2]), atol=1e-6, rtol=1e-6)
            results[w,b] = model
    # Full-data zero-bias mode does not need population probabilities.
    model = copy.deepcopy(base)
    initialize_global_covariance(model, data, gate_weight_mode='preserve_random', gate_bias_mode='zero')
    assert torch.count_nonzero(model.heads[2].mixture_logits.bias) == 0
    model = copy.deepcopy(base)
    result = model.fit_optimized(data, epochs=1, batch_size=12, structured_hold_epochs=0,
        simple_ramp_epochs=0, kl_warmup_epochs=1, initialization='global_covariance',
        gate_weight_mode='preserve_random', gate_bias_mode='zero')
    assert result['initialization']['gate_weight_mode'] == 'preserve_random'
    assert result['initialization']['gate_bias_mode'] == 'zero'


def test_version():
    assert __version__ == '0.1.4.dev1'
