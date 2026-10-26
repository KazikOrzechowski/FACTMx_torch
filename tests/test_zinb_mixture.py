"""Focused tests for the regularized count mixture head."""
from __future__ import annotations

import math

import pytest
import torch

from factmx_torch import FACTMxModel, ZINBMixtureHead


def _data(n=5, cells=8, genes=4, libraries=2):
    torch.manual_seed(72)
    sf = torch.rand(n, cells) * 1.6 + 0.3
    gene_rates = torch.tensor([0.3, 2.0, 6.0, 0.8])[:genes]
    counts = torch.poisson(sf[..., None] * gene_rates)
    batch = torch.randint(libraries, (n, cells))
    return counts, sf, batch


def _head(**kwargs):
    return ZINBMixtureHead(
        dim=3, dim_latent=2, dim_genes=4, head_name="rna",
        n_libraries=2, assignment_estimator="mean", **kwargs,
    )


def test_shapes_and_gradients():
    head = _head(inflation="gene_component", initial_inflation="gene", contextual_gate=True,
                 contextual_encoder="deepsets")
    data = _data()
    latent = torch.randn(5, 2, requires_grad=True)
    enc = head.encode(data, deterministic=True)
    assert enc["assignment"].shape == (5, 8, 3)
    assert enc["encoder_input"].shape == (5, 3)
    assert head.component_log_likelihood(data).shape == (5, 8, 3)
    assert head.contextual_gate_probs(latent, data).shape == (5, 3)
    loss = head.loss(data, latent, enc)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(latent.grad).all()
    assert head.log_component_mean.grad.abs().sum() > 0
    assert head.gene_inflation_logits.grad.abs().sum() > 0
    assert head.batch_log_effects.grad.abs().sum() > 0
    assert head.component_inflation_deviations.grad is None
    # Activating component inflation permits its own constrained gradients.
    head.zero_grad(set_to_none=True)
    head.set_inflation_mode("gene_component")
    head.loss(data, latent.detach(), head.encode(data, deterministic=True)).backward()
    assert head.component_inflation_deviations.grad.abs().sum() > 0


def test_nb_zero_probability_and_positive_counts():
    h = ZINBMixtureHead(dim=1, dim_latent=2, dim_genes=1,
                        head_name="x", inflation="none")
    with torch.no_grad():
        h.log_component_mean.fill_(math.log(2.5))
        h.raw_dispersion.fill_(math.log(math.expm1(4.0)))
    counts = torch.tensor([[[0.], [3.]]])
    sf = torch.ones(1, 2)
    expected = torch.distributions.NegativeBinomial(
        total_count=torch.tensor(4.0),
        probs=torch.tensor(2.5 / (4.0 + 2.5)),
    ).log_prob(counts)
    got = h.component_log_likelihood((counts, sf))[0, :, 0]
    torch.testing.assert_close(got, expected.reshape(-1), atol=1e-5, rtol=1e-5)


def test_zinb_zero_and_nonzero_probability():
    head = ZINBMixtureHead(dim=1, dim_latent=2, dim_genes=1,
                           head_name="rna", inflation="gene", initial_pi=0.2)
    with torch.no_grad():
        head.log_component_mean.fill_(math.log(3.0))
    x = torch.tensor([[[0.], [2.]]])
    sf = torch.ones(1, 2)
    head.set_inflation_mode("none")
    nb = head.component_log_likelihood((x, sf))[0, :, 0]
    head.set_inflation_mode("gene")
    zinb = head.component_log_likelihood((x, sf))[0, :, 0]
    expected_zero = torch.logaddexp(
        torch.tensor(math.log(0.2)), math.log(0.8) + nb[0]
    )
    torch.testing.assert_close(zinb[0], expected_zero)
    torch.testing.assert_close(zinb[1], math.log(0.8) + nb[1])


def test_batch_offsets_are_centered_and_shrunk():
    head = _head(inflation="none")
    with torch.no_grad():
        head.batch_log_effects[0].fill_(0.9)
        head.batch_log_effects[1].fill_(-0.1)
    assert torch.allclose(head.batch_effects().mean(0), torch.zeros(4), atol=1e-7)
    assert head.regularization_loss() > 0
    x, sf, batch = _data()
    both = head.component_log_likelihood((x, sf, batch))
    reverse = head.component_log_likelihood((x, sf, 1 - batch))
    assert not torch.allclose(both, reverse)


def test_count_validation():
    head = _head(inflation="none")
    counts, sf, lib = _data()
    bad = counts.clone()
    bad[0, 0, 0] += 0.4
    with pytest.raises(ValueError, match="raw integer"):
        head.component_log_likelihood((bad, sf, lib))
    with pytest.raises(ValueError, match="library range"):
        head.component_log_likelihood((counts, sf, lib + 2))
    with pytest.raises(ValueError, match="size_factors"):
        head.component_log_likelihood((counts, -sf, lib))
    with pytest.raises(ValueError, match="required"):
        head.component_log_likelihood((counts, sf))


def test_initializer_training_only_and_saves_mode(tmp_path):
    model = FACTMxModel(
        dim_latent=2,
        heads_config=[{
            "head_type": "ZINBMixture", "head_name": "rna", "dim": 3,
            "dim_genes": 4, "n_libraries": 2,
            "inflation": "gene_component", "initial_inflation": "none",
            "assignment_estimator": "mean", "contextual_gate": True,
        }],
        encoder_config={"encoder_type": "Linear", "scale_parameterization": "softplus"},
    )
    counts = _data(n=6)
    head = model.heads[0]
    info = head.initialize_from_counts(counts, max_init_cells=25, seed=5)
    assert info["cells_used"] == 25
    assert torch.count_nonzero(head.mixture_logits.weight) == 0
    assert head.inflation_mode == "none"
    result = model.fit([counts], epochs=1, batch_size=3, lr=1e-3)
    assert len(result) == 2 and all(math.isfinite(loss) for loss in result)
    head.set_inflation_mode("gene")
    model.save_checkpoint(str(tmp_path / "test.pt"))
    restored = FACTMxModel.load_checkpoint(str(tmp_path / "test.pt"))
    assert restored.heads[0].inflation_mode == "gene"
    torch.testing.assert_close(
        model.loss([counts], deterministic_latent=True),
        restored.loss([counts], deterministic_latent=True),
    )


def test_exact_mixture_marginal():
    h = _head(inflation="gene", initial_inflation="none")
    x = _data(n=3)
    latent = torch.randn(3, 2)
    component_ll = h.component_log_likelihood(x)
    gates = h.decode_mixture_logits(latent).log_softmax(-1)
    reference = -torch.logsumexp(component_ll + gates[:, None, :], -1).mean()
    torch.testing.assert_close(h.loss(x, latent, h.encode(x, deterministic=True)), reference)
    assert h.inflation_mode == "none"


def test_predictive_loss_context_only_and_explicit_sampling():
    h = _head(inflation="gene", contextual_gate=True)
    query = _data(n=3, cells=4)
    context = _data(n=3, cells=5)
    z = torch.randn(3, 2)
    nll = h.predictive_loss(query, z, context)
    assert torch.isfinite(nll)
    samples = h.sample_counts(z, query[1], query[2])
    assert samples.shape == query[0].shape
    assert torch.all(samples >= 0)
    assert torch.all(samples == samples.round())


def test_opt_fit_and_missing_view_encoder():
    from factmx_torch import MissingViewFACTMx
    data = _data(n=8)
    model = MissingViewFACTMx(
        dim_latent=2,
        heads_config=[{"head_type": "ZINBMixture", "head_name": "rna", "dim": 3,
                       "dim_genes": 4, "n_libraries": 2, "inflation": "gene",
                       "assignment_estimator": "mean"}],
        encoder_config={"encoder_type": "Linear", "scale_parameterization": "softplus"},
    )
    history = model.fit_optimized([data], epochs=1, batch_size=4,
                                  structured_hold_epochs=0, simple_ramp_epochs=0)
    assert len(history["history"]) == 1
    assert math.isfinite(history["history"][0]["train_objective"])
    latent = model.get_latent_representation([data])
    assert latent.shape == (8, 2)


def test_local_encoder_uses_batch_adjusted_count_rates():
    head = _head(inflation="none")
    with torch.no_grad():
        head.batch_log_effects[0].fill_(math.log(2.0))
        head.batch_log_effects[1].fill_(-math.log(2.0))
    counts = torch.tensor([[[8., 8., 8., 8.], [2., 2., 2., 2.]]])
    sf = torch.ones(1, 2)
    library = torch.tensor([[0, 1]])
    corrected = head.context_encoder_input((counts, sf, library))
    torch.testing.assert_close(corrected[:, 0], corrected[:, 1], atol=1e-6, rtol=1e-6)


def test_batch_only_model_missing_view_forward():
    from factmx_torch import MissingViewFACTMx
    data = _data(n=4)
    model = MissingViewFACTMx(
        dim_latent=2,
        heads_config=[{"head_type": "ZINBMixture", "head_name": "rna", "dim": 3,
                       "dim_genes": 4, "n_libraries": 2}],
        encoder_config={"encoder_type": "Linear", "scale_parameterization": "softplus"},
    )
    model.set_encoder_centers(torch.zeros(model.head_dims[0]), torch.zeros(model.head_dims[0]))
    masks = torch.tensor([[True], [False], [True], [False]])
    z, loc, scale, encoded = model.infer_masked([data], masks, deterministic=True)
    assert z.shape == (4, 2)
    assert torch.isfinite(loc).all()
