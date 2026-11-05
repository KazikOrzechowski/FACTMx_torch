import copy
import pytest
import torch
from torch import nn

from factmx_torch import MolecularGradNorm

NAMES = ("rna", "cna", "methylation")


def toy_losses(p):
    # Loss gradients at p=1: RNA=1, CNA=20, meth=2. Original losses >0.
    return {
        "rna": p * 1.0 + 100,
        "cna": p * 20.0 + 100,
        "methylation": p * 2.0 + 100,
    }


def test_initial_weights_and_no_gradnorm_network_weight_grads():
    p = nn.Parameter(torch.tensor(1.0))
    c = MolecularGradNorm(alpha=0.0)
    L = toy_losses(p)
    wloss = c.weighted_loss(L)
    torch.testing.assert_close(wloss, sum(L.values()))
    r = c.queue_update(L, [p])
    assert r["status"] == "queued"
    assert r["unweighted_grad_norms"]["cna"] == pytest.approx(20.0)
    assert r["unweighted_grad_norms"]["rna"] == pytest.approx(1.0)
    assert r["unweighted_grad_norms"]["methylation"] == pytest.approx(2.0)
    assert p.grad is None
    wloss.backward()
    torch.testing.assert_close(p.grad, torch.tensor(23.0))
    assert c.weights.grad is None
    assert c.step()
    assert c.weight_dict()["cna"] < 1.0
    assert c.weights.min().item() > 0
    assert c.weights.sum().item() == pytest.approx(3.0, abs=1e-6)


def test_gradnorm_with_constant_shared_model_redistributes_large_cna_gradient():
    p = nn.Parameter(torch.tensor(1.0))
    c = MolecularGradNorm(alpha=0.0, lr=0.03)
    for _ in range(80):
        L = toy_losses(p)
        c.queue_update(L, [p])
        c.step()
    w = c.weight_dict()
    assert w["cna"] < w["rna"]
    assert w["cna"] < w["methylation"]
    assert sum(w.values()) == pytest.approx(3.0, abs=1e-5)


def test_slow_task_gets_larger_training_rate_target():
    p = nn.Parameter(torch.tensor(1.0))
    c = MolecularGradNorm(alpha=1.5, lr=0.025)
    L = toy_losses(p)
    r0 = c.queue_update(L, [p], progress_losses={n: torch.tensor(10.0) for n in NAMES})
    assert r0["status"] == "queued"
    c.step()
    r1 = c.queue_update(L, [p], progress_losses={"rna": torch.tensor(9.0), "cna": torch.tensor(5.0), "methylation": torch.tensor(2.0)})
    assert r1["target_grad_norms"]["rna"] > r1["target_grad_norms"]["cna"]
    assert r1["target_grad_norms"]["cna"] > r1["target_grad_norms"]["methylation"]
    c.step()


def test_missing_view_skips_weight_update_but_network_loss_uses_observed():
    p = nn.Parameter(torch.tensor(1.0))
    c = MolecularGradNorm()
    L = toy_losses(p)
    mask = {"rna": True, "cna": False, "methylation": True}
    actual = c.weighted_loss(L, active=mask)
    torch.testing.assert_close(actual, L["rna"] + L["methylation"])
    r = c.queue_update(L, [p], active=mask)
    assert r["status"] == "skipped"
    assert r["reason"] == "not_all_simple_views_observed"
    assert not c.step()
    torch.testing.assert_close(c.weights, torch.ones(3))


def test_negative_nll_raises_without_positive_proxy_and_works_with_one():
    p = nn.Parameter(torch.tensor(1.0))
    c = MolecularGradNorm()
    L = toy_losses(p)
    L["rna"] = -p
    with pytest.raises(ValueError, match="positive progress losses"):
        c.queue_update(L, [p])
    r = c.queue_update(L, [p], progress_losses={n: torch.tensor(1.0) for n in NAMES})
    assert r["status"] == "queued"
    assert r["progress_source"] == "explicit_positive_proxy"
    c.step()


def test_state_roundtrip_restores_weights_rates_anchors_optimizer():
    p = nn.Parameter(torch.tensor(1.0))
    c = MolecularGradNorm(alpha=1.5, lr=0.01)
    c.queue_update(toy_losses(p), [p]); c.step()
    saved = copy.deepcopy(c.training_state_dict())
    restored = MolecularGradNorm(alpha=1.5, lr=0.01)
    restored.load_training_state_dict(saved)
    torch.testing.assert_close(c.weights, restored.weights)
    torch.testing.assert_close(c.initial_progress_losses, restored.initial_progress_losses)
    assert c.calls.item() == restored.calls.item()
    c.queue_update(toy_losses(p), [p]); c.step()
    restored.queue_update(toy_losses(p), [p]); restored.step()
    torch.testing.assert_close(c.weights, restored.weights)


def test_warmup_and_interval():
    p = nn.Parameter(torch.tensor(1.0))
    c = MolecularGradNorm(warmup_calls=2, update_interval=2)
    r = c.queue_update(toy_losses(p), [p]); assert r["reason"] == "warmup"
    r = c.queue_update(toy_losses(p), [p]); assert r["reason"] == "warmup"
    assert torch.isnan(c.initial_progress_losses).all()
    r = c.queue_update(toy_losses(p), [p]); assert r["status"] == "queued"
    c.step()
    r = c.queue_update(toy_losses(p), [p]); assert r["reason"] == "update_interval"
    r = c.queue_update(toy_losses(p), [p]); assert r["status"] == "queued"
    c.step()
    assert c.updates.item() == 2


def test_shared_layer_must_reach_all_heads():
    p = nn.Parameter(torch.tensor(1.0))
    other = nn.Parameter(torch.tensor(2.0))
    c = MolecularGradNorm()
    losses = toy_losses(p)
    losses["cna"] = other * 20 + 100
    with pytest.raises(RuntimeError, match="not connected"):
        c.queue_update(losses, [p])


def test_gradient_is_unaffected_by_weight_projection_after_weighted_loss():
    p = nn.Parameter(torch.tensor(1.0))
    c = MolecularGradNorm(alpha=0.)
    L = toy_losses(p)
    a = c.weighted_loss(L)
    c.queue_update(L, [p])
    c.step()  # intentionally before the model's backward: the snapshot is frozen
    a.backward()
    torch.testing.assert_close(p.grad, torch.tensor(23.))


def test_incomplete_mask_is_rejected():
    c = MolecularGradNorm()
    p = nn.Parameter(torch.tensor(1.0))
    with pytest.raises(KeyError, match="Missing activity flags"):
        c.weighted_loss(toy_losses(p), active={"rna": True})




def test_he_only_batch_has_zero_simple_loss_and_skips_weight_update():
    p = nn.Parameter(torch.tensor(1.))
    c = MolecularGradNorm()
    losses = toy_losses(p)
    missing = {name: False for name in NAMES}
    simple = c.weighted_loss(losses, active=missing)
    torch.testing.assert_close(simple, torch.zeros(()))
    assert simple.requires_grad
    result = c.queue_update(losses, [p], active=missing)
    assert result['status'] == 'skipped'
    assert c.step() is False
