"""Regression tests for the two experimental TCGA CNA encoder heads."""
from __future__ import annotations

import io
import math

import pytest
import torch

from factmx_torch import (
    CenteredOneHotMultinomialHead,
    FACTMxModel,
    FrequencyStandardizedMultinomialHead,
    MissingViewFACTMx,
    MultinomialHead,
    build_head,
)


def examples():
    # The third CNA locus deliberately has a rare class and the fourth can be
    # fully unobserved in a patient to exercise the neutral-position behavior.
    labels = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3], [1, 0, 2, 3], [0, 1, 2, 3]], dtype=torch.long)
    obs = torch.nn.functional.one_hot(labels, num_classes=5).to(torch.float32)
    counts = obs.sum(-1)
    return obs, counts


def make(head_type, **kw):
    return build_head({"head_type": head_type, "head_name": "cna", "dim":20,
                       "dim_pos":4, "dim_cat":5, "eps":.01, **kw}, dim_latent=3)


def test_fit_is_required():
    head=make("MultinomialCenteredOneHot")
    with pytest.raises(RuntimeError, match="fit_category_frequencies"):
        head.encode(examples())


def test_centered_onehot_matches_train_frequency_and_mean_zero():
    obs, counts=examples()
    head=make("MultinomialCenteredOneHot")
    report=head.fit_category_frequencies((obs, counts))
    assert report["n_training_patients_observed"] == 4
    actual=head.encode((obs,counts))["encoder_input"].reshape(obs.shape)
    p=obs.mean(0)
    torch.testing.assert_close(head.category_frequencies,p)
    torch.testing.assert_close(actual,obs-p)
    torch.testing.assert_close(actual.mean(0), torch.zeros_like(p), atol=1e-7,rtol=0)
    torch.testing.assert_close(actual.sum(-1),torch.zeros_like(counts),atol=1e-6,rtol=0)


def test_masked_frequency_fit_excludes_unobserved_and_validation():
    obs,counts=examples()
    train_obs=torch.cat([obs, torch.full((1,4,5),float("nan"))],0)
    train_counts=torch.cat([counts, torch.full((1,4),float("nan"))],0)
    head=make("MultinomialCenteredOneHot")
    report=head.fit_category_frequencies((train_obs,train_counts), patient_mask=torch.tensor([1,1,1,1,0],dtype=torch.bool))
    assert report["n_training_patients_observed"]==4
    torch.testing.assert_close(head.category_frequencies,obs.mean(0))
    test_obs=torch.nn.functional.one_hot(torch.full((2,4),4,dtype=torch.long),5).float()
    freq_before=head.category_frequencies.clone()
    head.encode((test_obs,test_obs.sum(-1)))
    torch.testing.assert_close(head.category_frequencies,freq_before)


def test_missing_locus_neutral_zero_and_empty_training_locus_uniform():
    obs, counts=examples()
    obs[:,3]=0
    counts[:,3]=0
    head=make("MultinomialCenteredOneHot")
    report=head.fit_category_frequencies((obs,counts))
    assert report["empty_training_positions"]==1
    torch.testing.assert_close(head.category_frequencies[3],torch.full((5,),.2))
    out=head.encode((obs,counts))["encoder_input"].reshape(obs.shape)
    assert bool((out[:,3]==0).all())


def test_frequency_standardized_variance_floor_and_cap():
    obs,counts=examples()
    head=make("MultinomialFrequencyStandardized",variance_floor=.01,max_inverse_std=3.)
    head.fit_category_frequencies((obs,counts))
    p=obs.mean(0)
    weights=torch.rsqrt(p*(1-p)+.01).clamp(max=3.)
    expected=(obs-p)*weights
    output=head.encode((obs,counts))["encoder_input"].reshape(obs.shape)
    torch.testing.assert_close(output,expected)
    assert bool(torch.isfinite(output).all())
    with pytest.raises(ValueError,match="variance_floor"):
        make("MultinomialFrequencyStandardized",variance_floor=0)
    with pytest.raises(ValueError,match="max_inverse_std"):
        make("MultinomialFrequencyStandardized",max_inverse_std=float("inf"))


def test_log_onehot_is_scaled_centered_onehot_for_one_hot():
    obs,counts=examples()
    original=make("Multinomial")
    centered=make("MultinomialCenteredOneHot")
    centered.fit_category_frequencies((obs,counts))
    x0=original.encode((obs,counts))["encoder_input"].reshape(obs.shape)
    x1=centered.encode((obs,counts))["encoder_input"].reshape(obs.shape)
    amp=math.log((1.+original.eps)/original.eps)
    torch.testing.assert_close(x0-x0.mean(0),x1*amp,rtol=1e-5,atol=2e-6)


@pytest.mark.parametrize("head_type",["MultinomialCenteredOneHot","MultinomialFrequencyStandardized"])
def test_loss_and_decoder_are_unchanged(head_type):
    obs,counts=examples()
    torch.manual_seed(29)
    baseline=make("Multinomial")
    variant=make(head_type)
    variant.fit_category_frequencies((obs,counts))
    with torch.no_grad():
        variant.decoder.load_state_dict(baseline.decoder.state_dict())
    z=torch.randn(4,3,requires_grad=True)
    log_base=baseline.decode_params(z)
    log_new=variant.decode_params(z)
    torch.testing.assert_close(log_base,log_new,rtol=0,atol=0)
    b=baseline.loss((obs,counts),z,{})
    v=variant.loss((obs,counts),z,{})
    torch.testing.assert_close(b,v,atol=0,rtol=0)
    grad_b=torch.autograd.grad(b,z,retain_graph=True)[0]
    grad_v=torch.autograd.grad(v,z)[0]
    torch.testing.assert_close(grad_b,grad_v,atol=0,rtol=0)


@pytest.mark.parametrize("head_type",["MultinomialCenteredOneHot","MultinomialFrequencyStandardized"])
def test_state_dict_preserves_frequencies_and_model_checkpoint(tmp_path,head_type):
    obs,counts=examples()
    cfg={"head_type":head_type,"head_name":"cna","dim":20,"dim_pos":4,"dim_cat":5,"eps":0.01}
    model=FACTMxModel(dim_latent=3,heads_config=[cfg])
    head=model.heads[0]
    head.fit_category_frequencies((obs,counts))
    before=head.encode((obs,counts))["encoder_input"].clone()
    path=tmp_path/"model.pt"
    model.save_checkpoint(str(path))
    loaded=FACTMxModel.load_checkpoint(str(path))
    assert isinstance(loaded.heads[0],type(head))
    assert bool(loaded.heads[0].category_frequencies_fitted)
    torch.testing.assert_close(loaded.heads[0].category_frequencies,head.category_frequencies)
    torch.testing.assert_close(loaded.heads[0].encode((obs,counts))["encoder_input"],before)


@pytest.mark.parametrize("head_type",["MultinomialCenteredOneHot","MultinomialFrequencyStandardized"])
def test_missing_view_encoder_and_gradients(head_type):
    obs, counts=examples()
    configs=[
        {"head_type":"MultiNormal","head_name":"rna","dim":2,"eps":.05},
        {"head_type":head_type,"head_name":"cna","dim":20,"dim_pos":4,"dim_cat":5},
    ]
    torch.manual_seed(3)
    model=MissingViewFACTMx(dim_latent=3,heads_config=configs,
                            encoder_config={"encoder_type":"Linear","scale_parameterization":"softplus"})
    model.heads[1].fit_category_frequencies((obs,counts))
    rna=torch.randn(4,2)
    center=torch.cat([rna.mean(0),model.heads[1].encode((obs,counts))["encoder_input"].mean(0)])
    model.set_encoder_centers(center)
    mask=torch.tensor([[1,1],[1,0],[0,1],[1,1]],dtype=torch.bool)
    latent,loc,scale,_ = model.infer_masked([rna,(obs,counts)], mask, deterministic=True)
    assert loc.shape==(4,3)
    assert bool(torch.isfinite(loc).all())
    # The masked API explicitly exposes infer_masked; reconstruction losses
    # are evaluated using the resulting latent, exactly as in the TCGA runner.
    rna_loss=model.heads[0].loss(rna,loc,{})
    cna_loss=model.heads[1].loss((obs,counts),loc,{})
    (rna_loss*0.5+cna_loss*0.25).backward()
    assert bool(torch.isfinite(rna_loss) and torch.isfinite(cna_loss))
    assert bool(torch.isfinite(model.encoder.loc.weight.grad).all())


def test_bad_frequency_inputs_rejected():
    obs,counts=examples()
    for x,c in [(obs,counts+1),(obs.nan_to_num()+float('nan'),counts),(-obs,counts)]:
        head=make('MultinomialCenteredOneHot')
        with pytest.raises(ValueError):
            head.fit_category_frequencies((x,c))
    with pytest.raises(ValueError,match="dim"):
        build_head({"head_type":"MultinomialCenteredOneHot","head_name":"x","dim":15,"dim_pos":4,"dim_cat":5},dim_latent=2)
    with pytest.raises(ValueError,match="encode_logits"):
        make("MultinomialCenteredOneHot",encode_logits=True)
