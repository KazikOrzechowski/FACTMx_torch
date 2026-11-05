"""GradNorm works with actual v0.1.4 CNA heads and masked shared encoder."""
import torch
import pytest
from factmx_torch import MissingViewFACTMx, MolecularGradNorm

@pytest.mark.parametrize('head_type',
    ['Multinomial','MultinomialCenteredOneHot','MultinomialFrequencyStandardized'])
def test_actual_v014_cna_heads_and_gaussians_gradnorm_one_step(head_type):
    torch.manual_seed(83)
    n=12
    rna=torch.randn(n,4)
    meth=torch.randn(n,4)
    y=torch.randint(0,5,(n,6))
    onehot=torch.nn.functional.one_hot(y,5).float()
    counts=onehot.sum(-1)
    configs=[
      {'head_type':'MultiNormal','head_name':'RNA','dim':4,'eps':0.2},
      {'head_type':head_type,'head_name':'CNA','dim':30,'dim_pos':6,'dim_cat':5,'eps':0.01},
      {'head_type':'MultiNormal','head_name':'METH','dim':4,'eps':0.2},
    ]
    model=MissingViewFACTMx(dim_latent=3,heads_config=configs,
          encoder_config={'encoder_type':'Linear','scale_parameterization':'softplus'})
    if head_type!='Multinomial':
        model.heads[1].fit_category_frequencies((onehot,counts))
    model.set_encoder_centers(torch.zeros(sum(model.head_dims)))
    x=[rna,(onehot,counts),meth]
    mask=torch.ones((n,3),dtype=torch.bool)
    latent,loc,scale,encoded=model.infer_masked(x,mask,deterministic=True)
    losses={head.head_name:head.loss(view,loc,encoded[j])*0.15
            for j,(head,view) in enumerate(zip(model.heads,x))}
    # Native losses are positive in this synthetic test. A separate test
    # covers nonpositive Gaussian NLL and the explicit positive proxy route.
    if any(float(loss)<0 for loss in losses.values()):
        # A fixed positive MSE for progress, never for the model's loss.
        progress={'RNA':(model.heads[0].loc(loc)-rna).square().mean()+.01,
                  'CNA':losses['CNA'].abs()+.01,
                  'METH':(model.heads[2].loc(loc)-meth).square().mean()+.01}
    else: progress=None
    gn=MolecularGradNorm(task_names=['RNA','CNA','METH'],alpha=1.5,lr=.02)
    weighted=gn.weighted_loss(losses)
    before=model.encoder.loc.weight.detach().clone()
    report=gn.queue_update(losses,model.encoder.loc.parameters(),progress_losses=progress)
    assert report['status']=='queued'
    optim=torch.optim.SGD(model.parameters(),lr=0.003)
    optim.zero_grad(set_to_none=True)
    weighted.backward()
    assert model.encoder.loc.weight.grad is not None
    optim.step(); gn.step()
    assert torch.isfinite(model.encoder.loc.weight).all()
    assert not torch.allclose(model.encoder.loc.weight,before)
    assert sum(gn.weight_dict().values())==pytest.approx(3.,abs=1e-5)
