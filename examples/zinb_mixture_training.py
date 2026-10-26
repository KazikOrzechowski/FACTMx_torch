"""Small, end-to-end synthetic example of the new NB/ZINB structured head.

Replace the synthetic tensors with **verified raw UMI counts** and external
size factors from all genes in real applications. The example uses only a few
training batches and is not a model-selection or convergence demonstration.
"""
import torch

from factmx_torch import FACTMxModel


torch.manual_seed(7)
n_patients, n_cells, n_genes, n_components = 16, 15, 6, 3
n_libraries = 2
library_ids = torch.randint(n_libraries, (n_patients, n_cells))
size_factors = 0.5 + torch.rand(n_patients, n_cells)
component_rate = torch.tensor([
    [6., 3., 1., 1., 2., 0.4],
    [0.3, 5., 7., 1., 2., 3.],
    [1., 1., 2., 7., 6., 1.],
])
cell_components = torch.randint(n_components, (n_patients, n_cells))
counts = torch.poisson(size_factors[..., None] * component_rate[cell_components])
view = (counts, size_factors, library_ids)

model = FACTMxModel(
    dim_latent=3,
    heads_config=[{
        "head_type": "ZINBMixture",
        "head_name": "cells",
        "dim": n_components,
        "dim_genes": n_genes,
        "n_libraries": n_libraries,
        "inflation": "gene_component",
        "initial_inflation": "none",
        "assignment_estimator": "mean",
        "objective": "marginal",
    }],
    encoder_config={"encoder_type": "Linear", "scale_parameterization": "softplus"},
)
head = model.heads[0]
print("Observed-data initialization:", head.initialize_from_counts(view, seed=7))
print("NB train loss:", model.fit([view], epochs=2, batch_size=8)[-1])
head.set_inflation_mode("gene")
print("Gene-ZINB train loss:", model.fit([view], epochs=2, batch_size=8, lr=3e-4)[-1])
print("Patient latents:", model.get_latent_representation([view]).shape)
print("Raw count draw:", head.sample_counts(
    model.get_latent_representation([view]), size_factors, library_ids
).shape)
