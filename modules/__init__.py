"""Top-level API for the `modules` package.

"""

#from .chemberta_embds import chembert_embedding
#from .chemeleon_embds import chemeleon_embedding
#from .molclr_embds import mol_clr_embed

from .basicmlp import VanillaMLP, fit_mlp_with_fixed_split
from .mlp_tuning import tune_mlp_with_fixed_split
from .linreg import fit_linear_model_with_fixed_split
from .random_forest import fit_random_forest_with_fixed_split
from .load_results import load_results

# Splitting / similarity utilities
from .datasplits import (
    SimilaritySummary,
    build_similarity_comparison_frame,
    descriptors_from_smiles,
    evaluate_split_similarity,
    group_by_scaffold,
    random_split_indices,
    scaffold_smiles,
    split_by_extremes,
    split_by_scaffold_group,
    top_n_similarities_per_train,
)

#from .maceoff_embds import compute_mace_off_embeddings, generate_optimized_3d_conformers

__all__ = [
    # Embeddings
    #"mol_clr_embed",
    #"chembert_embedding",
    #"chemeleon_embedding",
    #"compute_mace_off_embeddings",
    #"generate_optimized_3d_conformers",

    # Models / training helpers
    "VanillaMLP",
    "fit_mlp_with_fixed_split",
    "tune_mlp_with_fixed_split",
    "fit_linear_model_with_fixed_split",
    "fit_random_forest_with_fixed_split",

    # Splitting / similarity utilities
    "SimilaritySummary",
    "descriptors_from_smiles",
    "split_by_extremes",
    "random_split_indices",
    "scaffold_smiles",
    "group_by_scaffold",
    "split_by_scaffold_group",
    "evaluate_split_similarity",
    "top_n_similarities_per_train",
    "build_similarity_comparison_frame",
]
