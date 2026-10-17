"""monimm -- Kazhdan-Lusztig immanants, monomial immanants and Haiman's q-conjecture for n <= 10.

Modules
-------
combinatorics   : partitions, cycle types, permutation characters psi^mu and the
                  exact dual basis g_lambda defining the monomial immanants.
kazhdan_lusztig : reference implementation of KL polynomials and the coefficients c_{lambda,w} (small n).
kl_fast         : compiled full-interval KL engine (n <= 8).
kl_big          : parallel full-interval engine (n <= 9).
kl_extremal     : extremal-pair engine (n <= 10), the main computation.
kl_graded       : Haiman's graded coefficients c_{lambda,w}(q) via Geck-Pfeiffer class polynomials.
chromatic       : chromatic quasisymmetric functions at codominant w (an independent check of kl_graded).
ordering        : Kostka numbers, characters and the order of irreducible character immanants on TNN.
order_structure : structure of that order (dominance, lattices, chains, Hasse diagrams).
tnn_cone        : TNN cone versus KL cone on the full span of permutation monomials (n <= 5).
tnn, evaluate   : totally positive test matrices and exact evaluation of immanants.
verify          : one command for every check of paper/haiman_n10.tex.
"""

from .combinatorics import monomial_class_functions, partitions
from .evaluate import ImmanantEvaluator
from .tnn import build_tnn_batch, build_tnn_exact, j_matrix

__all__ = [
    "ImmanantEvaluator",
    "build_tnn_batch",
    "build_tnn_exact",
    "j_matrix",
    "monomial_class_functions",
    "partitions",
]
