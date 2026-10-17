"""Imm_u(x[I,T]) * Imm_v(x[I^c,T^c]) is KL-positive for all I, T (|I|=|T|=j), u in S_j, v in S_{n-j}?"""
import sys; import os
_root = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_root, 'monimm')):   # the repository root, wherever this file lives
    _root = os.path.dirname(_root)
sys.path.insert(0, _root)
import numpy as np
from itertools import permutations, combinations
from monimm.kl_fast import FastKL
def klmat(m):
    P=list(permutations(range(m)))
    if m<=1: return P,np.ones((1,1),dtype=np.int64)
    f=FastKL(m); M=np.array([[sum(f.p(x,u)) if f.p(x,u) else 0 for u in P] for x in P],dtype=np.int64)
    return P,M          # M[x,u] = P_{x,u}(1); f(C'_u) = sum_x M[x,u] f(x)
def std(seq):
    s=sorted(seq); return tuple(s.index(x) for x in seq)
for n in range(2,7):
    Pn,Mn=klmat(n); idx={p:i for i,p in enumerate(Pn)}
    worst=0; count=0
    for j in range(1,n):
        Pa,Ma=klmat(j); Pb,Mb=klmat(n-j)
        Fa=np.round(np.linalg.inv(Ma.astype(float))).astype(np.int64)   # row u: dual function f_u with f_u(C'_x)=delta
        Fb=np.round(np.linalg.inv(Mb.astype(float))).astype(np.int64)
        ia={p:i for i,p in enumerate(Pa)}; ib={p:i for i,p in enumerate(Pb)}
        for I in combinations(range(n),j):
            Ic=[x for x in range(n) if x not in I]
            for T in combinations(range(n),j):
                rows=[i for i,p in enumerate(Pn) if sorted(p[x] for x in I)==list(T)]
                A=np.array([ia[std([Pn[r][x] for x in I])] for r in rows]); B=np.array([ib[std([Pn[r][x] for x in Ic])] for r in rows])
                for u in range(len(Pa)):
                    for v in range(len(Pb)):
                        f=np.zeros(len(Pn),dtype=np.int64); f[rows]=Fa[u,A]*Fb[v,B]
                        c=f@Mn; worst=min(worst,c.min()); count+=1
    print(f"n={n}: {count} products of KL immanants of complementary submatrices; min KL coefficient {worst}",flush=True)
