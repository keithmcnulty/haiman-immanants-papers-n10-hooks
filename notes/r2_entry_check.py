#!/usr/bin/env python3
"""Check of the product theorem (Theorem 3.1 of the hooks paper) when one factor is a single entry x_ij.

Usage: .venv/bin/python notes/r2_entry_check.py N        (n = 7: ~8 min)

The coefficient of Imm_w in x_ij * Imm_u(x[^i,^j]) is
    c = sum_{sigma(i)=j} P_{sigma,w}(1) f_u(sigma'),   f_u(v) = (-1)^{l(v)-l(u)} P_{w0 v, w0 u}(1) [v >= u],
with sigma' = sigma with row i and column j deleted (standardised).  We check c >= 0 for all w, i, j, u.
Control: replacing P_{sigma,w}(1) by [sigma <= w] gives negative coefficients already at n = 5 (242 of 72000).
"""
import sys; sys.path.insert(0,'.')
import numpy as np
from itertools import permutations, combinations
from monimm.kl_fast import FastKL
n=int(sys.argv[1])
F=FastKL(n); Fm=FastKL(n-1)
P=list(permutations(range(n))); Pm=list(permutations(range(n-1))); rm={p:r for r,p in enumerate(Pm)}
def L(p): return sum(1 for a in range(len(p)) for b in range(a+1,len(p)) if p[a]>p[b])
w0m=tuple(range(n-2,-1,-1))
def comp(a,b): return tuple(a[b[i]] for i in range(len(b)))
# f matrix for S_{n-1}: Fu[v,u]
N1=len(Pm); Fu=np.zeros((N1,N1),dtype=np.int64)
for vi,v in enumerate(Pm):
    for ui,u in enumerate(Pm):
        p=Fm.p(comp(w0m,v),comp(w0m,u))
        if p: Fu[vi,ui]=(-1)**(L(v)-L(u))*sum(p)
# P(1) matrix for S_n: A[w, sigma]
A=np.zeros((len(P),len(P)),dtype=np.int64)
for wi,w in enumerate(P):
    for si,s in enumerate(P):
        p=F.p(s,w)
        if p: A[wi,si]=sum(p)
neg=0; tot=0; ex=None
for i in range(n):
    for j in range(n):
        cols=[si for si,s in enumerate(P) if s[i]==j]
        M=np.zeros((len(cols),N1),dtype=np.int64)
        for k,si in enumerate(cols):
            s=P[si]; rest=[s[a] for a in range(n) if a!=i]
            sp=tuple(x-1 if x>j else x for x in rest)
            M[k,rm[sp]]=1
        Cc=A[:,cols]@M@Fu       # [w, u]
        tot+=Cc.size; nn=(Cc<0).sum(); neg+=nn
        if nn and ex is None:
            wi,ui=np.argwhere(Cc<0)[0]; ex=(''.join(str(a+1) for a in P[wi]),i+1,j+1,''.join(str(a+1) for a in Pm[ui]),int(Cc[wi,ui]))
print(f"n={n}: coefficients checked {tot}, negative {neg}; example {ex}",flush=True)
