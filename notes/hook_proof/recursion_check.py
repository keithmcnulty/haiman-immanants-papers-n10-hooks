"""Recursive structure of Theorem 4.4 of the hooks paper (rotated per-vertex functions).  h(pi) = F_v(zeta pi) on S_m (right-edge config: block [lo, v], zeta(lo) = v).
Split by c = pi(v): H_c on S_{m-1} (coset {pi(v)=c}, delete position v and value c).  BB: h(C'_u) = sum_c sum_y a_y H_c(C'_{phi y}).
Tests: (1) H_c(C'_{u'}) >= 0 for u' != e, all c;  (2) bottom sum B(u) = [Imm_u] sum_c H_c(e) x_{vc} det x[^v,^c] >= 0 for u != e;
(3) R = h - B >= 0 everywhere.  Also: what are the H_c(e) values (signs) as functions of c?"""
import sys, time; sys.path.insert(0,'.')
import numpy as np
from math import factorial
from collections import Counter
from monimm.kl_fast import FastKL, evaluate_on_kl_basis
m=int(sys.argv[1]); t0=time.time()
def setup(k):
    fast=FastKL(k); P=np.array(fast.tables.perms,dtype=np.int64); N=len(P)
    inv=np.zeros(N,dtype=np.int64)
    for i in range(k): inv+=(P[:,i+1:]<P[:,i:i+1]).sum(axis=1)
    sgn=np.where(inv%2==0,1,-1)
    cyc=np.zeros((N,k),dtype=np.int64)
    for i in range(k):
        x=P[:,i].copy(); L=np.ones(N,dtype=np.int64)
        while True:
            o=x!=i
            if not o.any(): break
            x=np.where(o,P[np.arange(N),x],x); L+=o
        cyc[:,i]=L
    return fast,P,sgn,cyc
def ranks(Q,k):
    r=np.zeros(Q.shape[0],dtype=np.int64)
    for j in range(k): r+=(Q[:,j+1:]<Q[:,j:j+1]).sum(axis=1)*factorial(k-1-j)
    return r
fm,Pm,sgnm,cycm=setup(m); fk,Pk,sgnk,cyck=setup(m-1)
st=Counter(); hvals=Counter()
for b in range(2,min(m,5)+1):
    for v in range(1,m):
        for L in range(2,v+2):
            lo=v-L+1
            z=np.arange(m); z[lo]=v; z[lo+1:v+1]=np.arange(lo,v)
            hfun=((-1)**(b-1)*sgnm*(cycm[:,v]>=b))[ranks(z[Pm],m)]        # h on S_m
            Hrows=[]; He=[]
            for c in range(m):
                # elements of S_{m-1} -> lift: insert value c at position v
                Q=np.where(Pk>=c,Pk+1,Pk); Q=np.insert(Q,v,c,axis=1)
                Hc=hfun[ranks(Q,m)]
                Hrows.append(Hc); He.append(int(Hc[0]))            # rank 0 = identity of S_{m-1}
            HE=evaluate_on_kl_basis(fk,np.array(Hrows))
            ok1=all((HE[c][1:]>=0).all() for c in range(m))
            beta=np.array([He[p[v]]*(-1)**(v+p[v]) for p in Pm])*sgnm
            B=evaluate_on_kl_basis(fm,beta[None,:])[0]
            Hh=evaluate_on_kl_basis(fm,hfun[None,:])[0]
            R=Hh-B
            key=(b,'L<=b' if L<=b else 'L>b','H_c>=0 off e' if ok1 else 'H_c NEG off e',
                 'B>=0 off e' if (B[1:]>=0).all() else 'B NEG off e','R>=0' if (R>=0).all() else 'R NEG')
            st[key]+=1
            hvals[(b,L,tuple(He))]+=1
print(f"m={m} ({time.time()-t0:.0f}s)")
for k in sorted(st,key=str): print('  ',k,st[k])
print("  sample H_c(e) vectors (b,L,(H_0(e),...,H_{m-1}(e))):")
for k in list(sorted(hvals))[:12]: print('    ',k)
