"""theta_l = sum_{|T| = n-l} g_T, g_T(sigma) = (-1)^{|T|} sgn(sigma) [no cycle of sigma inside T].
(0) check the identity against sum_{l(mu)=l} phi^mu (KL coefficient tables); (1) is each g_T KL-positive?"""
import sys, time; sys.path.insert(0,'.')
import numpy as np
from itertools import combinations
from collections import Counter
from monimm.kl_fast import FastKL, evaluate_on_kl_basis
n=int(sys.argv[1]); t0=time.time()
fast=FastKL(n); fast.c_coefficients(); C=fast.C_scaled; T_=fast.tables
P=np.array(T_.perms,dtype=np.int64); N=len(P)
inv=np.zeros(N,dtype=np.int64)
for i in range(n): inv+=(P[:,i+1:]<P[:,i:i+1]).sum(axis=1)
sgn=np.where(inv%2==0,1,-1)
# cycle id of each vertex
cid=np.zeros((N,n),dtype=np.int64)
for r,p in enumerate(T_.perms):
    seen=[-1]*n; k=0
    for i in range(n):
        if seen[i]<0:
            j=i
            while seen[j]<0: seen[j]=k; j=p[j]
            k+=1
    cid[r]=seen
rows=[]; meta=[]
for k in range(0,n):
    for Tset in combinations(range(n),k):
        S=[i for i in range(n) if i not in Tset]
        # every cycle meets S  <=>  set of cycle ids of S = all cycle ids
        ncyc=cid.max(axis=1)+1
        hit=np.array([len(set(cid[r,S].tolist())) for r in range(N)])
        g=((-1)**k)*sgn*(hit==ncyc)
        rows.append(g); meta.append(Tset)
E=evaluate_on_kl_basis(fast,np.array(rows))
parts=T_.parts
st=Counter()
for l in range(1,n+1):
    theta=sum(E[q] for q,Tset in enumerate(meta) if len(Tset)==n-l)
    ref=sum(C[li] for li,lam in enumerate(parts) if len(lam)==l)
    st[('identity theta_l = sum phi^mu', bool(np.array_equal(theta,ref)))]+=1
for q,Tset in enumerate(meta):
    st[(f'|T|={len(Tset)}','g_T KL-pos' if (E[q]>=0).all() else 'g_T MIXED')]+=1
print(f"n={n} ({time.time()-t0:.0f}s):")
for kk in sorted(st,key=str): print('  ',kk,st[kk])
