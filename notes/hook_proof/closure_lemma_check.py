"""Closure lemma, explicit.  h(pi) = F^{(b)}_v(rho pi), rho good rotation on block B (v at leading end) or id.
For c with t = rho(c) != v: piece H_c(pi') = s * F^{(b-1)}_{v'}(rho' pi') with
   v' = std_{[m]\\v}(t),  rho' = std_{[m]\\v} o psi o rho o std_{[m]\\c}^{-1},  psi: [m]\\{t} -> [m]\\{v}, psi(v)=t.
Check: (1) the identity holds with a constant sign s for all pi'; (2) rho' is identity or a good rotation (v' at leading end);
(3) s = +1;  (4) t = v gives the zero piece.  Report block of rho' in terms of (B, c)."""
import sys
from itertools import permutations
from collections import Counter
m=int(sys.argv[1])
def sgn(p): return (-1)**sum(1 for a in range(len(p)) for b in range(a+1,len(p)) if p[a]>p[b])
def cyclen(p,i):
    L=1; x=p[i]
    while x!=i: x=p[x]; L+=1
    return L
def F(b,v,p): return (-1)**(b-1)*sgn(p)*(1 if cyclen(p,v)>=b else 0)
def rot(k,lo,hi,d):
    z=list(range(k))
    if lo==hi: return tuple(z)
    if d=='down': z[lo]=hi; z[lo+1:hi+1]=range(lo,hi)
    else: z[hi]=lo; z[lo:hi]=range(lo+1,hi+1)
    return tuple(z)
def classify(z):
    k=len(z); mv=[x for x in range(k) if z[x]!=x]
    if not mv: return ('id',None,None)
    lo,hi=min(mv),max(mv)
    for d in ('down','up'):
        if rot(k,lo,hi,d)==z: return (d,lo,hi)
    return ('other',lo,hi)
st=Counter(); bad=[]
Pk=list(permutations(range(m-1)))
for b in range(2,m+1):
    for v in range(m):
        configs=[('id',v,v)]+[('down',lo,v) for lo in range(v)]+[('up',v,hi) for hi in range(v+1,m)]
        for d,lo,hi in configs:
            rho=rot(m,lo,hi,d)
            for c in range(m):
                t=rho[c]
                if t==v:
                    # zero piece?
                    z=all(F(b,v,tuple(rho[q] for q in (lambda pp: pp)(tuple(pp)))) ==0 for pp in [])
                    st['t=v (zero piece)']+=1; continue
                stdv=lambda x: x-(x>v)          # std on [m]\{v}
                stdc_inv=lambda y: y+(y>=c)     # [m-1] -> [m]\{c}
                psi=lambda x: t if x==v else x
                rho_p=tuple(stdv(psi(rho[stdc_inv(y)])) for y in range(m-1))
                vp=stdv(t)
                # verify identity
                s=None; ok=True
                for pp in Pk:
                    # lift pi' to pi with pi(v)=c
                    full=[None]*m; full[v]=c
                    for idx,pos in enumerate([x for x in range(m) if x!=v]): full[pos]=stdc_inv(pp[idx])
                    lhs=F(b,v,tuple(rho[full[q]] for q in range(m)))
                    rhs=F(b-1,vp,tuple(rho_p[pp[q]] for q in range(m-1)))
                    if rhs==0 and lhs==0: continue
                    if rhs==0 or lhs==0: ok=False; break
                    r=lhs*rhs
                    if s is None: s=r
                    elif s!=r: ok=False; break
                cl=classify(rho_p)
                good = cl[0]=='id' or b-1==1 or (cl[0] in ('down','up') and cl[2]-cl[1]==1 and vp in (cl[1],cl[2])) or (cl[0]=='down' and cl[2]==vp) or (cl[0]=='up' and cl[1]==vp)
                st[('identity holds' if ok else 'IDENTITY FAILS', 'rho\' good' if good else 'rho\' NOT good', f'sign {s}')]+=1
                if not (ok and good and s==1) and len(bad)<5: bad.append((b,v,(d,lo,hi),c,cl,vp,s,ok))
print(f"m={m}:",dict(st)); print('  problems:',bad)
