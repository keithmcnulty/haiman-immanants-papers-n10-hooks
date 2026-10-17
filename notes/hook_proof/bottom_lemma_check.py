"""Bottom lemma pattern. For h = F^{(b)}_v o rho (good), H_c(e) = F^{(b-1)}_{v'}(rho'_c) (closure formula; 0 if rho(c)=v).
Check: (1) Z = {c : H_c(e) = 0} is an interval; (2) H_c(e)(-1)^{v+c} constant sign on the complement;
(3) predicted B via Temperley-Lieb telescoping: surviving boundary caps get coefficient
     kappa * (-1)^{v+c_b} ... we just record (Z, sign) and, for m <= 7, compare B >= 0 off e numerically elsewhere."""
import sys
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
st=Counter(); ex=[]
for b in range(2,m+1):
    for v in range(m):
        configs=[('id',v,v)]+[('down',lo,v) for lo in range(v)]+[('up',v,hi) for hi in range(v+1,m)]
        for d,lo,hi in configs:
            rho=rot(m,lo,hi,d); vals=[]
            for c in range(m):
                t=rho[c]
                if t==v: vals.append(0); continue
                stdv=lambda x: x-(x>v); stdc_inv=lambda y: y+(y>=c); psi=lambda x: t if x==v else x
                rho_p=tuple(stdv(psi(rho[stdc_inv(y)])) for y in range(m-1))
                vals.append(F(b-1,stdv(t),rho_p)*(-1)**(v+c))
            Z=[c for c in range(m) if vals[c]==0]
            interval = (not Z) or Z==list(range(min(Z),max(Z)+1))
            signs=set(x for x in vals if x)
            kappa=signs.pop() if signs else 0
            if not Z: st['Z empty: B = kappa*det, zero off e']+=1; continue
            Lb=min(Z)-1; R=max(Z)+1                     # left tail c<=Lb, right tail c>=R
            conds=[]
            if R<=m-1: conds.append(kappa*(-1)**(v+R)==1)
            if Lb>=0:  conds.append(kappa*(-1)**(v+Lb)==1)
            st[('sign condition holds' if all(conds) else 'SIGN CONDITION FAILS', f'{len(conds)} tails')]+=1
            if not all(conds) and len(ex)<5: ex.append((b,v,(d,lo,hi),vals,Z,kappa))
            if (not interval or len(signs)>1) and len(ex)<5: ex.append((b,v,(d,lo,hi),vals))
print(f"m={m}:",dict(st)); print('  problems:',ex)
