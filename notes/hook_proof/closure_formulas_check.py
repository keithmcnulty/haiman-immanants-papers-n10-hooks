"""Check the hand-derived formulas (down case and up case): block of rho'_c, v' position, length |c - lo| (down) /
|c - hi| (up), and H_c(e) = (-1)^{b+L+1} [L >= b-1]."""
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
st=Counter()
for v in range(m):
    for d,lo,hi in [('down',lo,v) for lo in range(v+1)]+[('up',v,hi) for hi in range(v+1,m)]:
        rho=rot(m,lo,hi,d); anchor=lo if d=='down' else hi     # the end opposite to v
        for c in range(m):
            t=rho[c]
            if t==v: st[('zero piece iff c=anchor', c==anchor)]+=1; continue
            stdv=lambda x: x-(x>v); stdc_inv=lambda y: y+(y>=c); psi=lambda x: t if x==v else x
            rp=tuple(stdv(psi(rho[stdc_inv(y)])) for y in range(m-1)); vp=stdv(t)
            mv=[y for y in range(m-1) if rp[y]!=y]
            L=(max(mv)-min(mv)+1) if mv else 1
            st[('L = |c-anchor|', L==abs(c-anchor))]+=1
            for b in range(2,m+2):
                st[('H_c(e) formula', F(b-1,vp,rp)==((-1)**(b+L+1) if L>=b-1 else 0))]+=1
print(m,dict(st))
