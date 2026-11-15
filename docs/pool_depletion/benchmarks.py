import mpmath as mp, numpy as np, pool_depletion as pdp
mp.mp.dps=30
T,S,d,a=mp.mpf(500),mp.mpf('0.1'),mp.mpf(300),mp.mpf(20)
def parts(p,c,eps):
    k=mp.sqrt((S*p+eps)/T); kap=mp.sqrt((S*p+c)/T)
    D=k*mp.besselk(1,k*a)*mp.besseli(0,kap*a)+kap*mp.besseli(1,kap*a)*mp.besselk(0,k*a)
    return k,kap,D
def a1(p,c,eps=0):
    k,kap,D=parts(p,c,eps); return c*mp.besselk(0,k*d)*mp.besseli(1,kap*a)/(T*kap*D)
def be(p,c,eps=0):
    k,kap,D=parts(p,c,eps)
    return 2*mp.pi*a*c*c*k*mp.besselk(1,k*a)*mp.besseli(1,kap*a)/((S*p+c)*kap*D)+mp.pi*a*a*c*S*p/(S*p+c)
ts=[1,10,30,100,365,1000,3650]
inv=lambda f,t: mp.invertlaplace(f,t,method='talbot')
def line(name,f): return '| '+name+' | '+' | '.join(mp.nstr(inv(f,t),5) for t in ts)+' |'
out=[]
for c in ['0.01','0.1','1','10']:
    cc=mp.mpf(c); out.append(line(f'F, c = {c}',lambda p: a1(p,cc)/p))
out.append(line('F, no bed resistance',lambda p: mp.besselk(0,mp.sqrt(S*p/T)*d)/mp.besselk(0,mp.sqrt(S*p/T)*a)/p))
out.append(line('F, c = 0.1, ε = 1e-4',lambda p: a1(p,mp.mpf('0.1'),mp.mpf('1e-4'))/p))
out.append(line('G (m²/d), c = 0.1',lambda p: be(p,mp.mpf('0.1'))/p))
Ap=mp.pi*a*a; c=mp.mpf('0.1')
out.append(line('Closed pool qₚ/Q',lambda p: Ap*a1(p,c)/(Ap*p+be(p,c))))
out.append(line('Closed pool sₚ (m), Q = 1000',lambda p: 1000*a1(p,c)/p/(Ap*p+be(p,c))))
print('\n'.join(out))
# B5 fine step
R,D,y0,cc,E=40.,3.,1.5,0.005,0.007; days=300
for dt in [0.1]:
    m=int(days/dt)
    for Q in [0.,500.]:
        y,q=pdp.pool_balance(np.full(m,Q),np.zeros(m),np.full(m,E),500.,0.1,300.,R,D,y0,cc,dt=dt)
        idx=(np.array([10,30,60,120,180,240,300])/dt).astype(int)-1
        print(dt,Q,' | '.join(f'{v:.3f}' for v in y[idx]))
