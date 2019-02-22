import alchemlyb
import mdsynthesis as mds
from dask.delayed import delayed
import os
import os.path
import pandas as pd
from alchemlyb.preprocessing import slicing
from alchemlyb.parsing.gmx import extract_dHdl
import sympy
from alchemlyb.estimators import TI

try:
	BASEPATH = os.environ['FEP_BASEPATH']
except KeyError:
	BASEPATH = "/"

T = 310
k_b = 8.3144621E-3
sympy.init_printing(use_unicode=True)
r, theta, k_r, k_theta, r_0, theta_0, beta, V = sympy.symbols('r theta k_r k_theta r_0 theta_0 beta V')

subs = {k_theta: 23.0,
        k_r: 16000.0,
        r_0: 0.275,
        theta_0: 0.0,
        beta : 1/(8.3144621E-3 * 310),
        V: 1.6605778811026237}  # standard volume in nm^3
f_theta = sympy.exp(-(beta * k_theta/2) * (theta - theta_0)**2 ) * sympy.sin(theta)
Z_rtheta = sympy.Integral(f_theta, (theta, 0, sympy.pi))

Z_rtheta.subs(subs).evalf()
f_r = sympy.exp(-(beta * k_r/2) * (r - r_0)**2 ) * r**2
Z_rr = sympy.Integral(f_r, (r, 0, sympy.oo))

Z_rr.subs(subs).evalf()
Z_r = 2 * sympy.pi * Z_rtheta * Z_rr
Z_r.subs(subs).evalf()

DG_r = - 1/beta * sympy.ln(V/Z_r)
DG_r_unbind = DG_r.subs(subs).evalf()
DG_r_unbind = -17.8435335231149 # why keep computing the same thing

def get_dHdl_XVG(sim, lower=None, upper=None, step=None):
    dHdls = []
    for xvg in sim['WORK/dhdl/'].glob('*.xvg'):
        dHdls.append(extract_dHdl(xvg.abspath, T=T))
    return slicing(pd.concat(dHdls).sort_index(0), 
                   lower=lower, 
                   upper=upper, 
                   step=step)

def get_dHdl(sim, lower=None, upper=None, step=None):
    try:
        dHdl = sim.data.retrieve('dHdl')
        if dHdl is None:
            dHdl = get_dHdl_XVG(sim, lower=lower, upper=upper, step=step)
        dHdl = slicing(dHdl.sort_index(0), 
                       lower=lower, 
                       upper=upper, 
                       step=step)
    except:
        # THIS WILL NOT STORE THE VALUE FOR LATER USE SO YOU SHOULD REALLY
        # CONTINUOUSLY UPDATE THE dHdl DATA IN THE SIMS
        dHdl = get_dHdl_XVG(sim, lower=lower, upper=upper, step=step) 
    return dHdl


def get_TI(dHdl):
    ti = TI().fit(dHdl)
    df = pd.DataFrame({'DG': k_b * T * ti.delta_f_.values[0,-1:], 
                       'std': k_b * T * ti.d_delta_f_.values[0,-1:]},
                      columns=['DG', 'std'])
    return df

if __name__ == "__main__":
    from dask.distributed import Client, LocalCluster
    cl = Client()
    
    ionsegs = {'repulsion_to_ghost': mds.discover(BASEPATH + '/Projects/Transporters/SYSTEMS/Na/repulsion_to_ghost/production1/'),
    	   'ghost_to_ion': mds.discover(BASEPATH + '/Projects/Transporters/SYSTEMS/Na/ghost_to_ion/production1/')}
    
    dHdls = {}
    for seg in ionsegs:
        dHdls[seg] = [delayed(get_dHdl, pure=True)(sim, lower=5000, step=200)
    				for sim in ionsegs[seg]]
    
    L_ionDG = {}
    for seg in ionsegs:
        iondg_d = delayed(get_TI)(delayed(pd.concat)(dHdls[seg]))
        L_ionDG[seg] = cl.compute(iondg_d)
    ionDG = cl.gather(L_ionDG)
    
    dfs = []
    for seg in ionDG:
        df = ionDG[seg]
        df['segment'] = seg
        dfs.append(df)
    
    ionDG = pd.concat(dfs)
    ionDG = ionDG.set_index('segment')
    ionDG.loc['ghost_to_ion', 'DG'] = -1 * ionDG.loc['ghost_to_ion', 'DG']
    topdir = mds.Tree(BASEPATH + '/Projects/Transporters/NapA/SYSTEMS/ionbinding/na/')
    
    segs_s2if = {'unrestrained_to_restrained': mds.discover(topdir['if/S2/unrestrained_to_restrained/production1']),
    	     'restrained_to_repulsion': mds.discover(topdir['if/S2/restrained_to_repulsion/production1/']),
    	     'repulsion_to_ghostrepulsion': mds.discover(topdir['if/S2/repulsion_to_ghostrepulsion/production1/'])}
    """
    segs_s4if = {'unrestrained_to_restrained': mds.discover(topdir['if/S4/unrestrained_to_restrained/production1']),
    	     'restrained_to_repulsion': mds.discover(topdir['if/S4/restrained_to_repulsion/production1/']),
    	     'repulsion_to_ghostrepulsion': mds.discover(topdir['if/S4/repulsion_to_ghostrepulsion/production1/'])}
    segs_s2of = {'unrestrained_to_restrained': mds.discover(topdir['of/S2/unrestrained_to_restrained/production1']),
    	     'restrained_to_repulsion': mds.discover(topdir['of/S2/restrained_to_repulsion/production1/']),
    	     'repulsion_to_ghostrepulsion': mds.discover(topdir['of/S2/repulsion_to_ghostrepulsion/production1/'])}
    
    segs_s4of = {'unrestrained_to_restrained': mds.discover(topdir['of/S4/unrestrained_to_restrained/production1']),
    	     'restrained_to_repulsion': mds.discover(topdir['of/S4/restrained_to_repulsion/production1/']),
    	     'repulsion_to_ghostrepulsion': mds.discover(topdir['of/S4/repulsion_to_ghostrepulsion/production1/'])}
    """
    segs = {'s2if': segs_s2if}
    """,
    	's4if': segs_s4if,
    	's2of': segs_s2of,
    	's4of': segs_s4of}
    """
    
    L_DG = {} 
    
    for state in segs:
        ddgs = {}
        for seg in segs[state]:
           dHdls = delayed(pd.concat)([delayed(get_dHdl)(sim, lower=5000, step=200) for sim in segs[state][seg]])
           ddgs[seg] = delayed(get_TI)(dHdls)
        L_DG[state] = {seg: cl.compute(ddgs[seg]) for seg in segs[state]}
    
    DG = cl.gather(L_DG)
    state = 's2if'
    
    dfs = []
    for seg in DG[state]:
        df = DG[state][seg]
        df['segment'] = seg
        dfs.append(df)
    
    dfDG = pd.concat(dfs)
    
    dfDG = dfDG.set_index('segment')
    order = ['unrestrained_to_restrained', 
    	 'restrained_to_repulsion',
    	 'repulsion_to_ghostrepulsion']
    
    dfDG = dfDG.loc[order]
    
    pd.concat([dfDG, ionDG])['DG'].sum()
    
    std = pd.np.sqrt((pd.concat([dfDG, ionDG])['std'] **2).sum())
    
    var = std**2
    DG_unbind = (pd.concat([dfDG, ionDG])['DG'].sum() + DG_r_unbind)
""" 
    state = 's4if'
    
    dfs = []
    for seg in DG[state]:
        df = DG[state][seg]
        df['segment'] = seg
        dfs.append(df)
    
    
    dfDG = pd.concat(dfs)
    
    dfDG = dfDG.set_index('segment')
    order = ['unrestrained_to_restrained', 
    	 'restrained_to_repulsion',
    	 'repulsion_to_ghostrepulsion']
    
    dfDG = dfDG.loc[order]
    pd.concat([dfDG, ionDG])['DG'].sum()
    pd.np.sqrt((pd.concat([dfDG, ionDG])['std'] **2).sum())
    DG_unbind = (pd.concat([dfDG, ionDG])['DG'].sum() + DG_r_unbind)
    state = 's2of'
    dfs = []
    for seg in DG[state]:
        df = DG[state][seg]
        df['segment'] = seg
        dfs.append(df)
    
    dfDG = pd.concat(dfs)
    dfDG = dfDG.set_index('segment')
    order = ['unrestrained_to_restrained', 
    	 'restrained_to_repulsion',
    	 'repulsion_to_ghostrepulsion']
    dfDG = dfDG.loc[order]
    pd.concat([dfDG, ionDG])['DG'].sum()
    pd.np.sqrt((pd.concat([dfDG, ionDG])['std'] **2).sum())
    
    DG_unbind = (pd.concat([dfDG, ionDG])['DG'].sum() + DG_r_unbind)
    state = 's4of'
    dfs = []
    for seg in DG[state]:
        df = DG[state][seg]
        df['segment'] = seg
        dfs.append(df)
    dfDG = pd.concat(dfs)
    dfDG = dfDG.set_index('segment')
    order = ['unrestrained_to_restrained', 
    	 'restrained_to_repulsion',
    	 'repulsion_to_ghostrepulsion']
    
    dfDG = dfDG.loc[order]
    
    pd.concat([dfDG, ionDG])['DG'].sum()
    pd.np.sqrt((pd.concat([dfDG, ionDG])['std'] **2).sum())
    
    DG_unbind = (pd.concat([dfDG, ionDG])['DG'].sum() + DG_r_unbind)
"""
