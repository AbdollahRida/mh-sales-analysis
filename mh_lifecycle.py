#!/usr/bin/env python3
"""
mh_lifecycle.py — Monster Hunter Steam lifecycle decomposition (single-file).
================================================================================
Reads SteamDB exports from ./data/. Models player-count dynamics as a structural
decomposition and produces a full diagnostic dashboard.

DATA in ./data/ (BOM-tolerant):
  steamdb_chart_<appid>.csv : DateTime, Players, Average Players  (concurrent)
  <game>_prices.csv         : DateTime, Final price, Historical Low

MODEL (per game g, months-since-launch t):
  N_g(t) = [core(t)+launch(t)] · (1+events(t)) · (1+sale(t)) · (1+spill(t)) · ε
    core   = β + (α−β)e^{−λt}                          slow decay to floor
    launch = L0 e^{−d0 t}                              fast hype transient
    events = Σ_dlc γ_d e^{−δ_d(t−τ)}  +  γ_TU Σ_tu e^{−δ_TU(t−τ)}
             (expansions fit individually; title-updates/collabs share γ_TU,δ_TU)
    sale   = φ d_t^κ σ(P*−eff_price)                   price-gated, convex
    spill  = Σ θ_h N_h(t−1)/N̄_h                       lagged cross-game

Usage:
  python mh_lifecycle.py                # full run + dashboard
  python mh_lifecycle.py --no-partial   # drop current incomplete month
  python mh_lifecycle.py --linear       # linear y-axis (default log)
"""
from __future__ import annotations
import os, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from scipy.optimize import least_squares
from scipy import stats

from events import EVENTS, SUPPORT

HERE = os.path.dirname(os.path.abspath(__file__)); DATA_DIR = os.path.join(HERE, "data")
GAMES = {
    "World": dict(appid=582010,  launch="2018-08-01", base_price=59.99, color="#e07b39"),
    "Rise":  dict(appid=1446780, launch="2022-01-12", base_price=59.99, color="#5b8fc9"),
    "Wilds": dict(appid=2246340, launch="2025-03-01", base_price=69.99, color="#6dbf67"),
}
TRAIN, TEST = ["World", "Rise"], "Wilds"
AFFORD_THRESHOLD, AFFORD_SHARPNESS, KAPPA_FIXED = 15.0, 0.4, 3.0

# Sale model options: 'sigmoid' (original), 'linear', 'loglinear', 'additive'
SALE_MODEL = "linear"  # Default to simpler linear model

# ════════════════════════ DATA ════════════════════════
def _read(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lstrip("\ufeff").lower().replace(" ", "_") for c in df.columns]
    return df

def event_offsets(game, kinds=None):
    launch = pd.Timestamp(GAMES[game]["launch"])
    out = []
    for event in EVENTS.get(game, []):
        # Handle both old format (name, date, kind) and new format (date, type, desc)
        if len(event) == 3:
            if isinstance(event[1], str) and '-' in event[1]:  # New format: (date, type, desc)
                dt, kind, desc = event[0], event[1], event[2]
                nm = f"{kind} {desc}" if desc else kind
            else:  # Old format: (name, date, kind)
                nm, dt, kind = event[0], event[1], event[2]
        else:
            continue
        if kinds and kind not in kinds: continue
        d = pd.Timestamp(dt)
        out.append((nm, (d.year-launch.year)*12 + (d.month-launch.month), kind))
    return out

def load_players(game, keep_partial=True):
    path = os.path.join(DATA_DIR, f"steamdb_chart_{GAMES[game]['appid']}.csv")
    df = _read(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["players"]  = pd.to_numeric(df["players"], errors="coerce")
    df = df.dropna(subset=["players"]).sort_values("datetime")
    m = df.set_index("datetime")["players"].resample("MS").mean().reset_index()
    m.columns = ["date", "avg_players"]
    today = pd.Timestamp.today().normalize().replace(day=1)
    if not keep_partial and m["date"].max() == today:
        m = m[m["date"] < today]
    launch = pd.Timestamp(GAMES[game]["launch"]).replace(day=1)
    return m[m["date"] >= launch].reset_index(drop=True)

def _regular(ev):
    e = ev.sort_values("date").copy(); e = e[e["final_price"] > 0].set_index("date")
    e["regular"] = e["final_price"].rolling("120D", min_periods=1).max().values
    e["discount_pct"] = (1 - e["final_price"]/e["regular"]).clip(lower=0)
    e.loc[e["discount_pct"] < 0.01, "discount_pct"] = 0.0
    return e.reset_index()[["date","final_price","regular","discount_pct"]]

def load_prices(game, months):
    path = os.path.join(DATA_DIR, f"{game.lower()}_prices.csv")
    if not os.path.exists(path):
        b = GAMES[game]["base_price"]
        return pd.DataFrame({"date":months,"discount_pct":0.0,"base_price":b,"eff_price":b})
    raw = _read(path); raw["date"] = pd.to_datetime(raw["datetime"])
    raw["final_price"] = pd.to_numeric(raw["final_price"], errors="coerce")
    rec = _regular(raw[["date","final_price"]]).set_index("date")
    rows = []
    for d in months:
        mr = rec.loc[(rec.index>=d)&(rec.index<d+pd.offsets.MonthBegin(1))]
        if len(mr):
            i = mr["discount_pct"].idxmax()
            fp,reg,dc = mr.loc[i,"final_price"],mr.loc[i,"regular"],mr.loc[i,"discount_pct"]
        else:
            s = rec.loc[:d]
            if len(s): fp,reg,dc = s["final_price"].iloc[-1],s["regular"].iloc[-1],0.0
            else: fp=reg=GAMES[game]["base_price"]; dc=0.0
        rows.append((d,float(dc),float(reg),float(fp)))
    return pd.DataFrame(rows, columns=["date","discount_pct","base_price","eff_price"])

def load_panel(game, keep_partial=True):
    launch = pd.Timestamp(GAMES[game]["launch"])
    pl = load_players(game, keep_partial)
    pr = load_prices(game, pd.DatetimeIndex(pl["date"]))
    p = pl.merge(pr, on="date", how="left")
    p["t"] = (p["date"].dt.year-launch.year)*12 + (p["date"].dt.month-launch.month)
    p.attrs["game"] = game
    p.attrs["dlc"]  = event_offsets(game, kinds={"dlc"})
    p.attrs["tu"]   = event_offsets(game, kinds={"title_update","collab","event"})
    return p[p["t"]>=0].sort_values("t").reset_index(drop=True)

def all_panels(keep_partial=True):
    return {g: load_panel(g, keep_partial) for g in GAMES}

# ════════════════════════ DIAGNOSTICS ═══════════════════════
def get_event_dates(game):
    """Get all event dates for a game."""
    dates = set()
    for event in EVENTS.get(game, []):
        # Handle both old format (name, date, kind) and new format (date, type, desc)
        if len(event) >= 2:
            # Check if second element is a date string (contains '-')
            if isinstance(event[1], str) and '-' in event[1]:
                dates.add(pd.Timestamp(event[0]))  # New format: date is first element
            else:
                dates.add(pd.Timestamp(event[1]))  # Old format: date is second element
    return dates

def identify_sale_only_periods(panel):
    """Identify periods with sales but no events (for clean sale effect estimation)."""
    game = panel.attrs["game"]
    event_dates = get_event_dates(game)
    sale_only_mask = (panel["discount_pct"] > 0.1) & (~panel["date"].isin(event_dates))
    return panel[sale_only_mask]

def plot_wilds_tu_multiplier(panels, fits):
    """Generate plot showing Wilds TU multiplier effect."""
    plt.style.use("dark_background")
    
    if TEST not in fits:
        print("Wilds not fitted, skipping TU multiplier plot")
        return
    
    wilds_fit = fits[TEST]
    wilds_model = wilds_fit["model"]
    wilds_panel = panels[TEST]
    c = wilds_model.components()
    
    # Debug: print first few values
    print(f"\n[DEBUG] Wilds TU multiplier data:")
    print(f"  t: {c['t'][:5]}")
    print(f"  core: {c['core'][:5]}")
    print(f"  launch: {c['launch'][:5]}")
    print(f"  base: {c['base'][:5]}")
    print(f"  L1: {c['L1'][:5]}")
    print(f"  TU γ={c['tu_g']:.2f}, HL={c['hl_tu']:.1f}mo, d0={c['d0']:.4f}")
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), facecolor=DARK)
    
    # Plot 1: TU Multiplier (L1/base)
    ax1 = axes[0]
    _style(ax1)
    
    base = c["base"]
    # Use core as the baseline for multiplier to avoid launch decay issues
    # TU multiplier = (base + events) / base = 1 + (events / base)
    # But L1 = base * (1 + events), so L1/base = 1 + events
    # This is correct, but if base is near zero, we need to handle it
    # Use max(base, core) to ensure we don't divide by near-zero values
    safe_base = np.maximum(base, c["core"])
    tu_multiplier = c["L1"] / safe_base
    
    ax1.plot(c["t"], tu_multiplier, color=GAMES[TEST]["color"], lw=2, label="TU multiplier (L1/base)")
    ax1.axhline(1.0, color="#888", lw=1, ls="--", label="Baseline (no effect)")
    
    # Mark TU events
    for nm, offset, kind in EVENTS.get(TEST, []):
        if kind in {"title_update", "collab", "event"}:
            ax1.axvline(offset, color="#9b9bd0", lw=1, ls=":", alpha=0.7)
            ax1.text(offset, ax1.get_ylim()[1] * 0.95, f" {nm}", 
                    color="#9b9bd0", fontsize=8, rotation=90, va="top", ha="left")
    
    ax1.set_xlabel("Months since launch")
    ax1.set_ylabel("TU Multiplier (× baseline)")
    ax1.set_title(f"{TEST} — TU Multiplier (γ={c['tu_g']:.2f}, HL={c['hl_tu']:.1f}mo)", 
                  fontweight="bold", loc="left")
    ax1.legend(fontsize=8, facecolor="#1c1c24", edgecolor="#444", labelcolor=TXT)
    ax1.set_ylim(0.8, 2.0)  # Force reasonable y-axis limits
    
    # Plot 2: Components breakdown
    ax2 = axes[1]
    _style(ax2)
    # Stack: core, then launch, then events (TU), then sale, then spillover
    ax2.fill_between(c["t"], 0, c["core"], color="#e07b39", alpha=0.5, label="core")
    ax2.fill_between(c["t"], c["core"], c["base"], color="#ffd27f", alpha=0.4, label="launch")
    ax2.fill_between(c["t"], c["base"], c["L1"], color="#9b7fe0", alpha=0.55, label="events (TU)")
    if 'L2' in c:
        ax2.fill_between(c["t"], c["L1"], c["L2"], color="#5fd0a0", alpha=0.4, label="sale")
    if 'L3' in c:
        ax2.fill_between(c["t"], c.get("L2", c["L1"]), c["L3"], color="#d0607f", alpha=0.3, label="spillover")
    ax2.plot(c["t"], wilds_panel["avg_players"].values, color="white", lw=1.5, label="actual")
    ax2.plot(c["t"], wilds_model.pred_, color="white", lw=1.0, ls="--", alpha=0.8, label="model")
    
    for nm, offset, kind in EVENTS.get(TEST, []):
        if kind in {"title_update", "collab", "event"}:
            ax2.axvline(offset, color="#9b9bd0", lw=1, ls=":", alpha=0.7)
    
    ax2.set_xlabel("Months since launch")
    ax2.set_ylabel("Avg Concurrent Players")
    ax2.set_title(f"{TEST} — Components Breakdown", fontweight="bold", loc="left")
    ax2.legend(fontsize=8, facecolor="#1c1c24", edgecolor="#444", labelcolor=TXT)
    
    plt.tight_layout()
    plt.savefig("wilds_tu_multiplier.png", dpi=150, facecolor=DARK, bbox_inches="tight")
    print("\n✅ saved wilds_tu_multiplier.png")


def plot_sale_diagnostics(panels, fits):
    """Generate diagnostic plots for sale effects."""
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, len(TRAIN), figsize=(16, 8), facecolor=DARK)
    fig.suptitle("Sale Effect Diagnostics", color="white", fontsize=14, y=1.02)
    
    for i, game in enumerate(TRAIN):
        panel = panels[game]
        fit = fits[game]
        c = fit["model"].components()
        
        # Plot 1: Discount vs Players
        ax = axes[0, i]
        _style(ax)
        ax.scatter(panel["discount_pct"], panel["avg_players"], 
                   color=GAMES[game]["color"], alpha=0.6, s=20)
        ax.set_xlabel("Discount %")
        ax.set_ylabel("Avg Players")
        ax.set_title(f"{game}: Discount vs Players", fontweight="bold", loc="left")
        
        # Highlight sale-only periods
        sale_only = identify_sale_only_periods(panel)
        if len(sale_only) > 0:
            ax.scatter(sale_only["discount_pct"], sale_only["avg_players"],
                       color="red", alpha=0.8, s=40, label="Sale-only")
            ax.legend(fontsize=7)
        
        # Plot 2: Sale component vs Actual
        ax2 = axes[1, i]
        _style(ax2)
        if SALE_MODEL == 'additive':
            sale_comp = c['L2'] - c['L1']  # Additive sale effect
        else:
            sale_comp = (c['L2'] - c['L1']) / c['L1']  # Multiplicative effect
        ax2.plot(c['t'], sale_comp, color=GAMES[game]["color"], lw=1.5, label="Sale effect")
        ax2.set_xlabel("Months since launch")
        ax2.set_ylabel("Sale Effect")
        ax2.set_title(f"{game}: Sale Component (φ={fit['model'].params_[10]:.2f})", 
                      fontweight="bold", loc="left")
        ax2.legend(fontsize=7)
    
    plt.tight_layout()
    plt.savefig("sale_diagnostics.png", dpi=150, facecolor=DARK, bbox_inches="tight")
    print("\n✅ saved sale_diagnostics.png")

# ════════════════════════ MODEL ════════════════════════
def _core(t,a,b,lam):  return b+(a-b)*np.exp(-lam*t)
def _launch(t,L0,d0):  return L0*np.exp(-d0*t)
def _impulse(t, taus, gammas, delta):
    out = np.zeros_like(t, float)
    for tau,g in zip(taus,gammas):
        out += g*np.exp(-delta*(t-tau))*(t>=tau)
    return out
def _sale_sigmoid(eff,disc,phi,kappa):
    """Original sigmoid sale effect - price-gated with affordability threshold"""
    return phi*(disc**kappa)/(1+np.exp(AFFORD_SHARPNESS*(eff-AFFORD_THRESHOLD)))

def _sale_linear(eff,disc,phi,kappa):
    """Linear sale effect - simple multiplicative boost proportional to discount"""
    return phi * disc

def _sale_loglinear(eff,disc,phi,kappa):
    """Log-linear sale effect - handles 0% discount gracefully"""
    return phi * np.log(1 + disc) if disc > 0 else 0.0

def _sale_additive(eff,disc,phi,kappa):
    """Additive sale effect - adds players rather than multiplying"""
    return phi * disc

# Select sale function based on SALE_MODEL
_sale_funcs = {
    'sigmoid': _sale_sigmoid,
    'linear': _sale_linear,
    'loglinear': _sale_loglinear,
    'additive': _sale_additive,
}
_sale = _sale_funcs[SALE_MODEL]

class MHModel:
    """core+launch + individual-DLC impulses + shared title-update impulse + sale + spillover."""
    def __init__(self, panel, kappa_fixed=KAPPA_FIXED):
        self.panel=panel; self.kappa_fixed=kappa_fixed
        self.dlc_taus=[o for _,o,_ in panel.attrs["dlc"]]
        self.tu_taus =[o for _,o,_ in panel.attrs["tu"]]

    def _unpack(self, p):
        nd = len(self.dlc_taus)
        a,b,lam,L0,d0 = p[:5]; i=5
        dlc_g = p[i:i+nd]; i+=nd
        dlc_delta = p[i]; i+=1
        tu_g = p[i]; i+=1           # shared TU magnitude
        tu_delta = p[i]; i+=1       # shared TU decay
        phi = p[i]; i+=1
        kappa = self.kappa_fixed if self.kappa_fixed else p[i]; i+= (0 if self.kappa_fixed else 1)
        thetas = p[i:]
        return a,b,lam,L0,d0,dlc_g,dlc_delta,tu_g,tu_delta,phi,kappa,thetas

    def _predict(self, p, t, eff, disc, spill):
        a,b,lam,L0,d0,dlc_g,dlc_d,tu_g,tu_d,phi,kappa,th = self._unpack(p)
        base = _core(t,a,b,lam)+_launch(t,L0,d0)
        ev = 1 + _impulse(t,self.dlc_taus,dlc_g,dlc_d) \
               + _impulse(t,self.tu_taus,[tu_g]*len(self.tu_taus),tu_d)
        
        # Handle additive vs multiplicative sale effects
        if SALE_MODEL == 'additive':
            # Additive: sale adds players directly
            sale_effect = _sale(eff,disc,phi,kappa) * base  # Scale by base to keep units consistent
            result = base * ev + sale_effect
        else:
            # Multiplicative: sale multiplies the base+events
            mult = ev * (1 + _sale(eff,disc,phi,kappa))
            result = base * mult
        
        if spill.shape[0]: 
            if SALE_MODEL == 'additive':
                result = result * (1 + (th[:,None]*spill).sum(0))
            else:
                result = result * (1 + (th[:,None]*spill).sum(0))
        return result

    def fit(self, spill=None):
        d=self.panel; t=d["t"].values.astype(float); y=d["avg_players"].values.astype(float)
        eff=d["eff_price"].values; disc=d["discount_pct"].values
        spill=np.zeros((0,len(d))) if spill is None else spill
        nd, nsp = len(self.dlc_taus), spill.shape[0]
        def resid(p):
            pr=self._predict(p,t,eff,disc,spill)
            return np.log(np.maximum(pr,1))-np.log(np.maximum(y,1))
        y0=y[0]
        p0=[np.percentile(y,40),np.percentile(y,15),0.12,y0,1.2]
        lb=[1000,100,0.03,0,0.7]; ub=[np.percentile(y,60),np.percentile(y,38),0.30,y0*3,3.0]
        p0+=[2.0]*nd+[0.40];      lb+=[0]*nd+[0.25];   ub+=[30]*nd+[0.6]      # dlc γ + δ
        p0+=[0.5,0.6];            lb+=[0,0.3];         ub+=[15,1.5]           # shared TU γ + δ
        p0+=[1.0];                lb+=[0.0];           ub+=[8.0]              # sale φ
        if not self.kappa_fixed: p0+=[1.8]; lb+=[1.0]; ub+=[3.0]
        p0+=[0.0]*nsp;            lb+=[-3]*nsp;        ub+=[3]*nsp
        r=least_squares(resid,p0,bounds=(lb,ub),max_nfev=30000)
        self.params_=r.x
        pred=self._predict(r.x,t,eff,disc,spill)
        self.resid_log_ = np.log(np.maximum(pred,1))-np.log(np.maximum(y,1))
        ssr=np.sum(self.resid_log_**2); sst=np.sum((np.log(np.maximum(y,1))-np.log(np.maximum(y,1)).mean())**2)
        self.r2_=1-ssr/sst; self.pred_=pred; self.spill_=spill
        n,k = len(y), len(r.x)
        self.aic_ = n*np.log(ssr/n) + 2*k
        self.bic_ = n*np.log(ssr/n) + k*np.log(n)
        self.rmse_log_ = np.sqrt(np.mean(self.resid_log_**2))
        return self

    def components(self):
        d=self.panel; t=d["t"].values.astype(float)
        a,b,lam,L0,d0,dlc_g,dlc_d,tu_g,tu_d,phi,kappa,th=self._unpack(self.params_)
        eff=d["eff_price"].values; disc=d["discount_pct"].values
        core=_core(t,a,b,lam); launch=_launch(t,L0,d0); base=core+launch
        ev=1+_impulse(t,self.dlc_taus,dlc_g,dlc_d)+_impulse(t,self.tu_taus,[tu_g]*len(self.tu_taus),tu_d)
        L1=base*ev; L2=L1*(1+_sale(eff,disc,phi,kappa))
        L3=L2*(1+(th[:,None]*self.spill_).sum(0) if self.spill_.shape[0] else 1)
        return dict(t=t,core=core,launch=launch,base=base,L1=L1,L2=L2,L3=L3,
                    alpha=a,beta=b,lam=lam,L0=L0,d0=d0,dlc_g=dlc_g,dlc_d=dlc_d,
                    tu_g=tu_g,tu_d=tu_d,phi=phi,kappa=kappa,thetas=th,disc=disc,
                    hl_core=np.log(2)/lam,hl_launch=np.log(2)/d0,hl_tu=np.log(2)/tu_d)

def build_spillover(target, others):
    rows=[]
    for _,op in others.items():
        nbar=op["avg_players"].mean()
        rows.append([(op.loc[op["date"]==d-pd.DateOffset(months=1),"avg_players"].squeeze()
                      if (op["date"]==d-pd.DateOffset(months=1)).any() else 0.0)/nbar
                     for d in target["date"]])
    return np.array(rows) if rows else np.zeros((0,len(target)))

def bootstrap(model, spill, n=300, seed=0):
    rng=np.random.default_rng(seed)
    y=model.panel["avg_players"].values.astype(float); logp=np.log(np.maximum(model.pred_,1))
    lr=np.log(np.maximum(y,1))-logp; keys=["beta","lam","L0","tu_g"]; out={k:[] for k in keys}
    for _ in range(n):
        yb=np.exp(logp+rng.choice(lr,len(lr),replace=True))
        pb=model.panel.copy(); pb["avg_players"]=yb; pb.attrs.update(model.panel.attrs)
        try:
            c=MHModel(pb,model.kappa_fixed).fit(spill).components()
            for k in keys: out[k].append(c[k])
        except Exception: pass
    return {k:(np.percentile(v,5),np.median(v),np.percentile(v,95)) for k,v in out.items() if v}

# ════════════════════════ DIAGNOSTICS ════════════════════════
def ljung_box(res, lags=12):
    n=len(res); r=res-res.mean(); acf=[np.sum(r[k:]*r[:-k])/np.sum(r**2) for k in range(1,lags+1)]
    stat=n*(n+2)*np.sum([(acf[k]**2)/(n-k-1) for k in range(lags)])
    p=1-stats.chi2.cdf(stat,lags); return stat,p,acf

def baseline_simple_exp(panel):
    """Single-exponential decay baseline (β+(α−β)e^{−λt}), log-fit."""
    t=panel["t"].values.astype(float); y=panel["avg_players"].values.astype(float)
    def resid(p): return np.log(np.maximum(_core(t,*p),1))-np.log(np.maximum(y,1))
    r=least_squares(resid,[y[0],np.percentile(y,15),0.1],
                    bounds=([0,0,0.001],[np.inf,np.inf,2]),max_nfev=5000)
    pred=_core(t,*r.x); rmse=np.sqrt(np.mean((np.log(np.maximum(pred,1))-np.log(np.maximum(y,1)))**2))
    return rmse

def expanding_cv(panel, build_full=True, min_train=18):
    """Expanding-window one-step-ahead log-RMSE: full model vs simple-exp vs random-walk."""
    t=panel["t"].values.astype(float); y=panel["avg_players"].values.astype(float)
    errs={"full":[],"simple":[],"rw":[]}
    for k in range(min_train,len(y)):
        sub=panel.iloc[:k].copy(); sub.attrs.update(panel.attrs)
        ya=y[k]
        # random walk
        errs["rw"].append((np.log(max(y[k-1],1))-np.log(max(ya,1)))**2)
        # simple exp
        try:
            tt=t[:k]; yy=y[:k]
            def rs(p): return np.log(np.maximum(_core(tt,*p),1))-np.log(np.maximum(yy,1))
            rr=least_squares(rs,[yy[0],np.percentile(yy,15),0.1],
                             bounds=([0,0,1e-3],[np.inf,np.inf,2]),max_nfev=3000)
            errs["simple"].append((np.log(max(_core(np.array([t[k]]),*rr.x)[0],1))-np.log(max(ya,1)))**2)
        except Exception: pass
        # full model (no spillover in CV for speed/identifiability)
        if build_full:
            try:
                m=MHModel(sub).fit(spill=None)
                a,b,lam,L0,d0,dg,dd,tg,td,phi,kp,th=m._unpack(m.params_)
                eff=panel["eff_price"].values[k]; disc=panel["discount_pct"].values[k]
                tk=np.array([t[k]])
                base=_core(tk,a,b,lam)+_launch(tk,L0,d0)
                ev=1+_impulse(tk,m.dlc_taus,dg,dd)+_impulse(tk,m.tu_taus,[tg]*len(m.tu_taus),td)
                pr=base*ev*(1+_sale(np.array([eff]),np.array([disc]),phi,kp))
                errs["full"].append((np.log(max(pr[0],1))-np.log(max(ya,1)))**2)
            except Exception: pass
    return {k:(np.sqrt(np.mean(v)) if v else np.nan) for k,v in errs.items()}


# ════════════════════════ DRY-PERIOD (MODEL-FREE) ════════════════════════
def _event_offsets_sorted(game):
    launch=pd.Timestamp(GAMES[game]["launch"])
    return sorted(((pd.Timestamp(dt).year-launch.year)*12+(pd.Timestamp(dt).month-launch.month))
                  for _,dt,_ in EVENTS.get(game,[]))

# events that count as real *content* (a reveal/announcement is NOT content)
_CONTENT_KINDS = {"dlc","title_update","collab"}

def _content_offsets_sorted(game):
    launch=pd.Timestamp(GAMES[game]["launch"])
    return sorted(((pd.Timestamp(dt).year-launch.year)*12+(pd.Timestamp(dt).month-launch.month))
                  for _,dt,k in EVENTS.get(game,[]) if k in _CONTENT_KINDS)

def content_gaps(game, min_len=2):
    """Inter-content gaps: >=min_len clear months strictly between two events.
       Plus a TRAILING dry period for games still in active support (awaiting
       announced-but-unreleased content). For retired games ('ended'), the
       post-final-content tail is NOT a drought — the game is simply done — so
       it is excluded."""
    evs=_event_offsets_sorted(game); out=[]
    for a,b in zip(evs[:-1],evs[1:]):
        if b-a>=min_len+1: out.append((a+1,b-1,"inter"))
    # trailing dry period — only for ACTIVE games
    sup=SUPPORT.get(game,{})
    if sup.get("status")=="active":
        launch=pd.Timestamp(GAMES[game]["launch"])
        fc=pd.Timestamp(sup["final_content"])
        fc_off=(fc.year-launch.year)*12+(fc.month-launch.month)
        out.append((fc_off+1, 9999, "trailing"))   # 9999 clipped to data end downstream
    return out

def dry_period_rotation(panels):
    """Model-free: during each game's content gaps (inter-content + trailing-while-active),
       measure its own change vs the others'. Retired games contribute only their
       genuine inter-content gaps, never their post-support tail."""
    rows=[]
    for g in GAMES:
        p=panels[g]; tmax=int(p["t"].max())
        for a,b,kind in content_gaps(g):
            b=min(b,tmax)
            sub=p[(p.t>=a)&(p.t<=b)]
            if len(sub)<2: continue
            d0,d1=sub["date"].iloc[0],sub["date"].iloc[-1]
            self_chg=(sub["avg_players"].iloc[-1]/sub["avg_players"].iloc[0]-1)*100
            others={}
            for o in GAMES:
                if o==g: continue
                # only count an 'other' game if IT is still in active support during
                # the window (an ended game can't 'receive' rotated players meaningfully)
                osup=SUPPORT.get(o,{})
                po=panels[o]; so=po[(po["date"]>=d0)&(po["date"]<=d1)]
                if len(so)<2: continue
                # require the other game to be live (has data) in window
                others[o]=(so["avg_players"].iloc[-1]/so["avg_players"].iloc[0]-1)*100
            rows.append(dict(game=g,t0=a,t1=b,d0=d0,d1=d1,kind=kind,
                             self_chg=self_chg,others=others))
    return rows

# ════════════════════════ DASHBOARD ════════════════════════
DARK="#0e0e14"; PANEL="#15151c"; TXT="#cfcfd6"; GRID="#24242e"
def _style(ax):
    ax.set_facecolor(PANEL); ax.grid(True,color=GRID,lw=.5,ls="--")
    ax.tick_params(colors=TXT,labelsize=8); ax.title.set_color("white")
    ax.xaxis.label.set_color(TXT); ax.yaxis.label.set_color(TXT)
    for s in ax.spines.values(): s.set_edgecolor("#333")

def _evlines(ax, panel, use_offset=True, ymax=None):
    g=panel.attrs["game"]; launch=pd.Timestamp(GAMES[g]["launch"])
    for nm,dt,kind in EVENTS.get(g,[]):
        d=pd.Timestamp(dt)
        x=((d.year-launch.year)*12+(d.month-launch.month)) if use_offset else d
        c={"dlc":"#ffffff","title_update":"#9b9bd0","collab":"#d0a050","event":"#50d0c0"}[kind]
        ls="-" if kind=="dlc" else ":"
        ax.axvline(x,color=c,ls=ls,lw=1.0 if kind=="dlc" else .7,alpha=.55,zorder=1)

def run(keep_partial=True, log_scale=True, sale_model=None):
    # Override global SALE_MODEL if specified
    global SALE_MODEL
    if sale_model:
        SALE_MODEL = sale_model
    
    panels=all_panels(keep_partial); fits={}
    print(f"="*64 + f"  [Sale model: {SALE_MODEL}]")
    for g in TRAIN:
        others={k:v for k,v in panels.items() if k!=g and v["t"].min()<=panels[g]["t"].max()}
        sp=build_spillover(panels[g],others)
        m=MHModel(panels[g]).fit(spill=sp); ci=bootstrap(m,sp,n=250)
        fits[g]=dict(model=m,spill=sp,others=list(others),ci=ci)
        c=m.components()
        lbstat,p_lb,_=ljung_box(m.resid_log_)
        print(f"\n{g}  R²={m.r2_:.3f}  AIC={m.aic_:.0f}  BIC={m.bic_:.0f}  "
              f"RMSE_log={m.rmse_log_:.3f}  (n={len(panels[g])} through {panels[g]['date'].max():%b %Y})")
        print(f"  core β={ci['beta'][1]:,.0f} [{ci['beta'][0]:,.0f},{ci['beta'][2]:,.0f}]  "
              f"λ HL={np.log(2)/ci['lam'][1]:.1f}mo  launch L0={c['L0']:,.0f}")
        print(f"  DLC γ={[f'{x:.1f}' for x in c['dlc_g']]}  "
              f"TU γ={c['tu_g']:.2f} [{ci['tu_g'][0]:.2f},{ci['tu_g'][2]:.2f}] (HL {c['hl_tu']:.1f}mo)  φ_sale={c['phi']:.2f}")
        print(f"  spill: "+", ".join(f"{k}={v:+.2f}" for k,v in zip(others,c['thetas'])))
        print(f"  Ljung-Box p={p_lb:.3f} {'(autocorrelated residuals)' if p_lb<.05 else '(clean)'}")
        
        # Print sale-only period info
        sale_only = identify_sale_only_periods(panels[g])
        if len(sale_only) > 0:
            print(f"  Sale-only periods: {len(sale_only)} months (avg discount: {sale_only['discount_pct'].mean():.1%})")
        else:
            print(f"  Sale-only periods: NONE (all sales coincide with events)")
        
        fits[g]["diag"]=dict(lb_p=p_lb)

    # Fit Wilds model separately to extract its parameters
    print("\n" + "="*64 + "\nFitting Wilds model separately...")
    wilds_panel = panels[TEST]
    wilds_others = {k:v for k,v in panels.items() if k!=TEST and v["t"].min()<=wilds_panel["t"].max()}
    wilds_sp = build_spillover(wilds_panel, wilds_others)
    wilds_model = MHModel(wilds_panel).fit(spill=wilds_sp)
    wilds_ci = bootstrap(wilds_model, wilds_sp, n=250)
    wilds_c = wilds_model.components()
    wilds_lb, wilds_p_lb, _ = ljung_box(wilds_model.resid_log_)
    
    print(f"\n{TEST}  R²={wilds_model.r2_:.3f}  AIC={wilds_model.aic_:.0f}  BIC={wilds_model.bic_:.0f}  "
          f"RMSE_log={wilds_model.rmse_log_:.3f}  (n={len(wilds_panel)} through {wilds_panel['date'].max():%b %Y})")
    print(f"  core β={wilds_ci['beta'][1]:,.0f} [{wilds_ci['beta'][0]:,.0f},{wilds_ci['beta'][2]:,.0f}]  "
          f"λ HL={np.log(2)/wilds_ci['lam'][1]:.1f}mo  launch L0={wilds_c['L0']:,.0f} (HL={wilds_c['hl_launch']:.1f}mo)  d0={wilds_c['d0']:.4f}")
    print(f"  DLC γ={[f'{x:.1f}' for x in wilds_c['dlc_g']]}  "
          f"TU γ={wilds_c['tu_g']:.2f} [{wilds_ci['tu_g'][0]:.2f},{wilds_ci['tu_g'][2]:.2f}] (HL {wilds_c['hl_tu']:.1f}mo)  φ_sale={wilds_c['phi']:.2f}")
    print(f"  spill: "+", ".join(f"{k}={v:+.2f}" for k,v in zip(wilds_others,wilds_c['thetas'])))
    print(f"  Ljung-Box p={wilds_p_lb:.3f} {'(autocorrelated residuals)' if wilds_p_lb<.05 else '(clean)'}")
    
    fits[TEST] = dict(model=wilds_model, spill=wilds_sp, others=list(wilds_others), ci=wilds_ci)

    # held-out comparison using World's template
    wp=panels[TEST]; wt=wp["t"].values.astype(float); wy=wp["avg_players"].values.astype(float)
    wc=fits["World"]["model"].components()
    shape=_core(wt,wc["alpha"],wc["beta"],wc["lam"])+_launch(wt,wc["L0"],wc["d0"])
    shape=shape/shape[0]*wy[0]
    print("\n"+"="*64+f"\n{TEST} (HELD OUT) vs World-template:")
    for i in range(len(wt)):
        mk=" ← latest" if i==len(wt)-1 else ""
        print(f"  t={int(wt[i]):2d} {wp['date'].iloc[i]:%b %Y}: {wy[i]:>8,.0f} vs {shape[i]:>8,.0f} ({(wy[i]-shape[i])/shape[i]*100:+.0f}%){mk}")

    _dashboard(panels,fits,wp,shape,log_scale)
    return panels,fits

def _evlines(ax, panel, use_offset=True):
    g=panel.attrs["game"]; launch=pd.Timestamp(GAMES[g]["launch"])
    for nm,dt,kind in EVENTS.get(g,[]):
        d=pd.Timestamp(dt)
        x=((d.year-launch.year)*12+(d.month-launch.month)) if use_offset else d
        if kind=="dlc":
            ax.axvline(x,color="#ffffff",ls="-",lw=1.8,alpha=.9,zorder=6)
            ax.text(x,ax.get_ylim()[1],f" {nm}",color="#fff",fontsize=7,rotation=90,va="top",ha="left",fontweight="bold",zorder=7)
        else:
            c={"title_update":"#9b9bd0","collab":"#d0a050","event":"#50d0c0"}[kind]
            ax.axvline(x,color=c,ls=":",lw=.8,alpha=.5,zorder=1)

def _dashboard(panels,fits,wp,shape,log_scale):
    plt.style.use("dark_background")
    fig=plt.figure(figsize=(20,26),facecolor=DARK)
    gs=gridspec.GridSpec(7,2,figure=fig,hspace=.5,wspace=.18,
                         left=.06,right=.97,top=.965,bottom=.025)
    fig.suptitle("Monster Hunter — Lifecycle Model Diagnostic Dashboard",
                 color="white",fontsize=18,fontweight="bold",y=.985)

    # ── Rows 0–1: per-game ABSOLUTE (linear) decomposition + MULTIPLIER view ──
    for r,g in enumerate(TRAIN):
        m=fits[g]["model"]; c=m.components(); d=m.panel
        t=c["t"]; y=d["avg_players"].values; col=GAMES[g]["color"]
        # LEFT: absolute linear, stacked layers (spikes visible at true magnitude)
        ax=fig.add_subplot(gs[r,0]); _style(ax)
        ax.fill_between(t,0,c["core"]/1e3,color=col,alpha=.5,label="core",lw=0)
        ax.fill_between(t,c["core"]/1e3,c["base"]/1e3,color="#ffd27f",alpha=.4,label="launch",lw=0)
        ax.fill_between(t,c["base"]/1e3,c["L1"]/1e3,color="#9b7fe0",alpha=.55,label="events",lw=0)
        ax.fill_between(t,c["L1"]/1e3,c["L2"]/1e3,color="#5fd0a0",alpha=.6,label="sale",lw=0)
        ax.fill_between(t,c["L2"]/1e3,c["L3"]/1e3,color="#d0607f",alpha=.5,label="spill",lw=0)
        ax.plot(t,y/1e3,color="white",lw=1.5,label="actual",zorder=4)
        ax.plot(t,m.pred_/1e3,color="white",lw=1.0,ls="--",alpha=.8,label="model",zorder=5)
        _evlines(ax,d,use_offset=True)
        ax.set_title(f"{g} — absolute decomposition (linear, R²={m.r2_:.2f})",fontweight="bold",loc="left")
        ax.set_xlabel("months since launch"); ax.set_ylabel("avg concurrent (k)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:.0f}k"))
        ax.legend(fontsize=6.5,ncol=4,loc="upper right",facecolor="#1c1c24",edgecolor="#444",labelcolor=TXT)

        # RIGHT: MULTIPLIER view — departures above 1.0 baseline (scale-free)
        ax2=fig.add_subplot(gs[r,1]); _style(ax2)
        base=np.maximum(c["base"],1)
        ev_mult   = c["L1"]/base
        sale_mult = c["L2"]/c["L1"]
        spill_mult= c["L3"]/c["L2"]
        actual_mult = y/base
        ax2.axhline(1.0,color="#888",lw=1,ls="--",zorder=2)
        ax2.fill_between(t,1.0,ev_mult,color="#9b7fe0",alpha=.55,label="events ×",lw=0)
        ax2.plot(t,sale_mult,color="#5fd0a0",lw=1.3,label="sale ×",zorder=3)
        ax2.plot(t,spill_mult,color="#d0607f",lw=1.3,label="spill ×",zorder=3)
        ax2.plot(t,actual_mult,color="white",lw=1.3,label="actual/baseline",zorder=4)
        _evlines(ax2,d,use_offset=True)
        ax2.set_title(f"{g} — multiplier view (× above core+launch baseline)",fontweight="bold",loc="left")
        ax2.set_xlabel("months since launch"); ax2.set_ylabel("× baseline")
        ax2.legend(fontsize=6.5,ncol=4,loc="upper right",facecolor="#1c1c24",edgecolor="#444",labelcolor=TXT)

    # ── Row 2: calendar (linear) + Wilds held-out (linear) ──
    ax=fig.add_subplot(gs[2,0]); _style(ax)
    for g in GAMES:
        p=panels[g]; ax.plot(p["date"],p["avg_players"]/1e3,color=GAMES[g]["color"],lw=1.6,label=g)
        _evlines(ax,p,use_offset=False)
    ax.set_title("All titles — calendar (linear, event lines)",fontweight="bold",loc="left")
    ax.set_xlabel("date"); ax.set_ylabel("avg concurrent (k)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:.0f}k"))
    ax.tick_params(axis="x",rotation=30)
    ax.legend(fontsize=8,facecolor="#1c1c24",edgecolor="#444",labelcolor=TXT)

    ax=fig.add_subplot(gs[2,1]); _style(ax)
    wt=wp["t"].values.astype(float); wy=wp["avg_players"].values
    ax.plot(wt,wy/1e3,color=GAMES["Wilds"]["color"],lw=2,label="Wilds actual",zorder=3)
    ax.plot(wt,shape/1e3,color="white",lw=1.2,ls="--",label="World template")
    ax.fill_between(wt,wy/1e3,shape/1e3,where=(wy<shape),alpha=.25,color="#e0593b")
    _evlines(ax,wp,use_offset=True)
    ax.set_title("Wilds held-out vs template (linear)",fontweight="bold",loc="left")
    ax.set_xlabel("months since launch"); ax.set_ylabel("avg concurrent (k)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:.0f}k"))
    ax.legend(fontsize=8,facecolor="#1c1c24",edgecolor="#444",labelcolor=TXT)

    # ── Row 3: residual time-series + ACF (World focus, both shown) ──
    for ci_,g in enumerate(TRAIN):
        ax=fig.add_subplot(gs[3,ci_]); _style(ax)
        m=fits[g]["model"]; _,p_lb,acf=ljung_box(m.resid_log_)
        t=m.panel["t"].values.astype(float)
        ax.axhline(0,color="#666",lw=.8); ax.bar(t,m.resid_log_,color=GAMES[g]["color"],alpha=.7,width=.8)
        ax.set_title(f"{g} — log residuals (Ljung-Box p={p_lb:.3f})",fontweight="bold",loc="left")
        ax.set_xlabel("months since launch"); ax.set_ylabel("log resid")

    # ── Row 4: residual ACF both ──
    for ci_,g in enumerate(TRAIN):
        ax=fig.add_subplot(gs[4,ci_]); _style(ax)
        m=fits[g]["model"]; _,p_lb,acf=ljung_box(m.resid_log_)
        cib=1.96/np.sqrt(len(m.resid_log_))
        ax.bar(range(1,len(acf)+1),acf,color=GAMES[g]["color"],alpha=.75)
        ax.axhline(cib,color="#888",ls="--",lw=.8); ax.axhline(-cib,color="#888",ls="--",lw=.8); ax.axhline(0,color="#666",lw=.8)
        ax.set_title(f"{g} — residual ACF (95% bands)",fontweight="bold",loc="left")
        ax.set_xlabel("lag (months)"); ax.set_ylabel("autocorr")

    # ── Row 5: DRY-PERIOD rotation (model-free) + variance decomposition ──
    ax=fig.add_subplot(gs[5,0]); _style(ax)
    rot=dry_period_rotation(panels)
    xlabs=[]; selfs=[]; oth=[]; edge=[]
    for r_ in rot:
        if not r_["others"]: continue
        tag="*" if r_["kind"]=="trailing" else ""
        xlabs.append(f"{r_['game'][:2]}:{r_['d0']:%b%y}{tag}")
        selfs.append(r_["self_chg"]); oth.append(np.mean(list(r_["others"].values())))
        edge.append("#fff" if r_["kind"]=="trailing" else "none")
    xx=np.arange(len(xlabs)); w=.4
    ax.bar(xx-w/2,selfs,w,label="this game Δ%",color="#888",edgecolor=edge,lw=1.5)
    ax.bar(xx+w/2,oth,w,label="other MH games Δ% (mean)",color="#6dbf67",edgecolor=edge,lw=1.5)
    ax.axhline(0,color="#aaa",lw=.8)
    ax.set_xticks(xx); ax.set_xticklabels(xlabs,fontsize=7,rotation=30)
    ax.set_title("DRY-PERIOD rotation (model-free): * = trailing (game still awaiting content)",fontweight="bold",loc="left")
    ax.set_ylabel("% change over gap")
    ax.legend(fontsize=8,facecolor="#1c1c24",edgecolor="#444",labelcolor=TXT)

    ax=fig.add_subplot(gs[5,1]); _style(ax)
    # variance decomposition: share of fitted log-signal var per layer
    names=["core+launch","events","sale","spill"]; cols=["#e07b39","#9b7fe0","#5fd0a0","#d0607f"]
    width=.35
    for gi,g in enumerate(TRAIN):
        c=fits[g]["model"].components()
        lb=np.log(np.maximum(c["base"],1)); l1=np.log(np.maximum(c["L1"],1))
        l2=np.log(np.maximum(c["L2"],1)); l3=np.log(np.maximum(c["L3"],1))
        contribs=[np.var(lb),np.var(l1-lb),np.var(l2-l1),np.var(l3-l2)]
        tot=sum(contribs); shares=[x/tot*100 for x in contribs]
        bottom=0
        for s,cc,nm in zip(shares,cols,names):
            ax.bar(gi,s,width,bottom=bottom,color=cc,label=nm if gi==0 else None)
            bottom+=s
    ax.set_xticks(range(len(TRAIN))); ax.set_xticklabels(TRAIN)
    ax.set_title("Variance share by component (log-signal)",fontweight="bold",loc="left")
    ax.set_ylabel("% of fitted variance")
    ax.legend(fontsize=7,facecolor="#1c1c24",edgecolor="#444",labelcolor=TXT)

    # ── Row 6: spillover + parameter text ──
    ax=fig.add_subplot(gs[6,0]); _style(ax); ax.axis("off")
    lines=["SPILLOVER θ (lagged, normalized)\n"]
    for g in TRAIN:
        c=fits[g]["model"].components()
        for k,v in zip(fits[g]["others"],c["thetas"]):
            lines.append(f"  {k} -> {g}:  {v:+.3f}  ({'feeds in' if v>0 else 'competes'})")
    lines.append("\nDRY-PERIOD ROTATION (model-free)")
    for r_ in rot:
        if not r_["others"]: continue
        os=", ".join(f"{o}{v:+.0f}%" for o,v in r_["others"].items())
        lines.append(f"  {r_['game']} gap {r_['d0']:%b%y}-{r_['d1']:%b%y}: self{r_['self_chg']:+.0f}% | {os}")
    ax.text(.02,.98,"\n".join(lines),color=TXT,fontsize=9.5,va="top",family="monospace")

    ax=fig.add_subplot(gs[6,1]); _style(ax); ax.axis("off")
    tab=[f"PARAMETERS  (sale model: {SALE_MODEL})\n"]
    for g in TRAIN:
        m=fits[g]["model"]; c=m.components()
        tab.append(f"{g}:  R2={m.r2_:.2f}")
        tab.append(f"  alpha={c['alpha']:,.0f} beta(floor)={c['beta']:,.0f}")
        tab.append(f"  core HL={c['hl_core']:.1f}mo  launch HL={c['hl_launch']:.2f}mo")
        tab.append(f"  DLC gamma={[f'{x:.1f}' for x in c['dlc_g']]}  TU gamma={c['tu_g']:.2f} (HL {c['hl_tu']:.1f}mo)")
        tab.append(f"  phi_sale={c['phi']:.2f}\n")
    ax.text(.02,.98,"\n".join(tab),color=TXT,fontsize=9.5,va="top",family="monospace")

    plt.savefig("dashboard.png",dpi=115,facecolor=DARK,bbox_inches="tight")
    print("\n✅ saved dashboard.png")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--no-partial",action="store_true")
    ap.add_argument("--linear",action="store_true")
    ap.add_argument("--sale-model", choices=["sigmoid", "linear", "loglinear", "additive"], 
                    default="linear", help="Sale effect model: sigmoid (original), linear, loglinear, or additive")
    a=ap.parse_args()
    panels, fits = run(keep_partial=not a.no_partial, log_scale=not a.linear, 
                       sale_model=a.sale_model)
    
    # Generate sale diagnostics
    plot_sale_diagnostics(panels, fits)
    
    # Generate Wilds TU multiplier plot
    plot_wilds_tu_multiplier(panels, fits)
