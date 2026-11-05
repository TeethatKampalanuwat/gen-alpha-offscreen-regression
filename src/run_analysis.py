"""
Full analysis of the real survey responses.

    python src/clean.py && python src/run_analysis.py

Tables -> results/, charts -> figures/.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from stats_tools import backward_elimination, diagnostics, kfold_cv, ols, standardized_coefs, vif

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
RES.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

Y_RAW, Y = "Y_offscreen_min_week", "log_Y"
XS = ["X1_enjoyment", "X2_good_weather", "X3_friends", "X4_parent_support",
      "X5_screen_entertain_h", "X6_screen_study_h"]
LABEL = {
    Y_RAW: "Off-screen time (min/week)", Y: "log(1 + off-screen min/week)",
    "X1_enjoyment": "X1 Enjoyment (1-10)", "X2_good_weather": "X2 Good weather (0/1)",
    "X3_friends": "X3 Friends joining", "X4_parent_support": "X4 Parental support (1-5)",
    "X5_screen_entertain_h": "X5 Entertainment screen (h/day)",
    "X6_screen_study_h": "X6 Study screen (h/day)",
}
SHORT = {k: v.split(" (")[0] for k, v in LABEL.items()}
BLUE, RED, GREY, LIGHT = "#2f6db5", "#c8463d", "#8a8f98", "#dfe7f2"
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9.5})


def md(df: pd.DataFrame, digits=3) -> str:
    return df.round(digits).to_markdown()


# ---------------------------------------------------------------- data
def load():
    all_ = pd.read_csv(ROOT / "data/processed/survey_coded.csv")
    ga = all_[all_.gen_alpha == 1].reset_index(drop=True)
    flow = pd.DataFrame({
        "Step": ["Form responses", "Exclude age 19+ (not Gen Alpha)", "Analysis sample"],
        "Removed": [None, int((all_.gen_alpha == 0).sum()), None],
        "n": [len(all_), len(ga), len(ga)],
    })
    flow.to_csv(RES / "01_sample_flow.csv", index=False)
    return all_, ga, flow


# ---------------------------------------------------------------- descriptive
def describe(ga):
    cols = [Y_RAW] + XS
    d = ga[cols].describe().T[["count", "mean", "std", "min", "50%", "max"]]
    d.columns = ["N", "Mean", "StDev", "Min", "Median", "Max"]
    d["Skewness"] = [stats.skew(ga[c]) for c in cols]
    d.index = [LABEL[c] for c in cols]
    d.to_csv(RES / "02_descriptive_stats.csv", float_format="%.3f")

    fig, axes = plt.subplots(2, 4, figsize=(13, 6))
    for ax, col in zip(axes.flat, [Y_RAW, Y] + XS):
        v = ga[col]
        if col in (Y_RAW, Y):
            ax.hist(v, bins=12, color=BLUE, edgecolor="white")
        else:
            lo, hi = np.floor(v.min()), np.ceil(v.max())
            ax.hist(v, bins=np.arange(lo - 0.5, hi + 1.5, 1), color=BLUE, edgecolor="white")
        ax.set_title(LABEL[col], fontsize=9)
        ax.set_ylabel("Frequency")
    fig.suptitle(f"Gen Alpha respondents (n = {len(ga)}): distribution of each variable",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "01_distributions.png")
    plt.close(fig)
    return d


def correlation(ga):
    cols = [Y] + XS
    r = ga[cols].corr()
    p = pd.DataFrame(np.ones_like(r), index=cols, columns=cols)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            p.loc[a, b] = p.loc[b, a] = stats.pearsonr(ga[a], ga[b]).pvalue
    r.to_csv(RES / "03_correlation_matrix.csv", float_format="%.3f")
    p.to_csv(RES / "03_correlation_pvalues.csv", float_format="%.4f")

    fig, ax = plt.subplots(figsize=(7.2, 6))
    mask = np.triu(np.ones_like(r, dtype=bool), k=1)
    im = ax.imshow(np.ma.array(r, mask=mask), cmap="RdBu_r", vmin=-1, vmax=1)
    names = [SHORT[c] for c in cols]
    ax.set_xticks(range(len(cols)), names, rotation=40, ha="right")
    ax.set_yticks(range(len(cols)), names)
    for i in range(len(cols)):
        for j in range(i):
            v = r.iloc[i, j]
            ax.text(j, i, f"{v:.2f}{'*' if p.iloc[i, j] < 0.05 else ''}", ha="center",
                    va="center", fontsize=8, color="white" if abs(v) > 0.6 else "black")
    ax.spines[:].set_visible(False)
    fig.colorbar(im, ax=ax, shrink=0.75, label="Pearson r")
    ax.set_title(f"Correlation matrix, Gen Alpha (n = {len(ga)})  * p < 0.05", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "02_correlation_heatmap.png")
    plt.close(fig)
    return r, p


# ---------------------------------------------------------------- modelling
def bootstrap(df, y, xs, B=5000, seed=1):
    rng = np.random.default_rng(seed)
    n = len(df)
    draws = []
    for _ in range(B):
        s = df.iloc[rng.integers(n, size=n)]
        try:
            draws.append(ols(s, y, xs).coef)
        except np.linalg.LinAlgError:
            continue
    draws = np.array(draws)
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    return pd.DataFrame({"Boot 2.5%": lo, "Boot 97.5%": hi}, index=["Constant"] + list(xs))


def stepwise_table(steps):
    rows = []
    for i, s in enumerate(steps, 1):
        removed = "—" if i == 1 else next(t for t in steps[i - 2].terms if t not in s.terms)
        rows.append({"Step": i, "Removed": SHORT.get(removed, removed),
                     "Predictors left": len(s.terms) - 1, "S": s.s,
                     "R-sq %": 100 * s.r2, "R-sq(adj) %": 100 * s.r2_adj,
                     "Mallows Cp": s.mallows_cp, "BIC": s.bic})
    t = pd.DataFrame(rows).set_index("Step")
    t.to_csv(RES / "05_backward_elimination.csv", float_format="%.3f")
    return t


def coef_table(df, res, y):
    v = vif(df, res.terms[1:]) if res.k > 1 else {res.terms[1]: 1.0}
    t = res.table(v)
    t = t.join(bootstrap(df, y, res.terms[1:]))
    tcrit = stats.t.ppf(0.975, res.n - res.k - 1)
    t.insert(4, "95% CI low", t["Coef"] - tcrit * t["SE Coef"])
    t.insert(5, "95% CI high", t["Coef"] + tcrit * t["SE Coef"])
    return t


def robustness(all_, ga):
    """Is the X1 effect stable across reasonable analysis choices? (full 6-predictor models)"""
    no_out = ga[ga[Y_RAW] < 1000]
    specs = [
        ("Main: Gen Alpha, log(1+Y)", ga, Y),
        ("Gen Alpha, raw Y", ga, Y_RAW),
        ("Gen Alpha, raw Y, drop 1,680-min case", no_out, Y_RAW),
        ("Gen Alpha, drop non-participants (Y=0)", ga[ga[Y_RAW] > 0], Y),
        ("All ages (n=85), log(1+Y)", all_, Y),
        ("All ages, raw Y, drop 1,680-min case", all_[all_[Y_RAW] < 1000], Y_RAW),
    ]
    rows = []
    for name, d, y in specs:
        r = ols(d, y, XS)
        j = r.terms.index("X1_enjoyment")
        scale = d["X1_enjoyment"].std() / d[y].std()
        tc = stats.t.ppf(0.975, r.n - r.k - 1)
        rows.append({"Specification": name, "n": r.n,
                     "X1 std beta": r.coef[j] * scale,
                     "CI low": (r.coef[j] - tc * r.se[j]) * scale,
                     "CI high": (r.coef[j] + tc * r.se[j]) * scale,
                     "X1 p": r.p[j], "Model R-sq %": 100 * r.r2, "Model F p": r.f_p,
                     "Other predictors with p<0.05": ", ".join(
                         SHORT[t] for t, pv in zip(r.terms[1:], r.p[1:])
                         if pv < 0.05 and t != "X1_enjoyment") or "none"})
    t = pd.DataFrame(rows).set_index("Specification")
    t.to_csv(RES / "07_robustness.csv", float_format="%.4f")

    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    yy = np.arange(len(t))[::-1]
    ax.hlines(yy, t["CI low"], t["CI high"], color=BLUE, lw=2)
    ax.plot(t["X1 std beta"], yy, "o", color=BLUE, ms=7)
    ax.axvline(0, color=GREY, ls="--", lw=1)
    ax.set_yticks(yy, [f"{s}  (n={n})" for s, n in zip(t.index, t.n)])
    ax.set_xlabel("Standardized effect of enjoyment (β, 95% CI) — full 6-predictor model")
    ax.set_title("Enjoyment effect across six analysis choices",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "05_robustness_enjoyment.png")
    plt.close(fig)
    return t


def participation(ga):
    """Do non-participants (0 min/week) differ from participants? Mann-Whitney per predictor."""
    zero = ga[Y_RAW] == 0
    rows = []
    for c in XS:
        a, b = ga.loc[zero, c], ga.loc[~zero, c]
        rows.append({"Predictor": SHORT[c], "Mean, 0 min (n=%d)" % zero.sum(): a.mean(),
                     "Mean, >0 min (n=%d)" % (~zero).sum(): b.mean(),
                     "Mann-Whitney p": stats.mannwhitneyu(a, b).pvalue})
    part = pd.DataFrame(rows).set_index("Predictor")
    act = ga[~zero]
    rho = stats.spearmanr(act.X1_enjoyment, act[Y_RAW])
    part.to_csv(RES / "08_participation.csv", float_format="%.4f")
    return part, rho


# ---------------------------------------------------------------- figures
def enjoyment_figure(ga):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    levels = sorted(ga.X1_enjoyment.unique())
    groups = [ga.loc[ga.X1_enjoyment == k, Y_RAW] for k in levels]
    ax[0].boxplot(groups, positions=levels, widths=0.55, patch_artist=True,
                  boxprops=dict(facecolor=LIGHT, edgecolor=BLUE),
                  medianprops=dict(color=RED, lw=1.5), showfliers=False)
    jitter = np.random.default_rng(0).uniform(-0.15, 0.15, len(ga))
    ax[0].scatter(ga.X1_enjoyment + jitter, ga[Y_RAW], s=16, color=BLUE, zorder=3)
    ax[0].set_yscale("symlog", linthresh=60)
    ax[0].set_yticks([0, 60, 120, 240, 600, 1680], ["0", "60", "120", "240", "600", "1,680"])
    ax[0].set(xlabel="Enjoyment rating of favourite activity (1-10)",
              ylabel="Off-screen min/week (log scale)",
              title="Higher enjoyment goes with more off-screen time")
    med = ga.groupby(ga.X1_enjoyment >= 9)[Y_RAW].median()
    ax[0].text(0.02, 0.97, f"Median: rating ≤ 8 → {med[False]:.0f} min, rating 9-10 → {med[True]:.0f} min",
               transform=ax[0].transAxes, va="top", fontsize=8.5)

    jit = np.random.default_rng(1).uniform(-0.12, 0.12, len(ga))
    ax[1].scatter(ga.X5_screen_entertain_h + jit, ga[Y_RAW], s=16, color=GREY)
    ax[1].set_yscale("symlog", linthresh=60)
    ax[1].set_yticks([0, 60, 120, 240, 600, 1680], ["0", "60", "120", "240", "600", "1,680"])
    r = stats.pearsonr(ga.X5_screen_entertain_h, ga[Y])
    ax[1].set(xlabel="Entertainment screen time (h/day)", ylabel="Off-screen min/week (log scale)",
              title=f"Entertainment screen time: weak link (r = {r.statistic:.2f}, p = {r.pvalue:.2f})")
    fig.tight_layout()
    fig.savefig(FIG / "03_enjoyment_vs_screen.png")
    plt.close(fig)


def residual_plots(res, title):
    e = res.resid
    fig, ax = plt.subplots(2, 2, figsize=(10, 7.5))
    (osm, osr), (slope, icpt, _) = stats.probplot(e, dist="norm")
    ax[0, 0].scatter(osr, osm, s=14, color=BLUE)
    ax[0, 0].plot(slope * osm + icpt, osm, color=RED, lw=1)
    ax[0, 0].set(title="Normal probability plot", xlabel="Residual", ylabel="Theoretical quantile")
    ax[0, 1].scatter(res.fitted, e, s=14, color=BLUE)
    ax[0, 1].axhline(0, color=GREY, ls="--", lw=1)
    ax[0, 1].set(title="Versus fits", xlabel="Fitted value", ylabel="Residual")
    ax[1, 0].hist(e, bins=12, color=BLUE, edgecolor="white")
    ax[1, 0].set(title="Histogram", xlabel="Residual", ylabel="Frequency")
    ax[1, 1].plot(range(1, len(e) + 1), e, "-o", ms=3, lw=0.7, color=BLUE)
    ax[1, 1].axhline(0, color=GREY, ls="--", lw=1)
    ax[1, 1].set(title="Versus order", xlabel="Observation order", ylabel="Residual")
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "04_residual_plots.png")
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    all_, ga, flow = load()
    desc = describe(ga)
    r, p = correlation(ga)

    full = ols(ga, Y, XS)
    full_t = coef_table(ga, full, Y)
    full_t.to_csv(RES / "04_full_model.csv", float_format="%.4f")

    steps = backward_elimination(ga, Y, XS, alpha_remove=0.10)
    step_t = stepwise_table(steps)
    final = steps[-1]
    fx = final.terms[1:]
    final_t = coef_table(ga, final, Y)
    final_t.to_csv(RES / "06_final_model.csv", float_format="%.4f")
    diag = diagnostics(final, ga, fx)
    beta = standardized_coefs(ga, Y, final)
    cv = kfold_cv(ga, Y, fx, k=5, repeats=50)
    cv_null = float(np.sqrt(np.mean((ga[Y] - ga[Y].mean()) ** 2)) * np.sqrt(len(ga) / (len(ga) - 1)))
    rob = robustness(all_, ga)
    part, rho = participation(ga)
    enjoyment_figure(ga)
    residual_plots(final, f"Residual plots — final model on log(1+Y), n = {final.n}")

    b1 = final.coef[final.terms.index("X1_enjoyment")]
    lo1, hi1 = final_t.loc["X1_enjoyment", ["95% CI low", "95% CI high"]]
    pct = lambda b: 100 * (np.exp(b) - 1)
    eq = "log(1+Y) = " + f"{final.coef[0]:.3f}" + "".join(
        f" {c:+.3f}·{SHORT[t].split()[0]}" for t, c in zip(fx, final.coef[1:]))

    with open(RES / "summary.md", "w") as f:
        f.write("# Results summary (generated by src/run_analysis.py)\n\n")
        f.write("## Sample\n\n" + flow.to_markdown(index=False) + "\n\n")
        f.write("## Descriptive statistics (Gen Alpha)\n\n" + md(desc, 2) + "\n\n")
        f.write("## Correlation with log(1+Y)\n\n" + md(pd.DataFrame(
            {"r": r[Y].drop(Y), "p": p[Y].drop(Y)}), 3) + "\n\n")
        f.write(f"## Full model (6 predictors)\n\nR² = {100*full.r2:.1f}%, adj R² = "
                f"{100*full.r2_adj:.1f}%, F-test p = {full.f_p:.3f}\n\n" + md(full_t, 3) + "\n\n")
        f.write("## Backward elimination (α-to-remove = 0.10)\n\n" + md(step_t, 3) + "\n\n")
        f.write(f"## Final model\n\n`{eq}`\n\n")
        f.write(f"n = {final.n}, S = {final.s:.3f}, R² = {100*final.r2:.1f}%, adj R² = "
                f"{100*final.r2_adj:.1f}%, predicted R² = {100*final.r2_pred:.1f}%, "
                f"F-test p = {final.f_p:.4f}\n\n" + md(final_t, 3) + "\n\n")
        f.write("Standardized β: " + ", ".join(f"{SHORT[k]} {v:+.2f}" for k, v in beta.items()) + "\n\n")
        f.write(f"Interpretation: each +1 enjoyment point multiplies (1 + weekly off-screen minutes) "
                f"by e^{b1:.3f} ≈ {pct(b1):+.0f}% (95% CI {pct(lo1):+.0f}% to {pct(hi1):+.0f}%), "
                f"holding the other terms fixed.\n\n")
        f.write("## Assumption checks (final model)\n\n")
        f.write(f"- Normality: Shapiro–Wilk p = {diag['shapiro_wilk_p']:.3f}, Anderson–Darling A² = "
                f"{diag['anderson_darling_stat']:.3f} (5% crit {diag['anderson_darling_crit_5pct']:.3f})\n")
        f.write(f"- Constant variance: Breusch–Pagan p = {diag['breusch_pagan_p']:.3f}\n")
        f.write(f"- Independence: Durbin–Watson = {diag['durbin_watson']:.2f}\n")
        f.write(f"- Multicollinearity: max VIF = {final_t['VIF'].max():.2f}\n")
        f.write(f"- |std residual| > 2: observations {diag['large_std_resid_obs']}; "
                f"max Cook's D = {diag['max_cooks_d']:.3f} (obs {diag['max_cooks_obs']})\n\n")
        f.write(f"## Out-of-sample (5-fold CV × 50)\n\nCV RMSE = {cv['cv_rmse']:.3f} vs "
                f"{cv_null:.3f} for predicting the mean (on the log scale)\n\n")
        f.write("## Robustness of the enjoyment effect (full 6-predictor model)\n\n" + md(rob, 3) + "\n\n")
        f.write("## Participation: who does no off-screen activity at all?\n\n" + md(part, 3) + "\n\n")
        f.write(f"Among participants only, Spearman ρ(enjoyment, minutes) = {rho.statistic:.2f} "
                f"(p = {rho.pvalue:.2f}).\n")

    print(open(RES / "summary.md").read())


if __name__ == "__main__":
    main()
