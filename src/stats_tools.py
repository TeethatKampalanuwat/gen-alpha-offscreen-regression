"""
Small, dependency-light regression toolkit (numpy + scipy only).

Implements what Minitab's "Fit Regression Model" + "Stepwise > Backward elimination"
produce, so every number in the report can be reproduced and audited line by line.
"""
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class OLSResult:
    terms: list
    coef: np.ndarray
    se: np.ndarray
    t: np.ndarray
    p: np.ndarray
    resid: np.ndarray
    fitted: np.ndarray
    s: float
    r2: float
    r2_adj: float
    r2_pred: float
    f_stat: float
    f_p: float
    aicc: float
    bic: float
    mallows_cp: float | None
    hat: np.ndarray
    n: int
    k: int  # number of predictors (excl. constant)

    def table(self, vif: dict | None = None) -> pd.DataFrame:
        df = pd.DataFrame({"Coef": self.coef, "SE Coef": self.se,
                           "T-Value": self.t, "P-Value": self.p}, index=self.terms)
        if vif is not None:
            df["VIF"] = [np.nan] + [vif[t] for t in self.terms[1:]]
        return df


def ols(df: pd.DataFrame, y: str, xs: list, full_mse: float | None = None) -> OLSResult:
    Y = df[y].to_numpy(float)
    X = np.column_stack([np.ones(len(df))] + [df[c].to_numpy(float) for c in xs])
    n, p = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ Y
    fitted = X @ beta
    resid = Y - fitted
    sse = resid @ resid
    sst = ((Y - Y.mean()) ** 2).sum()
    mse = sse / (n - p)
    se = np.sqrt(np.diag(XtX_inv) * mse)
    t = beta / se
    pval = 2 * stats.t.sf(np.abs(t), n - p)
    hat = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    with np.errstate(divide="ignore", invalid="ignore"):
        press = ((resid / (1 - hat)) ** 2).sum()
    k = p - 1
    f = ((sst - sse) / k) / mse if k else np.nan
    # Minitab's AICc/BIC convention (log-likelihood with p + 1 parameters incl. sigma)
    loglik = -n / 2 * (np.log(2 * np.pi) + np.log(sse / n) + 1)
    q = p + 1
    aic = -2 * loglik + 2 * q
    aicc = aic + 2 * q * (q + 1) / (n - q - 1)
    bic = -2 * loglik + q * np.log(n)
    cp = sse / full_mse - (n - 2 * p) if full_mse else None
    return OLSResult(["Constant"] + list(xs), beta, se, t, pval, resid, fitted,
                     np.sqrt(mse), 1 - sse / sst, 1 - mse / (sst / (n - 1)),
                     1 - press / sst, f, stats.f.sf(f, k, n - p) if k else np.nan,
                     aicc, bic, cp, hat, n, k)


def vif(df: pd.DataFrame, xs: list) -> dict:
    out = {}
    for c in xs:
        others = [o for o in xs if o != c]
        r2 = ols(df, c, others).r2 if others else 0.0
        out[c] = 1 / (1 - r2)
    return out


def backward_elimination(df, y, xs, alpha_remove=0.10):
    """Minitab-style backward elimination. Returns list of fitted steps."""
    full = ols(df, y, xs)
    full_mse = full.s ** 2
    current = list(xs)
    steps = []
    while True:
        res = ols(df, y, current, full_mse=full_mse)
        steps.append(res)
        pvals = dict(zip(res.terms[1:], res.p[1:]))
        worst = max(pvals, key=pvals.get) if pvals else None
        if worst is None or pvals[worst] <= alpha_remove:
            return steps
        current.remove(worst)


def standardized_coefs(df, y, res: OLSResult) -> pd.Series:
    sy = df[y].std()
    return pd.Series({t: b * df[t].std() / sy
                      for t, b in zip(res.terms[1:], res.coef[1:])})


def pearson_matrix(df: pd.DataFrame):
    cols = df.columns
    r = df.corr()
    p = pd.DataFrame(np.ones((len(cols), len(cols))), index=cols, columns=cols)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            p.loc[a, b] = p.loc[b, a] = stats.pearsonr(df[a], df[b]).pvalue
    return r, p


def diagnostics(res: OLSResult, df, xs) -> dict:
    e = res.resid
    n, p = res.n, res.k + 1
    std_resid = e / (res.s * np.sqrt(1 - res.hat))
    cooks = std_resid ** 2 / p * res.hat / (1 - res.hat)
    # Breusch-Pagan (Koenker studentized form): n * R^2 of e^2 on X
    aux = df[xs].copy()
    aux["_e2"] = e ** 2
    bp_r2 = ols(aux, "_e2", xs).r2
    bp = n * bp_r2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        ad = stats.anderson(e, dist="norm")
    return {
        "shapiro_wilk_p": stats.shapiro(e).pvalue,
        "anderson_darling_stat": ad.statistic,
        "anderson_darling_crit_5pct": ad.critical_values[2],
        "breusch_pagan_stat": bp,
        "breusch_pagan_p": stats.chi2.sf(bp, len(xs)),
        "durbin_watson": np.sum(np.diff(e) ** 2) / np.sum(e ** 2),
        "large_std_resid_obs": [int(i) + 1 for i in np.where(np.abs(std_resid) > 2)[0]],
        "max_cooks_d": float(cooks.max()),
        "max_cooks_obs": int(cooks.argmax()) + 1,
        "std_resid": std_resid,
        "cooks": cooks,
    }


def kfold_cv(df, y, xs, k=10, repeats=20, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    rmses, maes = [], []
    for _ in range(repeats):
        folds = np.array_split(rng.permutation(len(df)), k)
        err = np.empty(len(df))
        for f in folds:
            train = df.drop(df.index[f])
            r = ols(train, y, xs)
            Xt = np.column_stack([np.ones(len(f))] + [df.iloc[f][c] for c in xs])
            err[f] = df.iloc[f][y].to_numpy() - Xt @ r.coef
        rmses.append(np.sqrt(np.mean(err ** 2)))
        maes.append(np.mean(np.abs(err)))
    return {"cv_rmse": float(np.mean(rmses)), "cv_mae": float(np.mean(maes))}
