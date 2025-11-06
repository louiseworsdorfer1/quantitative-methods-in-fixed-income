from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CSV = DATA / "LW_monthly_1972-2024.csv"
EXCEL = DATA / "LW_monthly_1972-2024.xlsx"

df = pd.read_excel(EXCEL, sep = ';', decimal=",", skiprows = 1)
df = df.iloc[:, :122]

def getTimeSeriesMean(j):
    """
    Returns the time series mean for yield j (1 <= j <= 120)
    """
    # Load the data
    data = df
    
    # Ensure j is within valid range
    if j < 1 or j > 120:
        raise ValueError("Index j must be between 1 and 120.")
    
    # Select column j (adjusting for 0-based indexing)
    yield_series = data.iloc[:, j-1]
    
    # Compute and return the mean
    return yield_series.mean()

def _ols_lstsq(X, y):
    """
    OLS via least squares: returns beta (k,)
    X: (n,k), y: (n,)
    Negeert NaNs (in zowel X als y).
    """
    # Mask NaNs
    mask = np.isfinite(y)
    if mask.sum() < X.shape[1]:
        raise ValueError("Te weinig niet-NaN observaties voor OLS.")
    Xm = X[mask, :]
    ym = y[mask]
    beta, *_ = np.linalg.lstsq(Xm, ym, rcond=None)
    return beta


def _fit_ar1_and_forecast(x, h):
    """
    Fit AR(1): x_t = c + phi * x_{t-1} + e_t  (t=1..T-1)
    Returns h-step-ahead forecast from last observed x_T.
    """
    x = np.asarray(x, dtype=float)
    if np.sum(np.isfinite(x)) < 3:
        # te weinig data -> simpele fallback: hou constant
        return float(x[-1])

    # Maak lag en contemp
    x_lag = x[:-1]
    x_now = x[1:]

    # Drop NaNs
    mask = np.isfinite(x_lag) & np.isfinite(x_now)
    X = np.column_stack([np.ones(mask.sum()), x_lag[mask]])
    y = x_now[mask]

    if X.shape[0] < 3:
        return float(x[-1])

    # OLS
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    c, phi = b[0], b[1]
    x_T = x[np.where(np.isfinite(x))[0][-1]]  # laatste geldige

    # h-step-ahead forecast
    # x_{T+h|T} = phi^h * x_T + c * (1 - phi^h)/(1 - phi)  (voor phi != 1)
    if np.isclose(phi, 1.0):
        x_fore = x_T + h * c
    else:
        x_fore = (phi ** h) * x_T + c * (1 - (phi ** h)) / (1 - phi)
    return float(x_fore)


def  getNelsonSiegelForecast(i, h, j):

    """
    Returns the h-step-ahead Nelson–Siegel yield forecast for maturity index j (1..120),
    using subsample t = 1..i to estimate beta time series and AR(1) dynamics.

    Arguments:
      i : int, last row index of subsample (1-based in uitleg; hier gewoon als aantal rijen)
      h : int, forecast horizon in months (1..24)
      j : int, maturity index (1..120)  --> maturity in months
    """
    if j < 1 or j > 120:
        raise ValueError("j moet tussen 1 en 120 liggen.")
    if h < 1 or h > 24:
        raise ValueError("h moet tussen 1 en 24 liggen.")

    T = df.shape[0]
    if i < 12 or i > T:
        raise ValueError(f"i moet tussen 12 en {T} liggen.")

    # ---------- Nelson–Siegel loadings (tau in maanden, lambda vast) ----------
    lam = 0.0609  # vaste lambda; maturities in maanden
    taus = np.arange(1, 121, dtype=float)  # 1..120 maanden

    # Basisfuncties:
    # f1(τ)=1
    # f2(τ)=(1 - exp(-λτ))/(λτ)
    # f3(τ)=f2(τ) - exp(-λτ)
    f1 = np.ones_like(taus)
    exp_term = np.exp(-lam * taus)
    f2 = (1.0 - exp_term) / (lam * taus)
    f3 = f2 - exp_term

    X = np.column_stack([f1, f2, f3])  # (120,3), constant across t

    # ---------- Cross-sectionele OLS per tijd t (t=0..i-1) ----------
    betas = np.zeros((i, 3))
    Y = df.iloc[:i, :120].to_numpy(dtype=float)

    for t in range(i):
        y_t = Y[t, :]
        beta_t = _ols_lstsq(X, y_t)  # (3,)
        betas[t, :] = beta_t

    # ---------- AR(1) per beta-reeks + h-step forecast ----------
    beta1_fore = _fit_ar1_and_forecast(betas[:, 0], h)
    beta2_fore = _fit_ar1_and_forecast(betas[:, 1], h)
    beta3_fore = _fit_ar1_and_forecast(betas[:, 2], h)

    # ---------- Yield-forecast voor maturity j ----------
    tau_j = float(j)  # j maanden
    exp_j = np.exp(-lam * tau_j)
    f2_j = (1.0 - exp_j) / (lam * tau_j)
    f3_j = f2_j - exp_j

    y_fore_j = beta1_fore + beta2_fore * f2_j + beta3_fore * f3_j
    return float(y_fore_j)

print(getNelsonSiegelForecast(17,4,5))