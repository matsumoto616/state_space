# coding: utf-8
#%%
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import datetime
import copy
import scipy

#%%
weekday_true = [0, 100, 0, 10, 50, 100, 0]
weekday_range = [0, 20, 0, 5, 10, 20, 0]
weekday_peak_prob = [0.1, 0, 0, 0, 0, 0, 0.1]
weekday_peak_value = [20, 0, 0, 0, 0, 0, 20]
weekday_missing_prob = [0, 0.1, 0, 0, 0, 0.1, 0]
np.random.seed(0)

#%%
def make_random_data(
        weekday,
        weekday_true,
        weekday_range,
        weekday_peak_prob,
        weekday_peak_value,
        weekday_missing_prob
    ):
    
    value = weekday_true[weekday]
    value += np.random.randint(-weekday_range[weekday], weekday_range[weekday]+1)
    if np.random.rand() < weekday_peak_prob[weekday]:
        value += weekday_peak_value[weekday]
    if np.random.rand() < weekday_missing_prob[weekday]:
        value = 0

    return value

# %%
start_date = "2020/01/01"
end_date = "2020/06/30"

df_data = pd.DataFrame(index=pd.date_range(start=start_date, end=end_date))
df_data["weekday"] = df_data.index.map(lambda x: x.weekday())
df_data["yt"] = df_data["weekday"].map(lambda x: weekday_true[x])
df_data["y"] = df_data["weekday"].map(
    lambda x: make_random_data(
        x,
        weekday_true,
        weekday_range,
        weekday_peak_prob,
        weekday_peak_value,
        weekday_missing_prob
    )
)
for i in range(7):
    df_data[f"y_lag{(i)}"] = df_data["y"].shift(i)

df_data = df_data.fillna(0)

#%%
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_data.index, y=df_data["y"])) 

#%%
RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)

# %%
# True parameter values
alpha, sigma = 1, 1
beta = [1, 2.5]

# Size of dataset
size = 100

# Predictor variable
X1 = np.random.randn(size)
X2 = np.random.randn(size) * 0.2

# Simulate outcome variable
Y = alpha + beta[0] * X1 + beta[1] * X2 + rng.normal(size=size) * sigma

# %%
fig = go.Figure()
fig.add_trace(go.Scatter(x=X1, y=Y, mode="markers"))

# %%
import pymc as pm
import arviz as az
import plotly.graph_objects as go
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

print(f"Running on PyMC v{pm.__version__}")

RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)
az.style.use("arviz-darkgrid")
%config InlineBackend.figure_format = 'retina'

# %%
t = np.array(range(len(df_data)))
tmin = np.min(t)
tmax = np.max(t)
t = (t-tmin) / (tmax-tmin)

y = df_data["y"].values
ymax = np.max(y)
y = y / ymax

with pm.Model(check_bounds=False) as linear:
    α = pm.Normal("α", mu=0, sigma=0.5) # 切片
    β = pm.Normal("β", mu=0, sigma=0.5) # 傾き
    σ = pm.HalfNormal("σ", sigma=0.5)
    trend = pm.Deterministic("trend", α + β * t)
    pm.Normal("likelihood", mu=trend, sigma=σ, observed=y)

    linear_prior = pm.sample_prior_predictive()

#%%
fig = go.Figure()

y_likelihood = az.extract(linear_prior, group="prior_predictive", num_samples=100)["likelihood"]*ymax
y_likelihood = y_likelihood.values.T
y_trend = az.extract(linear_prior, group="prior", num_samples=100)["trend"]*ymax
y_trend = y_trend.values.T
for i in range(len(y_trend)):
    fig.add_trace(
        go.Scatter(
            x=df_data.index,
            y=y_trend[i]
        )
    )
fig.add_trace(
    go.Scatter(
        x=df_data.index,
        y=df_data["y"],
        mode="lines",
        marker_color="red"
    )
)

#%%
with linear:
    linear_trace = pm.sample(return_inferencedata=True)
    linear_prior = pm.sample_posterior_predictive(trace=linear_trace)

# %%
ytrend = az.extract_dataset(linear_trace, group="posterior", num_samples=100)["trend"].values.T*ymax

fig = go.Figure()
for i in range(len(ytrend)):
    fig.add_trace(
        go.Scatter(
            x=df_data.index,
            y=ytrend[i]
        )
    )
fig.add_trace(
    go.Scatter(
        x=df_data.index,
        y=df_data["y"],
        mode="lines",
        marker_color="red"
    )
)

# %%
n_order = 10
periods = np.array(range(len(df_data))) / 7

fourier_features = pd.DataFrame(
    {
        f"{func}_order_{order}": getattr(np, func)(2 * np.pi * periods * order)
        for order in range(1, n_order + 1)
        for func in ("sin", "cos")
    }
)
fourier_features

# %%
coords = {"fourier_features": np.arange(2 * n_order)}
with pm.Model(check_bounds=False, coords=coords) as linear_with_seasonality:
    a = pm.Normal("a", mu=0, sigma=0.5)
    b = pm.Normal("b", mu=0, sigma=0.5)

    β_fourier = pm.Normal("β_fourier", mu=0, sigma=0.1, dims="fourier_features")
    seasonality = pm.Deterministic(
        "seasonality", pm.math.dot(β_fourier, fourier_features.to_numpy().T)
    )

    λ = pm.math.exp(a + b * t + seasonality)
    α = pm.HalfNormal("α", 0.5)
    pm.NegativeBinomial("likelihood", mu=λ, alpha=α, observed=y)

    linear_seasonality_prior = pm.sample_prior_predictive()

#%%
with linear_with_seasonality:
    linear_seasonality_trace = pm.sample(return_inferencedata=True)
    linear_seasonality_posterior = pm.sample_posterior_predictive(trace=linear_seasonality_trace)
    
#%%
y_likelihood = az.extract_dataset(
    linear_seasonality_posterior, group="posterior_predictive", num_samples=100)["likelihood"].values.T*ymax
# y_trend = az.extract_dataset(linear_seasonality_trace, group="posterior", num_samples=100)["trend"].values.T*ymax
# y_seasonal = az.extract_dataset(linear_seasonality_trace, group="posterior", num_samples=100)["seasonality"].values.T*ymax
fig = go.Figure()
# for i in range(len(y_trend)):
#     fig.add_trace(
#         go.Scatter(
#             x=df_data.index,
#             y=y_likelihood[i]
#         )
#     )
fig.add_trace(
    go.Scatter(
        x=df_data.index,
        y=df_data["y"],
        mode="lines",
        marker_color="blue"
    )
)
fig.add_trace(
    go.Scatter(
        x=df_data.index,
        y=[np.mean(s) for s in y_likelihood.T],
        mode="lines",
        marker_color="red"
    )
)
 # %%
