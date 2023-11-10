# coding: utf-8
#%%
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import datetime
import copy
import scipy
import stan

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
# stanコードの記述
stan_code = '''
data {
    int T;        // 学習期間
    vector[T] y;  // 観測値
}
parameters {
    vector[T] mu; // 状態の推定値
    real<lower=0> s_v; // システム誤差の標準偏差
    real<lower=0> s_w; // 観測誤差の標準偏差
}
model {
    // 状態方程式
    for (i in 2:T) {
        mu[i] ~ normal(mu[i-1], s_v);
    }
    // 観測方程式
    for (i in 1:T) {
        y[i] ~ normal(mu[i], s_w);
    }
}

'''

#%% モデルのコンパイル
data = {
    "T": len(df_data),
    "y": df_data["y"].values
}

posterior = stan.build(stan_code, data)

# %%
fit = posterior.sample(num_chains=4, num_samples=1000)

# %%
df = fit.to_frame()
print(df.describe().T)

# %%
df