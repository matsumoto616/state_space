# coding: utf-8
#%%
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import datetime
import copy
import scipy
from scipy.optimize import minimize
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

# %%
# 状態空間モデル
# x_t = G_t x_(t-1) + w_t   w_t ~ N(0, W_t)
# y_t = F_t x_t + v_t       v_t ~ N(0, V_t)

def kalman_filter(m, C, y, G, F, W, V):
    """
    Kalman Filter（1時点先の状態の期待値と分散共分散行列を求める）
    m: 時点t-1のフィルタリング分布の平均
    C: 時点t-1のフィルタリング分布の分散共分散行列
    y: 時点tの観測値
    """
    a = G @ m
    R = G @ C @ G.T + W
    f = F @ a
    Q = F @ R @ F.T + V
    # 逆行列と何かの積を取る場合は、invよりsolveを使った方がいいらしい
    K = (np.linalg.solve(Q.T, F @ R.T)).T
    # K = R @ F.T @ np.linalg.inv(Q)
    m = a + K @ (y - f)
    C = R - K @ F @ R
    return m, C

def kalman_smoothing(s, S, m, C, G, W):
    """
    Kalman smoothing
    """
    # 1時点先予測分布のパラメータ計算
    a = G @ m
    R = G @ C @ G.T + W
    # 平滑化利得の計算
    # solveを使った方が約30%速くなる
    A = np.linalg.solve(R.T, G @ C.T).T
    # A = C @ G.T @ np.linalg.inv(R)
    # 状態の更新
    s = m + A @ (s - a)
    S = C + A @ (S - R) @ A.T
    return s, S

def kalman_filter_without_data(m, C, G, W):
    """
    Kalman Predict（データを用いずに1時点先の状態の期待値と分散共分散行列を求める）
    m: 時点t-1のフィルタリング分布の平均
    C: 時点t-1のフィルタリング分布の分散共分散行列
    """
    a = G @ m
    R = G @ C @ G.T + W
    m = a
    C = R
    return m, C

def timeseries_predict(m, C, F, V):
    """
    状態の期待値と分散共分散行列から時系列の予測期待値と分散共分散行列を求める
    m: 時点tのフィルタリング分布の平均
    C: 時点tのフィルタリング分布の分散共分散行列
    """
    y = F @ m
    D = F @ C @ F.T + V
    return y, D

def loglikelihood(yobs, y, D):
    """
    対数尤度（定数を除く）を求める
    """
    l = np.log(np.linalg.det(D))
    l += (yobs-y).T @ np.linalg.inv(D) @ (yobs-y)
    return -0.5 * l

#%% パラメータ推定（最尤法）
data = np.array([[df_data.at[idx, f"y_lag{i}"] for i in range(6)] for idx in df_data.index])
# 状態方程式の行列
G = np.array(
    [[2, -1, 0 ,0 ,0 ,0 ,0 ,0],
     [1, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, -1, -1, -1, -1, -1, -1],
     [0, 0, 1, 0, 0, 0, 0, 0],
     [0, 0, 0, 1, 0, 0, 0, 0],
     [0, 0, 0, 0, 1, 0, 0, 0],
     [0, 0, 0, 0, 0, 1, 0, 0],
     [0, 0, 0, 0, 0, 0, 1, 0]]
)
F = np.array([[1, 0, 1, 0, 0, 0, 0, 0]]) # 観測方程式の行列
T = len(df_data)
Tpred = 7*2

#%%
def objective_function(params):
    W = np.eye(7) * params[0] # 恣意的に与える必要がある
    V = np.eye(7) * params[1] # 上に同じ
    m0 = np.array([params[i] for i in range(2, 9)])
    C0 = np.eye(7) * params[9]

    # 結果を格納するarray
    m = np.zeros((T, 7))
    C = np.zeros((T, 7, 7))
    y = np.zeros((T, 7))
    D = np.zeros((T, 7, 7))

    l = 0 # 対数尤度
    for t in range(T):
        if t == 0:
            m[t], C[t] = kalman_filter(m0, C0, data[t], G, F, W, V)
            y[t], D[t] = timeseries_predict(m[t], C[t], F, V)
            l += loglikelihood(data[t], y[t], D[t])
        else:
            m[t], C[t] = kalman_filter(m[t-1], C[t-1], data[t], G, F, W, V)
            y[t], D[t] = timeseries_predict(m[t], C[t], F, V)
            l += loglikelihood(data[t], y[t], D[t])
    
    return -l # lの最大化＝-lの最小化

result = minimize(
    fun=objective_function,
    x0=[10, 10, 0, 0, 0, 0, 0, 0, 0, 10],
    bounds=[(1e-3, None), (1e-3, None), (0, None), (0, None), (0, None), (0, None), (0, None), (0, None), (0, None), (1e-3, None)], 
    method="L-BFGS-B"
)

# %%
data = np.array([[df_data.at[idx, f"y_lag{i}"] for i in range(7)] for idx in df_data.index])
G = np.zeros((7,7))# 状態方程式の行列
G[0,6] = G[1,0] = G[2,1] = G[3,2] = G[4,3] = G[5,4] = G[6,5] = 1 # weeklymodel
F = np.eye(7) # 観測方程式の行列
T = len(df_data)
Tpred = 7*2

# 最適パラメータを使う
W = np.eye(7) * result.x[0] # 恣意的に与える必要がある
V = np.eye(7) * result.x[1] # 上に同じ
m0 = np.array([result.x[i] for i in range(2, 9)])
C0 = np.eye(7) * result.x[9]

# 結果を格納するarray
m = np.zeros((T, 7))
C = np.zeros((T, 7, 7))
y = np.zeros((T, 7))
D = np.zeros((T, 7, 7))
s = np.zeros((T, 7))
S = np.zeros((T, 7, 7))
mpred = np.zeros((Tpred, 7))
Cpred = np.zeros((Tpred, 7, 7))
ypred = np.zeros((Tpred, 7))
Dpred = np.zeros((Tpred, 7, 7))

# %%
# カルマンフィルタ（と予測）
l = 0 # 対数尤度
for t in range(T):
    if t == 0:
        m[t], C[t] = kalman_filter(m0, C0, data[t], G, F, W, V)
        y[t], D[t] = timeseries_predict(m[t], C[t], F, V)
        l += loglikelihood(data[t], y[t], D[t])
    else:
        m[t], C[t] = kalman_filter(m[t-1], C[t-1], data[t], G, F, W, V)
        y[t], D[t] = timeseries_predict(m[t], C[t], F, V)
        l += loglikelihood(data[t], y[t], D[t])

# カルマン平滑化
for t in range(T):
    t = T - t - 1
    if t == T - 1:
        s[t] = m[t]
        S[t] = C[t]
    else:
        s[t], S[t] = kalman_smoothing(s[t+1], S[t+1], m[t], C[t], G, W)

# 予測
for t in range(Tpred):
    if t == 0:
        mpred[t], Cpred[t] = kalman_filter_without_data(m[-1], C[-1], G, W)
        ypred[t], Dpred[t] = timeseries_predict(mpred[t], Cpred[t], F, V)
    else:
        mpred[t], Cpred[t] = kalman_filter_without_data(mpred[t-1], Cpred[t-1], G, W)
        ypred[t], Dpred[t] = timeseries_predict(mpred[t], Cpred[t], F, V)

# %%
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_data.index, y=df_data["y"], mode='lines', name="data", marker_color="blue")) 
fig.add_trace(go.Scatter(x=df_data.index, y=df_data["yt"], mode='lines', name="true", marker_color="green"))
fig.add_trace(go.Scatter(x=df_data.index, y=m[:, 0], mode='lines', name="xfilter", marker_color="orange")) 
# fig.add_trace(go.Scatter(x=df_data.index, y=s[:, 0], mode='lines', name="smoother")) 

df_pred = pd.DataFrame(index=pd.date_range(start=end_date, periods=Tpred+1)).iloc[1:]
df_pred["weekday"] = df_pred.index.map(lambda x: x.weekday())
df_pred["yt"] = df_pred["weekday"].map(lambda x: weekday_true[x])
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
fig.add_trace(go.Scatter(x=df_pred.index, y=df_pred["yt"], mode='lines', name="true", marker_color="green"))
fig.add_trace(go.Scatter(x=df_pred.index, y=mpred[:, 0], mode='lines', name="xpred", marker_color="orange")) 
upper = [scipy.stats.norm.ppf(0.975, mpred[t,0], Cpred[t,0,0]) for t in range(Tpred)]
lower = [scipy.stats.norm.ppf(0.025, mpred[t,0], Cpred[t,0,0]) for t in range(Tpred)]
fig.add_trace(go.Scatter(x=df_pred.index, y=lower, mode='lines', name="xpred_lower", marker_color="orange"))
fig.add_trace(go.Scatter(x=df_pred.index, y=upper, mode='lines', fill="tonexty", name="xpred_upper", marker_color="orange")) 

fig.add_trace(go.Scatter(x=df_pred.index, y=ypred[:, 0], mode='lines', name="ypred", marker_color="red")) 
upper = [scipy.stats.norm.ppf(0.975, ypred[t,0], Dpred[t,0,0]) for t in range(Tpred)]
lower = [scipy.stats.norm.ppf(0.025, ypred[t,0], Dpred[t,0,0]) for t in range(Tpred)]
fig.add_trace(go.Scatter(x=df_pred.index, y=lower, mode='lines', name="ypred_lower", marker_color="red"))
fig.add_trace(go.Scatter(x=df_pred.index, y=upper, mode='lines', fill="tonexty", name="ypred_upper", marker_color="red")) 

upper = [scipy.stats.norm.ppf(0.975, m[t,0], C[t,0,0]) for t in range(T)]
lower = [scipy.stats.norm.ppf(0.025, m[t,0], C[t,0,0]) for t in range(T)]
fig.add_trace(go.Scatter(x=df_data.index, y=lower, mode='lines', name="xfilter_lower", marker_color="red"))
fig.add_trace(go.Scatter(x=df_data.index, y=upper, mode='lines', fill="tonexty", name="xfilter_upper", marker_color="red")) 
upper = [scipy.stats.norm.ppf(0.975, y[t,0], D[t,0,0]) for t in range(T)]
lower = [scipy.stats.norm.ppf(0.025, y[t,0], D[t,0,0]) for t in range(T)]
fig.add_trace(go.Scatter(x=df_data.index, y=lower, mode='lines', name="yfilter_lower", marker_color="red"))
fig.add_trace(go.Scatter(x=df_data.index, y=upper, mode='lines', fill="tonexty", name="yfilter_upper", marker_color="red"))

fig.update_xaxes(rangeslider_visible=True)
fig.update_layout(
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=7,
                     label="1w",
                     step="day",
                     stepmode="backward"),
                dict(count=1,
                     label="1m",
                     step="month",
                     stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(
            visible=True
        ),
        type="date"
    )
)

fig.show()

# %%
