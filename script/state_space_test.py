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

# %%
def kalman_filter(m, C, y, G, F, W, V):
    """
    Kalman Filter
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

def kalman_predict(m, C, G, W):
    """
    Kalman Predict
    m: 時点t-1のフィルタリング分布の平均
    C: 時点t-1のフィルタリング分布の分散共分散行列
    """
    a = G @ m
    R = G @ C @ G.T + W
    m = a
    C = R
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

# %%
data = np.array([[df_data.at[idx, f"y_lag{i}"] for i in range(7)] for idx in df_data.index])
G = np.zeros((7,7))# 状態方程式の行列
G[0,6] = G[1,0] = G[2,1] = G[3,2] = G[4,3] = G[5,4] = G[6,5] = 1 # weeklymodel
F = np.eye(7) # 観測方程式の行列
W = np.eye(7) * 10 # 恣意的に与える必要がある
V = np.eye(7) * 10 # 上に同じ
T = len(df_data)
Tpred = 7*2

m0 = np.zeros(7)
C0 = np.eye(7) * 100

# 結果を格納するarray
m = np.zeros((T, 7))
C = np.zeros((T, 7, 7))
s = np.zeros((T, 7))
S = np.zeros((T, 7, 7))
mpred = np.zeros((Tpred, 7))
Cpred = np.zeros((Tpred, 7, 7))

# %%
# カルマンフィルタ
for t in range(T):
    if t == 0:
        m[t], C[t] = kalman_filter(m0, C0, data[t], G, F, W, V)
    else:
        m[t], C[t] = kalman_filter(m[t-1], C[t-1], data[t], G, F, W, V)

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
        mpred[t], Cpred[t] = kalman_predict(m[-1], C[-1], G, W)
    else:
        mpred[t], Cpred[t] = kalman_predict(mpred[t-1], Cpred[t-1], G, W)

# %%
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_data.index, y=df_data["y"], mode='lines', name="data")) 
# fig.add_trace(go.Scatter(x=df_data.index, y=df_data["yt"], mode='lines', name="true"))
fig.add_trace(go.Scatter(x=df_data.index, y=m[:, 0], mode='lines', name="filter", marker_color="red")) 
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
fig.add_trace(go.Scatter(x=df_pred.index, y=df_pred["yt"], mode='lines', name="true"))
fig.add_trace(go.Scatter(x=df_pred.index, y=mpred[:, 0], mode='lines', name="pred", marker_color="red")) 
upper = [scipy.stats.norm.ppf(0.975, mpred[t,0], Cpred[t,0,0]) for t in range(Tpred)]
lower = [scipy.stats.norm.ppf(0.025, mpred[t,0], Cpred[t,0,0]) for t in range(Tpred)]
fig.add_trace(go.Scatter(x=df_pred.index, y=lower, mode='lines', name="pred_lower", marker_color="pink"))
fig.add_trace(go.Scatter(x=df_pred.index, y=upper, mode='lines', fill="tonexty", name="pred_upper", marker_color="pink")) 

# 信頼区間
upper = [scipy.stats.norm.ppf(0.975, m[t,0], C[t,0,0]) for t in range(T)]
lower = [scipy.stats.norm.ppf(0.025, m[t,0], C[t,0,0]) for t in range(T)]
fig.add_trace(go.Scatter(x=df_data.index, y=lower, mode='lines', name="filter_lower", marker_color="pink"))
fig.add_trace(go.Scatter(x=df_data.index, y=upper, mode='lines', fill="tonexty", name="filter_upper", marker_color="pink")) 

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
