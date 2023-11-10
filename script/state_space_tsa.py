# coding: utf-8
#%%
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

# weekday_true = [0, 100, 0, 10, 50, 100, 0]
# weekday_range = [0, 0, 0, 0, 0, 0, 0]
# weekday_peak_prob = [0, 0, 0, 0, 0, 0, 0]
# weekday_peak_value = [20, 0, 0, 0, 0, 0, 20]
# weekday_missing_prob = [0, 0, 0, 0, 0, 0, 0]
# np.random.seed(0)

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
end_date = "2020/12/31"

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
# x_t = F_t x_(t-1) + Gv_t   v_t ~ N(0, Q_t)
# y_t = H_t x_t + w_t       w_t ~ N(0, R_t)

def kalman_filter(x, V, y, F, G, H, Q, R):
    """
    Kalman Filter（1時点先の状態の期待値と分散共分散行列を求める）
    x: 時点t-1のフィルタリング分布の平均
    V: 時点t-1のフィルタリング分布の分散共分散行列
    y: 時点tの観測値
    """
    # 単位行列
    I = np.identity(F.shape[0])

    # 1期先予測
    x_next = F @ x
    V_next = F @ V @ F.T + G @ Q @ G.T

    # フィルタ（逆行列計算の高速化はTODO）
    K = V_next @ H.T @ np.linalg.inv(H @ V_next @ H.T + R)
    x_filter = x_next + K @ (y - H @ x_next)
    V_filter = (I - K @ H) @ V_next

    return x_filter, V_filter, x_next, V_next # nextは尤度計算用

def kalman_filter_without_data(x, V, F, G, Q):
    """
    Kalman Predict（データを用いずに1時点先の状態の期待値と分散共分散行列を求める）
    x: 時点t-1のフィルタリング分布の平均
    V: 時点t-1のフィルタリング分布の分散共分散行列
    """
    # 1期先予測
    x_next = F @ x
    V_next = F @ V @ F.T + G @ Q @ G.T

    return x_next, V_next

def timeseries_predict(x, V, H, R):
    """
    状態の期待値と分散共分散行列から時系列の予測期待値と分散共分散行列を求める
    x: 時点tのフィルタリング分布の平均
    V: 時点tのフィルタリング分布の分散共分散行列
    """
    y = H @ x
    D = H @ V @ H.T + R

    return y, D

def loglikelihood(yobs, y, D):
    """
    対数尤度（定数を除く）を求める
    """
    l = np.log(np.linalg.det(D))
    l += (yobs-y).T @ np.linalg.inv(D) @ (yobs-y)
    return -0.5 * l

def loglikelihood_fast(yobs, y, D):
    """
    高速化版の対数尤度（定数を除く）を求める（dim_l=1, R=σ^2のとき）
    """
    l = np.log(D)
    sigma2 = (y - yobs)**2 / D
    return -0.5 * l, sigma2

def make_diag_stack_matrix(matrix_list):
    """
    行列のリストから対角方向に結合した行列を作成する
    """
    dim_i = sum([m.shape[0] for m in matrix_list])
    dim_j = sum([m.shape[1] for m in matrix_list])
    block_diag = np.zeros((dim_i, dim_j))

    pos_i = pos_j = 0
    for m in matrix_list:
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                block_diag[pos_i+i, pos_j+j] = m[i, j]
        pos_i += m.shape[0]
        pos_j += m.shape[1]

    return block_diag

def make_hstack_matrix(matrix_list):
    """
    行列のリストから横方向に結合した行列を作成する
    """
    return np.concatenate(matrix_list, 1)

#%% パラメータ推定（最尤法）
data = np.array([[[df_data.at[idx, "y"]]] for idx in df_data.index])
T = len(df_data)

# 状態方程式の行列
F_trend = np.array(
    [[2, -1],
     [1,  0]]
)
F_seasonal = np.array(
    [[-1, -1, -1, -1, -1, -1],
     [ 1,  0,  0,  0,  0,  0],
     [ 0,  1,  0,  0,  0,  0],
     [ 0,  0,  1,  0,  0,  0],
     [ 0,  0,  0,  1,  0,  0],
     [ 0,  0,  0,  0,  1,  0]]
)
F_cycle = np.array( # 仮
    [[1,  1],
     [1,  0]]
)
F = make_diag_stack_matrix([F_trend, F_seasonal, F_cycle])

# システムノイズの行列
G_trend = np.array(
    [[1],
     [0]]
)
G_seasonal = np.array(
    [[1],
     [0],
     [0],
     [0],
     [0],
     [0]]
)
G_cycle = np.array(
    [[1],
     [0]]
)
G = make_diag_stack_matrix([G_trend, G_seasonal, G_cycle])

# 観測方程式の行列
H_trend = np.array([[1, 0]])
H_seasonal = np.array([[1, 0, 0, 0, 0, 0]])
H_cycle = np.array([[1, 0]])
H = make_hstack_matrix([H_trend, H_seasonal, H_cycle])

# 次数
dim_k = F.shape[0]
dim_m = G.shape[0]
dim_l = H.shape[0]

#%%
def objective_function(params):
    # システムノイズの分散共分散行列
    Q = np.array(
        [[params[0], 0, 0],
        [0, params[1], 0],
        [0, 0, params[2]]]
    )

    # 観測ノイズの分散共分散行列
    R =np.array([[params[3]]])

    # AR効果を含む状態モデルの行列
    F_cycle = np.array( # 仮
        [[params[4],  params[5]],
        [1,  0]]
    )
    F = make_diag_stack_matrix([F_trend, F_seasonal, F_cycle])

    # 初期状態
    x0 = np.array([[0 for _ in range(dim_k)]]).T
    V0 = np.eye(dim_k) * 1

    # 結果を格納するarray
    x = np.zeros((T, dim_k, dim_l))
    V = np.zeros((T, dim_k, dim_k))
    y = np.zeros((T, dim_l, dim_l))
    D = np.zeros((T, dim_l, dim_l))

    l = -0.5 * dim_l * T * np.log(2*np.pi) # 対数尤度
    for t in range(T):
        if t == 0:
            x[t], V[t], xforl, Vforl = kalman_filter(x0, V0, data[t], F, G, H, Q, R)
            y[t], D[t] = timeseries_predict(x[t], V[t], H, R)
            yforl, Dforl = timeseries_predict(xforl, Vforl, H, R)
            l += loglikelihood(data[t], yforl, Dforl)
        else:
            x[t], V[t], xforl, Vforl = kalman_filter(x[t-1], V[t-1], data[t], F, G, H, Q, R)
            y[t], D[t] = timeseries_predict(x[t], V[t], H, R)
            yforl, Dforl = timeseries_predict(xforl, Vforl, H, R)
            l += loglikelihood(data[t], yforl, Dforl)
    
    return -l # lの最大化＝-lの最小化

# def objective_function_fast(params, get_sigma2):
#     # dim_l = 1, R = σ^2（不変）の場合の高速化
#     # システムノイズの分散共分散行列
#     Q = np.array(
#         [[params[0], 0],
#         [0, params[1]]]
#     )

#     # 観測ノイズの分散共分散行列（1とおく）
#     R =np.array([[1]])

#     # 初期状態
#     x0 = np.array([[0 for i in range(dim_k)]]).T
#     V0 = np.eye(dim_k) * 1

#     # 結果を格納するarray
#     x = np.zeros((T, dim_k, dim_l))
#     V = np.zeros((T, dim_k, dim_k))
#     y = np.zeros((T, dim_l, dim_l))
#     D = np.zeros((T, dim_l, dim_l))

#     l = -0.5 * T # 対数尤度
#     sigma2 = 0
#     for t in range(T):
#         if t == 0:
#             x[t], V[t], xforl, Vforl = kalman_filter(x0, V0, data[t], F, G, H, Q, R)
#             y[t], D[t] = timeseries_predict(x[t], V[t], H, R)
#             yforl, Dforl = timeseries_predict(xforl, Vforl, H, R)
#             dl, dsigma2 = loglikelihood_fast(data[t], yforl, Dforl)
#             l += dl
#             sigma2 += dsigma2[0, 0]
#         else:
#             x[t], V[t], xforl, Vforl = kalman_filter(x[t-1], V[t-1], data[t], F, G, H, Q, R)
#             y[t], D[t] = timeseries_predict(x[t], V[t], H, R)
#             yforl, Dforl = timeseries_predict(xforl, Vforl, H, R)
#             dl, dsigma2 = loglikelihood_fast(data[t], yforl, Dforl)
#             l += dl
#             sigma2 += dsigma2[0, 0]

#     sigma2 /= T
#     l += -0.5*T*np.log(2*np.pi*sigma2)
    
#     if get_sigma2:
#         return sigma2
#     else:
#         return -l # lの最大化＝-lの最小化

#%%
result = minimize(
    fun=objective_function,
    x0=[100, 100, 100, 100, 0, 0],
    bounds=[(1e-5, None), (1e-5, None), (1e-5, None), (1e-5, None), (None, None), (None, None)], 
    method="L-BFGS-B"
)

#%% なぜかうまくいかない。。。
# result = minimize(
#     fun=lambda x: objective_function_fast(x, False),
#     x0=[100, 100],
#     bounds=[(1e-5, None), (1e-5, None)], 
#     method="L-BFGS-B"
# )

# sigma2 = objective_function_fast(result.x, True)
# result.x = list(result.x)
# result.x.append(sigma2)
# result.x[0] *= sigma2
# result.x[1] *= sigma2

# %%
Tpred = 7*2

# 最適パラメータを使う
Q = np.array(
    [[result.x[0], 0, 0],
    [0, result.x[1], 0],
    [0, 0, result.x[2]]]
)

# 観測ノイズの分散共分散行列
R =np.array([[result.x[3]]])

# AR効果を含む状態モデルの行列
F_cycle = np.array( # 仮
    [[result.x[4],  result.x[5]],
    [1,  0]]
)
F = make_diag_stack_matrix([F_trend, F_seasonal, F_cycle])
x0 = np.array([[0 for i in range(dim_k)]]).T
V0 = np.eye(dim_k) * 1

# 結果を格納するarray
x = np.zeros((T, dim_k, dim_l))
V = np.zeros((T, dim_k, dim_k))
y = np.zeros((T, dim_l, dim_l))
D = np.zeros((T, dim_l, dim_l))
xpred = np.zeros((Tpred, dim_k, dim_l))
Vpred = np.zeros((Tpred, dim_k, dim_k))
ypred = np.zeros((Tpred, dim_l, dim_l))
Dpred = np.zeros((Tpred, dim_l, dim_l))

# %%
# カルマンフィルタ（と予測）
for t in range(T):
    if t == 0:
        x[t], V[t], _, _ = kalman_filter(x0, V0, data[t], F, G, H, Q, R)
        y[t], D[t] = timeseries_predict(x[t], V[t], H, R)
    else:
        x[t], V[t], _, _ = kalman_filter(x[t-1], V[t-1], data[t], F, G, H, Q, R)
        y[t], D[t] = timeseries_predict(x[t], V[t], H, R)

# 予測
for t in range(Tpred):
    if t == 0:
        xpred[t], Vpred[t] = kalman_filter_without_data(x[-1], V[-1], F, G, Q)
        ypred[t], Dpred[t] = timeseries_predict(xpred[t], Vpred[t], H, R)
    else:
        xpred[t], Vpred[t] = kalman_filter_without_data(xpred[t-1], Vpred[t-1], F, G, Q)
        ypred[t], Dpred[t] = timeseries_predict(xpred[t], Vpred[t], H, R)

# %%
# 学習期間のプロット
fig = make_subplots(rows=5, cols=1)

# 全体
fig.add_trace(
    go.Scatter(x=df_data.index, y=df_data["y"], mode='lines', name="data", marker_color="blue"),
    row=1, col=1
) 
fig.add_trace(
    go.Scatter(x=df_data.index, y=df_data["yt"], mode='lines', name="true", marker_color="green"),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=df_data.index, y=x[:,0,0]+x[:,2,0]+x[:,8,0], mode='lines', name="trend+seasonal", marker_color="red"),
    row=1, col=1     
) 

# トレンド
trend_upper = [scipy.stats.norm.ppf(0.975, x[t,0,0], np.sqrt(V[t,0,0])) for t in range(T)]
trend_lower = [scipy.stats.norm.ppf(0.025, x[t,0,0], np.sqrt(V[t,0,0])) for t in range(T)]
fig.add_trace(
    go.Scatter(x=df_data.index, y=trend_lower, mode='lines', name="trend_lower", marker_color="pink"),
    row=2, col=1     
)
fig.add_trace(
    go.Scatter(x=df_data.index, y=trend_upper, mode='lines', name="trend_upper", fill="tonexty", marker_color="pink"),
    row=2, col=1     
)
fig.add_trace(
    go.Scatter(x=df_data.index, y=x[:,0,0], mode='lines', name="trend", marker_color="red"),
    row=2, col=1     
)

# 季節
seasonal_upper = [scipy.stats.norm.ppf(0.975, x[t,2,0], np.sqrt(V[t,2,2])) for t in range(T)]
seasonal_lower = [scipy.stats.norm.ppf(0.025, x[t,2,0], np.sqrt(V[t,2,2])) for t in range(T)]
fig.add_trace(
    go.Scatter(x=df_data.index, y=seasonal_lower, mode='lines', name="seasonal_lower", marker_color="pink"),
    row=3, col=1     
)
fig.add_trace(
    go.Scatter(x=df_data.index, y=seasonal_upper, mode='lines', name="seasonal_upper", fill="tonexty", marker_color="pink"),
    row=3, col=1     
)
fig.add_trace(
    go.Scatter(x=df_data.index, y=x[:,2,0], mode='lines', name="seasonal", marker_color="red"),
    row=3, col=1     
)

# サイクル
cycle_upper = [scipy.stats.norm.ppf(0.975, x[t,8,0], np.sqrt(V[t,8,8])) for t in range(T)]
cycle_lower = [scipy.stats.norm.ppf(0.025, x[t,8,0], np.sqrt(V[t,8,8])) for t in range(T)]
fig.add_trace(
    go.Scatter(x=df_data.index, y=cycle_lower, mode='lines', name="cycle_lower", marker_color="pink"),
    row=4, col=1     
)
fig.add_trace(
    go.Scatter(x=df_data.index, y=cycle_upper, mode='lines', name="cycle_upper", fill="tonexty", marker_color="pink"),
    row=4, col=1     
)
fig.add_trace(
    go.Scatter(x=df_data.index, y=x[:,8,0], mode='lines', name="cycle", marker_color="red"),
    row=4, col=1     
)

# 誤差
fig.add_trace(
    go.Scatter(x=df_data.index, y=df_data["y"]-x[:,0,0]-x[:,2,0]-x[:,8,0], mode='lines', name="error", marker_color="red"),
    row=5, col=1
) 

#%% 予測期間のプロット
# 予測時のデータ
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

fig = make_subplots(rows=3, cols=1)

# 全体
y_upper = [scipy.stats.norm.ppf(0.975, ypred[t,0,0], np.sqrt(Dpred[t,0,0])) for t in range(Tpred)]
y_lower = [scipy.stats.norm.ppf(0.025, ypred[t,0,0], np.sqrt(Dpred[t,0,0])) for t in range(Tpred)]
fig.add_trace(
    go.Scatter(x=df_pred.index, y=df_pred["yt"], mode='lines', name="true", marker_color="green"),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=df_pred.index, y=ypred[:,0,0], mode='lines', name="y_pred", marker_color="red"),
    row=1, col=1     
)
fig.add_trace(
    go.Scatter(x=df_pred.index, y=y_lower, mode='lines', name="y_lower", marker_color="pink"),
    row=1, col=1     
)
fig.add_trace(
    go.Scatter(x=df_pred.index, y=y_upper, mode='lines', name="y_upper", fill="tonexty", marker_color="pink"),
    row=1, col=1     
)

# トレンド
trend_upper = [scipy.stats.norm.ppf(0.975, xpred[t,0,0], np.sqrt(Vpred[t,0,0])) for t in range(Tpred)]
trend_lower = [scipy.stats.norm.ppf(0.025, xpred[t,0,0], np.sqrt(Vpred[t,0,0])) for t in range(Tpred)]
fig.add_trace(
    go.Scatter(x=df_pred.index, y=xpred[:,0,0], mode='lines', name="trend", marker_color="red"),
    row=2, col=1     
)
fig.add_trace(
    go.Scatter(x=df_pred.index, y=trend_lower, mode='lines', name="trend_lower", marker_color="pink"),
    row=2, col=1     
)
fig.add_trace(
    go.Scatter(x=df_pred.index, y=trend_upper, mode='lines', name="trend_upper", fill="tonexty", marker_color="pink"),
    row=2, col=1     
)

# 季節
seasonal_upper = [scipy.stats.norm.ppf(0.975, xpred[t,2,0], np.sqrt(Vpred[t,2,2])) for t in range(Tpred)]
seasonal_lower = [scipy.stats.norm.ppf(0.025, xpred[t,2,0], np.sqrt(Vpred[t,2,2])) for t in range(Tpred)]
fig.add_trace(
    go.Scatter(x=df_pred.index, y=seasonal_lower, mode='lines', name="seasonal_lower", marker_color="pink"),
    row=3, col=1     
)
fig.add_trace(
    go.Scatter(x=df_pred.index, y=seasonal_upper, mode='lines', name="seasonal_upper", fill="tonexty", marker_color="pink"),
    row=3, col=1     
)
fig.add_trace(
    go.Scatter(x=df_pred.index, y=xpred[:,2,0], mode='lines', name="seasonal", marker_color="red"),
    row=3, col=1     
)
# fig.update_xaxes(rangeslider_visible=True)
# fig.update_layout(
#     xaxis=dict(
#         rangeselector=dict(
#             buttons=list([
#                 dict(count=7,
#                      label="1w",
#                      step="day",
#                      stepmode="backward"),
#                 dict(count=1,
#                      label="1m",
#                      step="month",
#                      stepmode="backward"),
#                 dict(step="all")
#             ])
#         ),
#         rangeslider=dict(
#             visible=True
#         ),
#         type="date"
#     )
# )

fig.show()

# %%
