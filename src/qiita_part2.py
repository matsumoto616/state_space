#%% インポート
import numpy as np
import pandas as pd
import statsmodels.datasets

#%% エルニーニョデータ
df = statsmodels.datasets.elnino.load().data

#%% 整形
df = pd.DataFrame(
    index=pd.date_range(start="1950/01", end="2010/12", freq="MS"),
    columns=["Temperature"],
    data=df.drop("YEAR", axis=1).values.flatten()
)

#%% 人工的に欠測値を作成
df["Temperature_missing"] = df["Temperature"]
df["Temperature_missing"].iloc[100:150] = np.nan # 50か月欠損
df["Temperature_missing"].iloc[550:600] = np.nan # 50か月欠損

#%% プロット
import plotly.graph_objects as go
from utils import plot

config = {
    "yaxis": {"min": 15, "max": 35},
    "rows": 2,
    "cols": 1,
    "titles": ["original data", "missing data"]
}
traces = {
    "data": {
        "x": df.index,
        "y": df["Temperature"],
        "color": "black",
        "row": 1,
        "col": 1,
        "showlegend": False
    },
    "missing_data": {
        "x": df.index,
        "y": df["Temperature_missing"],
        "color": "black",
        "row": 2,
        "col": 1,
        "showlegend": False
    }
}
request = {"config": config, "traces": traces}
fig = plot(request)
fig.show()

#%% ランダムウォークモデル
# データの長さ
T = len(df)

# 状態方程式の行列
F0 = np.array([[1]])
              
# システムノイズの行列
G0 = np.array([[1]])

# 観測方程式の行列
H0 = np.array([[1]])

# システムノイズの分散共分散行列
Q0 = np.array([[1]])

# 観測ノイズの分散共分散行列
R0 = np.array([[1]])

# 初期状態（データの平均・分散で適当に決定）
mu0 = np.array([[df["Temperature"].mean()]])
V0 = np.array([[df["Temperature"].var()]])

#%% スタック
def stack_matrix(M0, N):
    """
    ndarray Mを0軸にN個重ねた ndarrayを作成する
    """
    M = np.zeros((N, M0.shape[0], M0.shape[1]))
    for n in range(N):
        M[n] = M0

    return M

F = stack_matrix(F0, T)
G = stack_matrix(G0, T)
H = stack_matrix(H0, T)
Q = stack_matrix(Q0, T)
R = stack_matrix(R0, T)

# 観測値（T×1×1行列とすることに注意）
y = np.expand_dims(df["Temperature"].values, (1, 2))

#%% モデル作成
from model import LinearGaussianStateSpaceModel
model = LinearGaussianStateSpaceModel(mu0, V0, F, G, H, Q, R, y)

# %%
filter_result = model.kalman_filter()

#%% 長期予測
horizon = 12*5
pred_index = pd.date_range(start=df.index[-1], periods=horizon+1, freq="MS")[1:]
Fp = stack_matrix(F0, horizon)
Gp = stack_matrix(G0, horizon)
Hp = stack_matrix(H0, horizon)
Qp = stack_matrix(Q0, horizon)
Rp = stack_matrix(R0, horizon)
predictor_result = model.kalman_predictor(Fp, Gp, Hp, Qp, Rp)

#%% プロット
config = {
    "yaxis": {"min": 15, "max": 35},
    "rows": 1,
    "cols": 1,
    "titles": [None]
}
traces = {
    "data": {
        "x": df.index,
        "y": df["Temperature"],
        "color": "black",
        "row": 1,
        "col": 1,
        "showlegend": True
    },
    "filter": {
        "x": df.index,
        "y": filter_result["mu_filtered"][:,0,0],
        "color": "red",
        "row": 1,
        "col": 1,
        "showlegend": True
    },
    "predict": {
        "x": pred_index,
        "y": predictor_result["mu_predicted"][:,0,0],
        "color": "blue",
        "row": 1,
        "col": 1,
        "showlegend": True
    }
}

request = {"config": config, "traces": traces}
fig = plot(request)
fig.show()


#%% システムノイズを小さくした場合
Q0 = np.array([[1e-3]])
Q = stack_matrix(Q0, T)
model.set_params(Q=Q)

filter_result = model.kalman_filter()
predictor_result = model.kalman_predictor(Fp, Gp, Hp, Qp, Rp)

#%% プロット
config = {
    "yaxis": {"min": 15, "max": 35},
    "rows": 1,
    "cols": 1,
    "titles": [None]
}
traces = {
    "data": {
        "x": df.index,
        "y": df["Temperature"],
        "color": "black",
        "row": 1,
        "col": 1,
        "showlegend": True
    },
    "filter": {
        "x": df.index,
        "y": filter_result["mu_filtered"][:,0,0],
        "color": "red",
        "row": 1,
        "col": 1,
        "showlegend": True
    },
    "predict": {
        "x": pred_index,
        "y": predictor_result["mu_predicted"][:,0,0],
        "color": "blue",
        "row": 1,
        "col": 1,
        "showlegend": True
    }
}

request = {"config": config, "traces": traces}
fig = plot(request)
fig.show()

#%% 季節調整モデル
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

# 状態方程式の行列
F0_trend = np.array(
    [[1]]
)
F0_seasonal = np.array(
    [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
     [ 1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
     [ 0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
     [ 0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0],
     [ 0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0],
     [ 0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0],
     [ 0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0],
     [ 0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0],
     [ 0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0],
     [ 0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0],
     [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0]]
)
F0 = make_diag_stack_matrix([F0_trend, F0_seasonal])

# システムノイズの行列
G0_trend = np.array(
    [[1]]
)
G0_seasonal = np.array(
    [[1],
     [0],
     [0],
     [0],
     [0],
     [0],
     [0],
     [0],
     [0],
     [0],
     [0]]
)
G0 = make_diag_stack_matrix([G0_trend, G0_seasonal])

# 観測方程式の行列
H0_trend = np.array([[1]])
H0_seasonal = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
H0 = make_hstack_matrix([H0_trend, H0_seasonal])

# システムノイズの分散共分散行列
Q0_trend = np.array([[1]])
Q0_seasonal = np.array([[1]])
Q0 = make_diag_stack_matrix([Q0_trend, Q0_seasonal])

# 観測ノイズの分散共分散行列
R0 = np.array([[1]])

# スタック
F = stack_matrix(F0, T)
G = stack_matrix(G0, T)
H = stack_matrix(H0, T)
Q = stack_matrix(Q0, T)
R = stack_matrix(R0, T)

# 初期状態（データの平均・分散で適当に決定）
mu0 = np.array([[df["Temperature"].mean() for i in range(F0.shape[0])]]).T
V0 = np.eye(F0.shape[0]) * df["Temperature"].var()

# 観測値（T×1×1行列とすることに注意）
y = np.expand_dims(df["Temperature"].values, (1, 2))

#%% モデル作成
model = LinearGaussianStateSpaceModel(mu0, V0, F, G, H, Q, R, y)

#%% カルマンフィルタ
filter_result = model.kalman_filter()

#%% 長期予測
horizon = 12*5
pred_index = pd.date_range(start=df.index[-1], periods=horizon+1, freq="MS")[1:]
Fp = stack_matrix(F0, horizon)
Gp = stack_matrix(G0, horizon)
Hp = stack_matrix(H0, horizon)
Qp = stack_matrix(Q0, horizon)
Rp = stack_matrix(R0, horizon)
predictor_result = model.kalman_predictor(Fp, Gp, Hp, Qp, Rp)

#%%
config = {
    "yaxis": {"min": None, "max": None},
    "rows": 3,
    "cols": 1,
    "titles": [None, "trend", "seasonal"]
}
traces = {
    "data": {
        "x": df.index,
        "y": df["Temperature"],
        "color": "black",
        "row": 1,
        "col": 1,
        "showlegend": True
    },
    "filter": {
        "x": df.index,
        "y": filter_result["nu_predicted"][:,0,0],
        "color": "red",
        "row": 1,
        "col": 1,
        "showlegend": True
    },
    "predict": {
        "x": pred_index,
        "y": predictor_result["nu_predicted"][:,0,0],
        "color": "blue",
        "row": 1,
        "col": 1,
        "showlegend": True
    },
    "filter_trend": {
        "x": df.index,
        "y": filter_result["mu_filtered"][:,0,0],
        "color": "red",
        "row": 2,
        "col": 1,
        "showlegend": True
    },
    "predict_trend": {
        "x": pred_index,
        "y": predictor_result["mu_predicted"][:,0,0],
        "color": "blue",
        "row": 2,
        "col": 1,
        "showlegend": True
    },
    "filter_seasonal": {
        "x": df.index,
        "y": filter_result["mu_filtered"][:,1,0],
        "color": "red",
        "row": 3,
        "col": 1,
        "showlegend": True
    },
    "predict_seasonal": {
        "x": pred_index,
        "y": predictor_result["mu_predicted"][:,1,0],
        "color": "blue",
        "row": 3,
        "col": 1,
        "showlegend": True
    }
}

request = {"config": config, "traces": traces}
fig = plot(request)
fig.show()

#%% 欠測値の補間
y = np.expand_dims(df["Temperature_missing"].values, (1, 2))
model.set_params(y=y)

# カルマンフィルタ実行（比較用）
filter_result = model.kalman_filter()
# 固定区間平滑化実行
smoother_result = model.kalman_smoother()

#%% プロット
config = {
    "yaxis": {"min": 15, "max": 35},
    "rows": 3,
    "cols": 1,
    "titles": ["data", "filter", "smoother"]
}
traces = {
    "missing_data": {
        "x": df.index,
        "y": df["Temperature_missing"],
        "color": "black",
        "row": 1,
        "col": 1,
        "showlegend": True
    },
    "filter": {
        "x": df.index,
        "y": filter_result["nu_predicted"][:,0,0],
        "color": "red",
        "row": 2,
        "col": 1,
        "showlegend": True
    },
    "filter_upper": {
        "x": df.index,
        "y": filter_result["nu_predicted"][:,0,0] \
            + 2 * np.sqrt(filter_result["D_predicted"][:,0,0]),
        "color": "pink",
        "row": 2,
        "col": 1,
        "showlegend": True
    },
    "filter_lower": {
        "x": df.index,
        "y": filter_result["nu_predicted"][:,0,0] \
            - 2 * np.sqrt(filter_result["D_predicted"][:,0,0]),
        "color": "pink",
        "row": 2,
        "col": 1,
        "showlegend": True
    },
    "smoother": {
        "x": df.index,
        "y": smoother_result["nu_predicted"][:,0,0],
        "color": "blue",
        "row": 3,
        "col": 1,
        "showlegend": True
    },
    "smoother_upper": {
        "x": df.index,
        "y": smoother_result["nu_predicted"][:,0,0] \
            + 2 *np.sqrt(smoother_result["D_predicted"][:,0,0]),
        "color": "skyblue",
        "row": 3,
        "col": 1,
        "showlegend": True
    },
    "smoother_lower": {
        "x": df.index,
        "y": smoother_result["nu_predicted"][:,0,0] \
            - 2 *np.sqrt(smoother_result["D_predicted"][:,0,0]),
        "color": "skyblue",
        "row": 3,
        "col": 1,
        "showlegend": True
    },
}

request = {"config": config, "traces": traces}
fig = plot(request)
fig.show()

#%% 最尤法
def objective(params):
    # 初期状態
    mu0 = np.array([[params[0] for _ in range(F0.shape[0])]]).T
    V0 = np.eye(F0.shape[0]) * params[1]

    # システムノイズの分散共分散行列
    Q0_trend = np.array([[params[2]]])
    Q0_seasonal = np.array([[params[3]]])
    Q0 = make_diag_stack_matrix([Q0_trend, Q0_seasonal])
    Q = stack_matrix(Q0, T)
    
    # 観測ノイズの分散共分散行列
    R0 = np.array([[params[4]]])
    R = stack_matrix(R0, T)

    # パラメータ更新
    model.set_params(mu0=mu0, V0=V0, Q=Q, R=R)

    # カルマンフィルタを実行
    result = model.kalman_filter(calc_liklihood=True)

    return -1 * result["logliklihood"] # llの最大化＝-llの最小化

#%% パラメータ最適化
from scipy.optimize import minimize

optimization_result = minimize(
    fun=objective,
    x0=[0, 1, 1, 1, 1],
    bounds=[(None, None), (1e-5, None), (1e-5, None), (1e-5, None), (1e-5, None)], 
    method="L-BFGS-B"
)

#%% 最適パラメータで計算
params = optimization_result.x

# 初期状態
mu0 = np.array([[params[0] for _ in range(F0.shape[0])]]).T
V0 = np.eye(F0.shape[0]) * params[1]

# システムノイズの分散共分散行列
Q0_trend = np.array([[params[2]]])
Q0_seasonal = np.array([[params[3]]])
Q0 = make_diag_stack_matrix([Q0_trend, Q0_seasonal])
Q = stack_matrix(Q0, T)

# 観測ノイズの分散共分散行列（1×1行列とすることに注意）
R0 = np.array([[params[4]]])
R = stack_matrix(R0, T)

# パラメータ更新
model.set_params(mu0=mu0, V0=V0, Q=Q, R=R)

#%% カルマンフィルタを実行
filter_result = model.kalman_filter(calc_liklihood=True)
smoother_result = model.kalman_smoother()

#%% プロット
config = {
    "yaxis": {"min": 15, "max": 35},
    "rows": 3,
    "cols": 1,
    "titles": ["data", "filter", "smoother"]
}
traces = {
    "missing_data": {
        "x": df.index,
        "y": df["Temperature_missing"],
        "color": "black",
        "row": 1,
        "col": 1,
        "showlegend": True
    },
    "filter": {
        "x": df.index,
        "y": filter_result["nu_predicted"][:,0,0],
        "color": "red",
        "row": 2,
        "col": 1,
        "showlegend": True
    },
    "filter_upper": {
        "x": df.index,
        "y": filter_result["nu_predicted"][:,0,0] \
            + 2 * np.sqrt(filter_result["D_predicted"][:,0,0]),
        "color": "pink",
        "row": 2,
        "col": 1,
        "showlegend": True
    },
    "filter_lower": {
        "x": df.index,
        "y": filter_result["nu_predicted"][:,0,0] \
            - 2 * np.sqrt(filter_result["D_predicted"][:,0,0]),
        "color": "pink",
        "row": 2,
        "col": 1,
        "showlegend": True
    },
    "smoother": {
        "x": df.index,
        "y": smoother_result["nu_predicted"][:,0,0],
        "color": "blue",
        "row": 3,
        "col": 1,
        "showlegend": True
    },
    "smoother_upper": {
        "x": df.index,
        "y": smoother_result["nu_predicted"][:,0,0] \
            + 2 *np.sqrt(smoother_result["D_predicted"][:,0,0]),
        "color": "skyblue",
        "row": 3,
        "col": 1,
        "showlegend": True
    },
    "smoother_lower": {
        "x": df.index,
        "y": smoother_result["nu_predicted"][:,0,0] \
            - 2 *np.sqrt(smoother_result["D_predicted"][:,0,0]),
        "color": "skyblue",
        "row": 3,
        "col": 1,
        "showlegend": True
    },
}

request = {"config": config, "traces": traces}
fig = plot(request)
fig.show()

# %%
