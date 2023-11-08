#%%
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
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
from scipy.signal import lfilter

# True model parameters
nobs = int(1e3)
true_phi = np.r_[0.5, -0.2]
true_sigma = 1**0.5

# Simulate a time series
np.random.seed(1234)
disturbances = np.random.normal(0, true_sigma, size=(nobs,))
endog = lfilter([1], np.r_[1, -true_phi], disturbances)

# Construct the model
class AR2(sm.tsa.statespace.MLEModel):
    def __init__(self, endog):
        # Initialize the state space model
        super(AR2, self).__init__(endog, k_states=2, k_posdef=1,
                                  initialization='stationary')

        # Setup the fixed components of the state space representation
        self['design'] = [1, 0]
        self['transition'] = [[0, 0],
                                  [1, 0]]
        self['selection', 0, 0] = 1

    # Describe how parameters enter the model
    def update(self, params, transformed=True, **kwargs):
        params = super(AR2, self).update(params, transformed, **kwargs)

        self['transition', 0, :] = params[:2]
        self['state_cov', 0, 0] = params[2]

    # Specify start parameters and parameter names
    @property
    def start_params(self):
        return [0,0,1]  # these are very simple

# Create and fit the model
mod = AR2(endog)
res = mod.fit()
print(res.summary())
# %%
