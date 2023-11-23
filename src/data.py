#%%
import statsmodels.datasets.co2 as co2

co2_raw = co2.load().data
df = co2_raw.iloc[353:] # 1965以降のデータを抽出
df = df.resample('M').mean() # 月次データに変換 (月の最終日を取得)"
df.index.name = "YEAR"

# %%
class TimesereiesData():
    """
    時系列データを扱うためのデータクラス
    """
    def __init__(self):
        pass
    
# %%
