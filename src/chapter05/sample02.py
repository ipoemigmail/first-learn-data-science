import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
import sklearn.datasets as ds

mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams["font.family"] = "AppleGothic"

bs = ds.load_boston()
df = pd.DataFrame(bs.data, columns=bs.feature_names)
df["MEDV"] = bs.target
print(df)

df.plot(x="CRIM", y="MEDV", kind="scatter")
plt.title("일반 축에 나타낸 범죄 발생률")
plt.show()
plt.close()

df.plot(x="CRIM", y="MEDV", kind="scatter", logx=True)
plt.title("로그 축에 나타낸 범죄 발생률")
plt.show()
plt.close()
