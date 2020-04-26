import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
import sklearn.datasets


def get_iris_data():
    ds = sklearn.datasets.load_iris()
    df = pd.DataFrame(ds["data"], columns=ds["feature_names"])
    code_species_map = dict(zip(range(3), ds["target_names"]))
    df["species"] = [code_species_map[c] for c in ds["target"]]
    return df


mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams["font.family"] = "AppleGothic"

df = get_iris_data()
print(df)

# sums_by_species = df.groupby("species").sum()
# var = "sepal width (cm)"
# sums_by_species[var].plot(kind='pie', fontsize=20)
# plt.ylabel(var, horizontalalignment="left")
# plt.title(var + '로 분류된 붓꽃', fontsize=25)
# plt.show()
# plt.close()
#
# sums_by_species = df.groupby("species").sum()
# sums_by_species.plot(kind='pie', subplots=True, layout=(2, 2), legend=False)
# plt.title("종에 따른 전체 측정값")
# plt.show()
# plt.close()
#
# sums_by_species = df.groupby("species").sum()
# var = "sepal width (cm)"
# sums_by_species[var].plot(kind='bar', fontsize=15, rot=30)
# plt.title(var + '로 분류된 붓꽃', fontsize=20)
# plt.show()
# plt.close()
#
#
# sums_by_species = df.groupby("species").sum()
# sums_by_species.plot(kind='bar', subplots=True, fontsize=12)
# plt.title("종에 따른 전체 측정값")
# plt.show()
# plt.close()
#
# df.plot(kind="hist", subplots=True, layout=(2, 2))
# plt.suptitle("붓꽃 히스토그램", fontsize=20)
# plt.show()
#
# for spec in df["species"].unique():
#     forspec = df[df["species"] == spec]
#     forspec["petal length (cm)"].plot(kind="hist", alpha=0.4, label=spec)
#
# plt.legend(loc="upper right")
# plt.suptitle("종에 따른 꽃잎 길이")
# plt.show()
# plt.close()
#
# col = df["petal length (cm)"]
# average = col.mean()
# std = col.std()
# median = col.quantile(0.5)
# percentile25 = col.quantile(0.25)
# percentile75 = col.quantile(0.75)
# clean_avg = col[(col > percentile25) & (col < percentile75)].mean()
#
# print(f"average: {average}")
# print(f"clean_avg: {clean_avg}")
#
# col = "sepal length (cm)"
# df["ind"] = pd.Series(df.index).apply(lambda i: i % 50)
# df.pivot("ind", "species")[col].plot(kind="box")
# plt.show()
# plt.close()

df.plot(kind="scatter", x="sepal length (cm)", y="sepal width (cm)")
plt.title("길이 대 너비")
plt.show()
plt.close()
