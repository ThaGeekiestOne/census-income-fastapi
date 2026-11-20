import pandas as pd

DATA = "data/census.csv"

df = pd.read_csv(DATA, skipinitialspace=True)
df.columns = df.columns.str.strip()
for col in df.select_dtypes(include="object"):
    df[col] = df[col].str.strip()

df.to_csv(DATA, index=False)
print(df.shape, df["salary"].unique())
