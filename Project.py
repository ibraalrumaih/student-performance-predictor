import pandas as pd

df = pd.read_csv("study_data.csv")

print(df)

print(df["grade"].mean())
print(df["grade"].max())
print(df["grade"].min())

print(df[df["grade"] > 80])

from sklearn.linear_model import LinearRegression

X = df[["hours"]]
y = df["grade"]

model = LinearRegression()
model.fit(X, y)
print(model.predict([[9]]))

from sklearn.metrics import mean_squared_error

pred = model.predict(X)
mse = mean_squared_error(y, pred)

print(mse)