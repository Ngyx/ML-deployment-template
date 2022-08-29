from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


pipe = Pipeline(
    [
        (
            "LinearRegression",
            LinearRegression(
                fit_intercept=False,
            ),
        ),
    ]
)
