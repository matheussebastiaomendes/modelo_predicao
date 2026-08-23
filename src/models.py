from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor


def get_models():

    return {
        "lightgbm": LGBMRegressor(),

        "random_forest": RandomForestRegressor(
            n_estimators=15,
            criterion='squared_error',
            random_state=20,
            n_jobs=-1,
            min_samples_leaf=2
        )
    }