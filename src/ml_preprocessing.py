from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder


def create_preprocessor(
    numerical_features,
    categorical_features
):

    numerical_transformer = Pipeline([
        (
            'imputer',
            SimpleImputer(strategy='median')
        ),
        (
            'scaler',
            StandardScaler()
        )
    ])

    categorical_transformer = Pipeline([
        (
            'encoder',
            OneHotEncoder(
                drop='first',
                handle_unknown='ignore'
            )
        )
    ])

    preprocessor = ColumnTransformer([
        (
            'num',
            numerical_transformer,
            numerical_features
        ),
        (
            'cat',
            categorical_transformer,
            categorical_features
        )
    ])

    return preprocessor