import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression

from src.preprocessing import clean_data

from src.ml_preprocessing import create_preprocessor

from src.models import get_models

from src.evaluate import evaluate_model


# 1. Carregar
df = pd.read_csv(
    "dados/preco_carros.csv"
)

# 2. Limpar
df = clean_data(df)

# 3. Separar target
target = "preço"

X = df.drop(
    columns=[target]
)

y = df[target]

# 4. Definir features
categorical_features = X.select_dtypes(
    include=['object']
).columns.tolist()

numerical_features = X.select_dtypes(
    exclude=['object']
).columns.tolist()

# 5. Train/Test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

# 6. Preprocessing
preprocessor = create_preprocessor(
    numerical_features,
    categorical_features
)

# 7. Modelos
models = get_models()

# 8. Treinar e avaliar
results = {}

for name, model in models.items():

    pipeline = Pipeline([
        (
            'preprocessor',
            preprocessor
        ),

        (
            'feature_selection',
            SelectKBest(
                score_func=f_regression,
                k=10
            )
        ),

        (
            'model',
            model
        )
    ])

    pipeline.fit(
        X_train,
        y_train
    )

    y_pred = pipeline.predict(
        X_test
    )

    metrics = evaluate_model(
        y_test,
        y_pred
    )

    results[name] = metrics

    print(name)
    print(metrics)