from components.pipelines.tabular.random_forest_classifier import RandomForestClassifier
from components.pipelines.tabular.random_forest_regressor import RandomForestRegressor

from sklearn.datasets import load_digits, load_boston
from sklearn.metrics import f1_score, r2_score
from sklearn.model_selection import train_test_split


def test_rf_classifier():
    print("\n============================================" +
          "\nTesting Random Forest Classifier" +
          "\n============================================")

    # Manual validataion that our pipeline object is as we expect.
    rf_pipeline = RandomForestClassifier()
    tunable_hyperparams = rf_pipeline.get_tunable_hyperparams()

    # Check that the hyperparameters are correct.
    for hyperparam in tunable_hyperparams:
        print(hyperparam)

    # Check that the steps are correct.
    print(rf_pipeline.steps_dict)

    # Check that we can score properly.
    digits = load_digits()
    X, X_test, y, y_test = train_test_split(
        digits.data, digits.target, train_size=1000, test_size=300)

    print("\nFitting pipeline...")

    rf_pipeline.fit(X, y)

    print("\nFit pipeline.")

    print("\nScoring pipeline...")

    predicted_y_val = rf_pipeline.predict(X_test)
    score = f1_score(predicted_y_val, y_test, average='micro')

    print("\nf1 micro score: %f" % score)


def test_rf_regressor():
    print("\n============================================" +
          "\nTesting Random Forest Regressor" +
          "\n============================================")
    # Manual validataion that our pipeline object is as we expect.
    rf_pipeline = RandomForestRegressor()
    tunable_hyperparams = rf_pipeline.get_tunable_hyperparams()

    # Check that the hyperparameters are correct.
    for hyperparam in tunable_hyperparams:
        print(hyperparam)

    # Check that the steps are correct.
    print(rf_pipeline.steps_dict)

    # Check that we can score properly.
    housing = load_boston()
    X, X_test, y, y_test = train_test_split(
        housing.data, housing.target, train_size=400, test_size=100)

    print("\nFitting pipeline...")

    rf_pipeline.fit(X, y)

    print("\nFit pipeline.")

    print("\nScoring pipeline...")

    predicted_y_val = rf_pipeline.predict(X_test)
    score = r2_score(y_test, predicted_y_val)

    print("\nr2 score: %f" % score)


if __name__ == "__main__":
    test_rf_classifier()
    test_rf_regressor()
