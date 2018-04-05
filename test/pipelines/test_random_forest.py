from components.pipelines.tabular.random_forest_classifier import RandomForestClassifier

from sklearn.datasets import fetch_mldata
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# Manual validataion that our pipeline object is as we expect.
rf_pipeline = RandomForestClassifier()
tunable_hyperparams = rf_pipeline.get_tunable_hyperparams()

# Check that the hyperparameters are correct.
for hyperparam in tunable_hyperparams:
    print(hyperparam)

# Check that the steps are correct.
print(rf_pipeline.steps_dict)

# Check that we can score properly.
mnist = fetch_mldata('MNIST original')
X, X_test, y, y_test = train_test_split(
    mnist.data, mnist.target, train_size=1000, test_size=300)

print("\nFitting pipeline...")

rf_pipeline.fit(X, y)

print("\nFit pipeline.")

print("\nScoring pipeline...")

predicted_y_val = rf_pipeline.predict(X_test)
score = f1_score(predicted_y_val, y_test, average='micro')

print("\nf1 micro score: %f" % score)
