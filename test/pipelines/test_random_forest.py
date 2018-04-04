from components.pipelines.tabular.random_forest_classifier import RandomForestClassifier

# Manual validataion that our pipeline object is as we expect.
rf_pipeline = RandomForestClassifier()
tunable_hyperparams = rf_pipeline.get_tunable_hyperparams()

# Check that the hyperparameters are correct.
for hyperparam in tunable_hyperparams:
    print(hyperparam)

# Check that the steps are correct.
print(rf_pipeline.steps_dict)
