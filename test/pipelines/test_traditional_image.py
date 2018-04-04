from sklearn.metrics import f1_score
from components.pipelines.image.traditional_image import TraditionalImagePipeline
from load_data import d3m_load_data

# Manual validation that our pipeline object is as we expect.
image_pipeline = TraditionalImagePipeline()
tunable_hyperparams = image_pipeline.get_tunable_hyperparams()

# Check that the hyperparameters are correct.
for hyperparam in tunable_hyperparams:
    print(hyperparam)

# Check that the steps are correct.
print
print(image_pipeline.steps_dict)

# Check that we can score properly.
image_data_directory = 'data/MNIST_D3M'
image_data = d3m_load_data(
    data_directory=image_data_directory, is_d3m=True, sample_size_pct=0.1)

print("\nFitting pipeline...")

image_pipeline.fit(image_data.X_train, image_data.y_train)

print("\nFit pipeline.")

print("\nScoring pipeline...")

predicted_y_val = image_pipeline.predict(image_data.X_val)
score = f1_score(predicted_y_val, image_data.y_val, average='micro')

print("\nf1 micro score: %f" % score)
