#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Usage example for TraditionalTextPipeline on the Newsgroups Dataset."""

from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from mlblocks.components.pipelines.text.traditional_text import TraditionalTextPipeline

print("""
============================================
Testing Traditional Text Pipeline
============================================
""")

newsgroups = fetch_20newsgroups()
X, X_test, y, y_test = train_test_split(
    newsgroups.data,
    newsgroups.target,
    train_size=9051,
    test_size=2263)

traditional_text = TraditionalTextPipeline()

# Check that the hyperparameters are correct.
for hyperparam in traditional_text.get_fixed_hyperparams():
    print(
        str(hyperparam) + ":",
        traditional_text.get_fixed_hyperparams()[hyperparam])
for hyperparam in traditional_text.get_tunable_hyperparams():
    print(hyperparam)

# Check that the steps are correct.
expected_steps = {
    'count_vectorizer', 'to_array', 'tfidf_transformer',
    'multinomial_nb'
}
steps = set(traditional_text.steps_dict.keys())
assert expected_steps == steps

# Check that we can score properly.
print("\nFitting pipeline...")
traditional_text.fit(X, y)
print("\nFit pipeline.")

print("\nScoring pipeline...")
predicted_y_val = traditional_text.predict(X_test)
score = f1_score(predicted_y_val, y_test, average='micro')
print("\nf1 micro score: %f" % score)
