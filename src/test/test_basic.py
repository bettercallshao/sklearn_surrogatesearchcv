from time import time

from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier

from sklearn_surrogatesearchcv import SurrogateSearchCV


def test_basic():
    digits = load_digits()
    X, y = digits.data, digits.target
    clf = RandomForestClassifier(n_estimators=5)

    param_def = [
        {
            'name': "max_depth",
            'integer': True,
            'lb': 3,
            'ub': 6,
        },
        {
            'name': 'max_features',
            'integer': True,
            'lb': 1,
            'ub': 11,
        },
        {
            'name': 'min_samples_split',
            'integer': True,
            'lb': 2,
            'ub': 11,
        },
    ]

    n_iter_search = 100
    surrogate_search = SurrogateSearchCV(clf, param_def=param_def,
                                         n_iter=n_iter_search, cv=5)

    start = time()
    surrogate_search.fit(X, y)
    print("SurrogateSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter_search))
    print("Best score is {}".format(surrogate_search.best_score_))
    print("Best params are {}".format(surrogate_search.best_params_))

    assert len(surrogate_search.params_history_) == n_iter_search
    assert len(surrogate_search.score_history_) == n_iter_search
