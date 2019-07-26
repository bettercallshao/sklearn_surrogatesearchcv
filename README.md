# Surrogate Search CV

This package implements a randomized hyper parameter search for sklearn (similar to `RandomizedSearchCV`) but utilizes surrogate adaptive sampling from pySOT. Use this similarly to GridSearchCV with a few extra paramters.

## Usage

```
pip install sklearn-surrogatesearchcv
```

The interface is unimaginative, stylistically similar to `RandomizedSearchCV`.

```
class SurrogateSearchCV(object):
    """Surrogate search with cross validation for hyper parameter tuning.
    """

    def __init__(self, estimator, n_iter=10, param_def=None, refit=False,
                 **kwargs):
        """
        :param estimator: estimator
        :param n_iter: number of iterations to run (default 10)
        :param param_def: list of dictionaries, e.g.
            [
                {
                    'name': 'alpha',
                    'integer': False,
                    'lb': 0.1,
                    'ub': 0.9,
                },
                {
                    'name': 'max_depth',
                    'integer': True,
                    'lb': 3,
                    'ub': 12,
                }
            ]
        :param **: every other parameter is the same as GridSearchCV
        """
```

The result can be found in the following properties of the class instance after running.

```
params_history_
score_history_
best_params_
best_score_
```

For a complete example, please refer to `src/test/test_basic.py`.

## CI

[![CircleCI](https://circleci.com/gh/timlyrics/sklearn_surrogatesearchcv.svg?style=svg)](https://circleci.com/gh/timlyrics/sklearn_surrogatesearchcv)

## Resources

A slide about role of surrogate optimization in ml. [link](https://www.slideshare.net/TimTan2/machine-learning-vs-traditional-optimization)
