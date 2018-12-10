from __future__ import print_function
import numpy as np
from sklearn.model_selection import GridSearchCV
from pySOT.experimental_design import SymmetricLatinHypercube
from pySOT.strategy import SRBFStrategy
from pySOT.surrogate import RBFInterpolant, CubicKernel, LinearTail, SurrogateUnitBox
from pySOT.optimization_problems import OptimizationProblem
from poap.controller import SerialController

class SurrogateSearchCV(object):
    """Surrogate search with cross validation for hyper parameter tuning.
    """

    def __init__(self, estimator, n_iter=10, param_def=None, refit=False, **kwargs):
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
        self.estimator = estimator
        self.n_iter = n_iter

        if not param_def:
            raise Exception('Parameter param_def must be defined!')

        if refit:
            raise Exception('Refit not supported')

        for d in param_def:
            if 'name' not in d:
                raise Exception('Name must be defined for each parameter')

            if 'integer' not in d:
                d['integer'] = False

            if 'lb' not in d or 'ub' not in d:
                raise Exception('Fields lb and ub must be defined for {}'.format(d['name']))
            if d['ub'] <= d['lb']:
                raise Exception('Field ub must be larger than lb for {}'.format(d['name']))

        self.param_def = param_def
        self.kwargs = kwargs
        self.best_score_ = None
        self.best_params_ = None
        self.params_history_ = []
        self.score_history_ = []

    def fit(self, X, y=None, **kwargs):
        """

        :param X: training data
        :param **: parameters to be passed to GridSearchCV
        """
        # wrap for pySOT
        class Target(OptimizationProblem):
            def __init__(self, outer):
                self.outer = outer
                param_def = outer.param_def
                self.lb = np.array(list(d['lb'] for d in param_def))
                self.ub = np.array(list(d['ub'] for d in param_def))
                self.dim = len(param_def)
                self.int_var = np.array(list(i for i, d in enumerate(param_def) if d['integer']))
                self.cont_var = np.array(list(i for i, d in enumerate(param_def) if i not in self.int_var))

            def eval(self, x):
                print('Eval {} ...'.format(x))
                param_def = self.outer.param_def
                outer = self.outer
                # prepare parameters grid for gridsearchcv
                param_grid = {d['name']: [x[i]] for i, d in enumerate(param_def)}
                # create gridsearchcv to evaluate the cv
                gs = GridSearchCV(outer.estimator, param_grid, refit=False, **outer.kwargs)
                # never refit during iteration, refit at the end
                gs.fit(X, y=y, **kwargs)
                gs_score = gs.best_score_
                # gridsearchcv score is better when greater
                if not outer.best_score_ or gs_score > outer.best_score_:
                    outer.best_score_ = gs_score
                    outer.best_params_ = gs.best_params_
                # also record history
                outer.params_history_.append(x)
                outer.score_history_.append(gs_score)
                print('Eval {} => {}'.format(x, gs_score))
                # pySOT score is the lower the better, so return the negated
                return -gs_score

        # pySOT routine
        # TODO: make this configurable
        target = Target(self)
        rbf = SurrogateUnitBox(RBFInterpolant(dim=target.dim, kernel=CubicKernel(),
                                              tail=LinearTail(target.dim)),
                               lb=target.lb, ub=target.ub)
        slhd = SymmetricLatinHypercube(
            dim=target.dim, num_pts=2 * (target.dim + 1))

        # Create a strategy and a controller
        controller = SerialController(objective=target.eval)
        controller.strategy = SRBFStrategy(
            max_evals=self.n_iter, batch_size=1, opt_prob=target, exp_design=slhd,
            surrogate=rbf, asynchronous=False)

        print("Maximum number of evaluations: {}".format(self.n_iter))
        print("Strategy: {}".format(controller.strategy.__class__.__name__))
        print("Experimental design: {}".format(slhd.__class__.__name__))
        print("Surrogate: {}".format(rbf.__class__.__name__))

        # Run the optimization strategy
        result = controller.run()

        print('Best value found: {0}'.format(result.value))
        print('Best solution found: {0}\n'.format(
            np.array_str(result.params[0], max_line_width=np.inf,
                         precision=5, suppress_small=True)))
