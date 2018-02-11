import numpy as np
from sklearn.model_selection import train_test_split as tt_split
from constants import LEARNING_RATE_TYPES, TASKS
from pyfm_fast import FM_fast, CSRDataset


class FM:
    """Implementation of Factorization Machine.
    Fitted by minimizing a regularized empirical loss with adaptive SGD.

    params
    ------
    k : int
        The dimensionality of the factorized 2-way interactions.
        モデルのハイパーパラメータである`V`の列方向の次元数.
    n_iter : int
        Number of iterations.
    k0 : bool
        Use bias. Defaults to true.
    k1 : bool
        Use 1-way interactions (learn feature weights).
        Defaults to true.
    init_stdev : double, optional
        Standard deviation for initialization of 2-way factors.
        Defaults to 0.01.
        Vの要素はデフォルトでN(0, 0.01)で初期化される.
    validation_size : double, optional
        Proportion of the training set to use for validation.
        Defaults to 0.01.
    learning_rate_schedule : string, optional
        The learning rate:
            constant: eta = eta0
            optimal: eta = 1.0/(t+t0) [default]
            invscaling: eta = eta0 / pow(t, power_t)
    initial_learning_rate : double
        Defaults to 0.01
    power_t : double
        The exponent for inverse scaling learning rate [default 0.5].
    t0 : double
        Constant in the denominator for optimal learning rate schedule.
        Defaults to 0.001.
    task : string
        possible values are defined in constants module.
        regression: Labels are real values.
        classification: Labels are either positive or negative.
    verbose : bool
        Whether or not to print current iteration, training error
        Defaults to True.
    shuffle_training: bool
        Whether or not to shuffle training dataset before learning
    seed : int
        The seed of the pseudo random number generator
    """
    def __init__(self,
                 k=10,
                 n_iter=1,
                 k0=True,
                 k1=True,
                 init_stdev=0.1,
                 validation_size=0.01,
                 learning_rate_schedule="optimal",
                 initial_learning_rate=0.01,
                 power_t=0.5,
                 t0=0.001,
                 task='classification',
                 verbose=True,
                 shuffle_training=True,
                 seed=3193):
        self.k = k
        self.n_iter = n_iter
        self.sum = np.zeros(self.k)
        self.sum_sqr = np.zeros(self.k)
        self.k0 = k0
        self.k1 = k1
        self.init_stdev = init_stdev
        self.validation_size = validation_size
        self.task = task
        self.shuffle_training = shuffle_training
        self.seed = seed

        # Learning rate Parameters
        self.learning_rate_schedule = learning_rate_schedule
        self.eta0 = initial_learning_rate
        self.power_t = power_t
        self.t = 1.0
        self.learning_rate = initial_learning_rate
        self.t0 = t0

        # Regularization Parameters (start with no regularization)
        self.reg_0 = 0.0
        self.reg_w = 0.0
        self.reg_v = np.repeat(0.0, k)

        # local parameters in the lambda_update step
        self.lambda_w_grad = 0.0
        self.lambda_v_grad = 0.0
        self.sum_f = 0.0
        self.sum_f_dash_f = 0.0
        self.verbose = verbose

    def _validate_params(self):
        """Validate input params. """
        if not isinstance(self.shuffle_training, bool):
            raise ValueError("shuffle must be either True or False")
        if self.n_iter <= 0:
            raise ValueError("n_iter must be > zero")
        if self.learning_rate_schedule in ("constant", "invscaling"):
            if self.eta0 <= 0.0:
                raise ValueError("eta0 must be > 0")

    def _get_learning_rate_type(self, learning_rate):
        """Map learning rate string to int for cython"""
        try:
            return LEARNING_RATE_TYPES[learning_rate]
        except KeyError:
            raise ValueError("learning rate %s "
                             "is not supported. " % learning_rate)

    def _get_task(self, task):
        """Map task string to int for cython"""
        try:
            return TASKS[task]
        except KeyError:
            raise ValueError("task %s "
                             "is not supported. " % task)

    def _bool_to_int(self, bool_arg):
        """Map bool to int for cython
        """
        flg = 1 if bool_arg else 0
        return flg

    def _prepare_y(self, y):
        """Maps labels to [-1, 1] space
        """
        y_i = np.ones(y.shape, dtype=np.float64, order="C")
        y_i[y != 1] = -1.0
        return y_i

    def fit(self, X, y):
        """Fit FM using SGD with Adaptive Regularization.

        params
        ------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training data

        y : numpy array of shape [n_samples]
            Target values

        returns
        -------
        self : returns an instance of itself.
        """
        if not isinstance(y, np.ndarray):
            y = np.array(y)

        self._validate_params()

        if self.task == "classification":
            y = self._prepare_y(y)

        self.max_target = max(y)
        self.min_target = min(y)

        # convert member variables to ints for use in cython
        k0 = self._bool_to_int(self.k0)
        k1 = self._bool_to_int(self.k1)
        shuffle_training = self._bool_to_int(self.shuffle_training)
        verbose = self._bool_to_int(self.verbose)
        learning_rate_schedule = self._get_learning_rate_type(
                self.learning_rate_schedule)
        task = self._get_task(self.task)

        X_train, validation, train_labels, validation_labels = tt_split(
            X, y, test_size=self.validation_size)
        self.num_attribute = X_train.shape[1]

        # Convert datasets to sklearn sequential datasets for fast traversal
        X_train_dataset = make_dataset(X_train, train_labels)
        validation_dataset = make_dataset(validation, validation_labels)

        # Set up params
        self.w0 = 0.0
        self.w = np.zeros(self.num_attribute)
        np.random.seed(seed=self.seed)
        self.v = np.random.normal(
                scale=self.init_stdev,
                size=(self.k, self.num_attribute))

        self.fm_fast = FM_fast(self.w,
                               self.v,
                               self.k,
                               self.num_attribute,
                               self.n_iter,
                               k0,
                               k1,
                               self.w0,
                               self.t,
                               self.t0,
                               self.power_t,
                               self.min_target,
                               self.max_target,
                               self.eta0,
                               learning_rate_schedule,
                               shuffle_training,
                               task,
                               self.seed,
                               verbose)

        return self.fm_fast.fit(X_train_dataset, validation_dataset)

    def predict(self, X):
        """Predict using the factorization machine

        Parameters
        ----------
        X : sparse matrix, shape = [n_samples, n_features]
        or
        X : single instance [1, n_features]

        Returns
        -------
        float if X is one instance
        array, shape = [n_samples] if X is sparse matrix
           Predicted target values per element in X.
        """
        sparse_X = make_dataset(X, np.ones(X.shape[0]))

        return self.fm_fast._predict(sparse_X)


def make_dataset(X, y_i):
    """Create ``Dataset`` abstraction for sparse and dense inputs."""
    sample_weight = np.ones(
            X.shape[0],
            dtype=np.float64,
            order='C')  # ignore sample weight for the moment
    dataset = CSRDataset(X.data, X.indptr, X.indices, y_i, sample_weight)
    return dataset
