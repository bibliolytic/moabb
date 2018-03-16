from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from pyriemann.spatialfilters import CSP
from pyriemann.estimation import Covariances as Cov
import numpy as np
import scipy.signal as signal
import logging

log = logging.getLogger()


class LogVariance(BaseEstimator, TransformerMixin):

    def __init__(self, picks=None, flatten=False):
        '''
        Picks allow you to choose only certain frequency bands
        '''
        self.picks = picks
        self.flatten = flatten

    def fit(self, X, y):
        """fit."""
        return self

    def _transform(self, X, picks):
        return np.log(np.var(X[..., picks], axis=2))

    def transform(self, X):
        """transform"""
        assert X.ndim == 4
        if self.picks is None:
            picks = np.arange(X.shape[-1])
        else:
            picks = self.picks
        out = self._transform(X, picks)
        log.debug('{}:{}'.format(type(self).__name__, out.shape))
        if self.flatten:
            return np.reshape(out, (X.shape[0], X.shape[1]*len(picks)))
        else:
            return out


class FM(LogVariance):

    def _transform(self, X, picks):
        '''
        Note that without sampling rate these are unnormalize and cannot be used
        cross-dataset
        '''
        xphase = np.unwrap(np.angle(signal.hilbert(X[..., picks], axis=2)))
        return np.median(np.diff(xphase, axis=2) / (2*np.pi), axis=2)


class MultibandCovariances(BaseEstimator, TransformerMixin):

    def __init__(self, estimator='scm', est=None):
        """Init."""
        if est is None:
            self.est = Cov(estimator=estimator)
        else:
            self.est = est

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert X.ndim == 4
        assert X.shape[-1] != 1
        out = np.stack([self.est.transform(X[..., i])
                        for i in range(X.shape[-1])], axis=3)
        return out


class AverageCovariance(MultibandCovariances):

    def transform(self, X):
        return super().transform(X).mean(axis=-1)


class FBCSP(BaseEstimator, TransformerMixin):
    """Implementation of the CSP spatial Filtering with Covariance as input.

    Implementation of the famous Common Spatial Pattern Algorithm, but with
    covariance matrices as input. In addition, the implementation allow
    different metric for the estimation of the class-related mean covariance
    matrices, as described in [3].

    This implementation support multiclass CSP by means of approximate joint
    diagonalization. In this case, the spatial filter selection is achieved
    according to [4].

    Parameters
    ----------
    filters_per_band : int (default 4)
        The number of components to decompose M/EEG signals.
    feature_selector : sklearn.feature_selection instance or None
        Feature selection algorithm, defaults to top 10 via MI
    filter_obj : object, or None
        Object that computes spatial filters, or CSP with filters_per_band filters if None

    Attributes
    ----------


    See Also
    --------
    MDM, SPoC

    References
    ----------
    """

    def __init__(self, filters_per_band=4, total_filters=10, score_func='mi',
                 filter_obj=None, feature_selector=None):
        """Init."""
        if filter_obj is None:
            self.filter_obj = CSP(filters_per_band)
        else:
            self.filter_obj = filter_obj
        if feature_selector is None:
            self.feature_selector = FBCSPFeatureSelection(
                score_func, k=total_filters)
        else:
            self.feature_selector = feature_selector

    def fit(self, X, y):
        assert X.ndim == 4, 'insufficient dimensions in given data'
        self.filters = [clone(self.filter_obj) for i in range(X.shape[-1])]
        [self.filters[i].fit(X[..., i], y) for i in range(X.shape[-1])]
        self.feature_selector.fit(self._transform(X), y)
        return self

    def _transform(self, X):
        assert X.shape[-1] == len(self.filters), "given X has {} band passes while trained model has {}".format(
            X.shape[-1], len(self.filters))
        out = np.concatenate([self.filters[i].transform(X[..., i])
                              for i in range(X.shape[-1])], axis=1)
        return out

    def transform(self, X):
        out = self.feature_selector.transform(self._transform(X))
        return out


class FBCSPFeatureSelection:

    def __repr__(self):
        return '{}(score={}, k={})'.format(self.__class__.__name__,
                                           self.score,
                                           self.k)

    def __init__(self, score, k=10):
        """

        """
        self.score = score
        self.k = k
        if score == 'mi':
            self.f = SelectKBest(score_func=mutual_info_classif, k=k)
        elif score == 'anova':
            self.f = SelectKBest(k=k)
        else:
            raise ValueError("Only mi and anova implemented so far")

    def fit(self, X, y):
        self.f.fit(X, y)
        return self

    def transform(self, X):
        return self.f.transform(X)
