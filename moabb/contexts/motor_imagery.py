"""Motor Imagery contexts"""

import numpy as np
from .base import BaseImageryParadigm
from mne import Epochs, find_events
from mne.epochs import concatenate_epochs, equalize_epoch_counts
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut, KFold

class BaseMotorImagery(BaseImageryParadigm):
    """Base Motor imagery context


    Parameters
    ----------
    datasets : List of Dataset instances.
        List of dataset instances on which the pipelines will be evaluated.
    pipelines : Dict of pipelines instances.
        Dictionary of pipelines. Keys identifies pipeline names, and values
        are scikit-learn pipelines instances.
    fmin : float | None, (default 7.)
        Low cut-off frequency in Hz. If None the data are only low-passed.
    fmax : float | None, (default 35)
        High cut-off frequency in Hz. If None the data are only high-passed.

    See Also
    --------
    MotorImageryTwoClasses
    """

    def __init__(self, pipelines, evaluator, datasets=None, fmin=7, fmax=35, **kwargs):
        super().__init__(pipelines, evaluator, datasets, fmin=fmin, fmax=fmax, **kwargs)


class ImageryNClass(BaseMotorImagery):
    """Imagery for multi class classification
    
    Returns n-class imagery results, visualization agnostic but forces all
    datasets to have exactly n classes. Uses 'accuracy' as metric

    """

    def __init__(self, pipelines, evaluator, n_classes , **kwargs):
        self.n_classes = n_classes
        super().__init__(pipelines, evaluator, **kwargs)

    def verify(self, d):
        super().verify(d)
        assert len(d.event_id) < self.n_classes, '{} is not not enough classes for {} class'.format(len(d.event_id), self.n_classes)
        if d.selected_events is None or len(d.selected_events) == 0:
            print('Randomly choosing {} events'.format(self.n_classes))
            keep = {}
            for k in d.event_id.keys():
                if len(keep) < self.n_classes:
                    keep[k] = d.event_id[k]
            d.selected_events = keep

    @property
    def scoring(self):
        return 'accuracy'


class LeftRightImagery(BaseMotorImagery):
    """Motor Imagery for left hand/right hand classification

    Metric is 'roc_auc'

    """

    def verify(self, d):
        events = ['left_hand','right_hand']
        super().verify(d)
        assert  set(events) <= set(d.event_id.keys())
        d.selected_events = dict(zip(events, [d.event_id[s] for s in events]))

    @property
    def scoring(self):
        return 'roc_auc'
