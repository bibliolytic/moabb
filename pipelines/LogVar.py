
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import moabb.pipelines.multi_pass as mp
from sklearn.pipeline import make_pipeline
import numpy as np

parameters = {'C': np.logspace(-2, 2, 10)}
clf = GridSearchCV(SVC(kernel='linear'), parameters)
pipe = make_pipeline(mp.MultibandCovariances(estimator='oas'), clf)

# this is what will be loaded
PIPELINE = {'name': 'FBCSP + optSVM',
            'paradigms': ['LeftRightImageryMultiPass'],
            'pipeline': pipe}
