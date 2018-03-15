
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from moabb.pipelines.single_pass import LogVariance, FM
from sklearn.pipeline import make_pipeline, FeatureUnion
import numpy as np

parameters = {'C': np.logspace(-2, 2, 10)}
clf = GridSearchCV(SVC(kernel='linear'), parameters)
features = FeatureUnion([
    ('AM', LogVariance()),
    ('FM', FM())])
pipe = make_pipeline(features, clf)

# this is what will be loaded
PIPELINE = {'name': 'AMFM + optSVM',
            'paradigms': ['LeftRightImagerySinglePass'],
            'pipeline': pipe}
