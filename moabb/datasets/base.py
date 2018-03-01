"""
Base class for a dataset
"""


class BaseDataset():
    """Base dataset"""

    def __init__(self, subjects, sessions_per_subject, events, name, interval, paradigm):
        self.subject_list = subjects
        self.n_sessions = sessions_per_subject
        self.event_id = events
        self.selected_events = None
        self.name = name
        self.interval = interval
        self.paradigm = paradigm

    def get_data(self, subjects=None, stack_sessions=False):
        """return data for a (list of) subject(s)
        If sessions are not stacked, return each session as a separate dataset"""
        data = []
        if type(subjects) is int:
            subjects = [subjects]
        elif subjects is None:
            subjects = self.subject_list
        for subject in subjects:
            if subject not in self.subject_list:
                raise ValueError('Invalid subject {:s} given'.format(subject))
            data.extend(self._get_single_subject_data(subject, stack_sessions))
        return data

    def _get_single_subject_data(self, subject, sessions):
        pass
