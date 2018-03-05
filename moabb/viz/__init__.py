import os
import platform
from datetime import datetime

import h5py
import numpy as np
import pandas as pd
import inspect
import shutil

from . import plotting as plt


class Results:
    '''Class to hold results from the evaluation.evaluate method. Appropriate test
    would be to ensure the result of 'evaluate' is consistent and can be
    accepted by 'results.add'

    Saves dataframe per pipeline and can query to see if particular subject has
    already been run

    '''

    def __init__(self, evaluation_class, paradigm_class, suffix='', overwrite=False):
        """
        class that will abstract result storage
        """
        import moabb.utils as ut
        from moabb.contexts.base import BaseParadigm, BaseEvaluation
        assert issubclass(evaluation_class, BaseEvaluation)
        assert issubclass(paradigm_class, BaseParadigm)
        self.mod_dir = os.path.dirname(os.path.abspath(inspect.getsourcefile(ut)))
        self.filepath = os.path.join(self.mod_dir, 'results', paradigm_class.__name__,
                                     evaluation_class.__name__, 'results{}.hdf5'.format('_'+suffix))
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        self.filepath = self.filepath
        if overwrite and os.path.isfile(self.filepath):
            os.remove(self.filepath)
        if not os.path.isfile(self.filepath):
            with h5py.File(self.filepath, 'w') as f:
                f.attrs['create_time'] = np.string_(
                    '{:%Y-%m-%d, %H:%M}'.format(datetime.now()))

    def add(self, pipeline_dict):
        def to_list(d):
            if type(d) is dict:
                return [d]
            elif type(d) is not list:
                raise ValueError('Results are given as neither dict nor list but {}'.format(
                    type(d).__name__))
            else:
                return d
        with h5py.File(self.filepath, 'r+') as f:
            for name, data_dict in pipeline_dict.items():
                if name not in f.keys():
                    # create pipeline main group if nonexistant
                    f.create_group(name)
                ppline_grp = f[name]
                dlist = to_list(data_dict)
                d1 = dlist[0]
                dname = d1['dataset'].name
                if dname not in ppline_grp.keys():
                    # create dataset subgroup if nonexistant
                    dset = ppline_grp.create_group(dname)
                    dset.attrs['n_subj'] = len(d1['dataset'].subject_list)
                    dset.attrs['n_sessions'] = d1['dataset'].n_sessions
                    dt = h5py.special_dtype(vlen=str)
                    dset.create_dataset('id', (0,), dtype=dt, maxshape=(None,))
                    dset.create_dataset('data', (0, 3), maxshape=(None, 3))
                    dset.attrs['channels'] = d1['n_channels']
                    dset.attrs.create(
                        'columns', ['score', 'time', 'samples'], dtype=dt)
                dset = ppline_grp[dname]
                for d in dlist:
                    # add id and scores to group
                    length = len(dset['id']) + 1
                    dset['id'].resize(length, 0)
                    dset['data'].resize(length, 0)
                    dset['id'][-1] = str(d['id'])
                    dset[
                        'data'][-1, :] = np.asarray([d['score'], d['time'], d['n_samples']])

    def to_dataframe(self):
        df_list = []
        with h5py.File(self.filepath, 'r') as f:
            for name, p_group in f.items():
                for dname, dset in p_group.items():
                    array = np.array(dset['data'])
                    ids = np.array(dset['id'])
                    df = pd.DataFrame(array, columns=dset.attrs['columns'])
                    df['id'] = ids
                    df['channels'] = dset.attrs['channels']
                    df['n_sessions'] = dset.attrs['n_sessions']
                    df['dataset'] = dname
                    df['pipeline'] = name
                    df_list.append(df)
        return pd.concat(df_list, ignore_index=True)

    def not_yet_computed(self, pipeline_dict, dataset, subj):
        def already_computed(p, d, s):
            with h5py.File(self.filepath, 'r') as f:
                if p not in f.keys():
                    return False
                else:
                    pipe_grp = f[p]
                    if d.name not in pipe_grp.keys():
                        return False
                    else:
                        dset = pipe_grp[d.name]
                        return (str(s) in dset['id'])
        return {k: pipeline_dict[k] for k in pipeline_dict.keys() if not already_computed(k, dataset, subj)}


def analyze(results, out_path, name='analysis', suffix=''):
    '''Given a results object (or the location for one), generates a folder with
    results and a dataframe of the exact data used to generate those results, as
    well as introspection to return information on the computer

    In:
    out_path: location to store analysis folder

    results: Obj/tuple; 

    path: string/None

    Either path or results is necessary

    '''
    ### input checks ###
    if type(results) is not Results:
        res = Results(*results, suffix=suffix)
    if type(out_path) is not str:
        raise ValueError('Given out_path argument is not string')
    elif not os.path.isdir(out_path):
        raise IOError('Given directory does not exist')
    else:
        analysis_path = os.path.join(out_path, name)
        if os.path.isdir(analysis_path):
            print("Analysis already exists; overwriting")
            shutil.rmtree(analysis_path)

    os.makedirs(analysis_path, exist_ok=True)
    # TODO: no good cross-platform way of recording CPU info?
    with open(os.path.join(analysis_path, 'info.txt'), 'a') as f:
        f.write(
            'Date: {:%Y-%m-%d}\n Time: {:%H:%M}\n'.format(datetime.now(), datetime.now()))
        f.write('System: {}\n'.format(platform.system()))
        f.write('CPU: {}\n'.format(platform.processor()))

    res = results

    data = res.to_dataframe()
    data.to_csv(os.path.join(analysis_path, 'data.csv'))
    fig, sig = plt.score_plot(data)
    fig.savefig(os.path.join(analysis_path, 'scores.pdf'))
    plt.time_line_plot(data).savefig(os.path.join(analysis_path, 'time2d.pdf'))
    if len(sig) != 0:
        order, bar = plt.ordering_plot(data, sig)
        order.savefig(os.path.join(analysis_path, 'ordering.pdf'))
        bar.savefig(os.path.join(analysis_path, 'summary.pdf'))
