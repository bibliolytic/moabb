import matplotlib
matplotlib.use('Agg')
from .meta_analysis import rmANOVA, permutation_pairedttest
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sea
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

sea.set_style("whitegrid")
sea.set_context("paper")
colors = sea.color_palette("husl", 10)
sea.set_palette(colors)

highlight_color = '#8fff76'


def score_plot(data, p_threshold=0.05):
    '''
    Input:
        data: dataframe

    Out:
        ax: pyplot Axes reference
    '''
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111)
    stats = rmANOVA(data)
    sea.violinplot(data=data, y="score", x="dataset",
                   hue="pipeline", inner="point", cut=0, ax=ax, scale='width')
    ax.set_ylim([0.5, 1])
    xticks = ax.get_xticks()
    category_width = xticks[1]-xticks[0]
    dataset_xvals = dict(zip([x.get_text()
                              for x in ax.get_xticklabels()], xticks))
    sig_diff = []
    for dname, (f, p) in stats.items():
        if p <= p_threshold:
            sig_diff.append(dname)
            print('{}:{},{}'.format(dname, f, p))
            xloc = dataset_xvals[dname] - 0.45*category_width
            ax.add_patch(patches.Rectangle((xloc, 0.5),
                                           0.9*category_width,
                                           0.5,
                                           edgecolor='none',
                                           facecolor=highlight_color,
                                           alpha=0.3))

    ax.set_title('Scores per dataset and algorithm')
    return fig, sig_diff


def orderplot(ax, array, p_names, d_names, margin=0.01):
    '''
    ax: Axes object
    array: ndarray of objects, size len(p_names)*len(d_names)
           Order is same as order in plot. Elements are indices of p_names
    p_names: pipeline names
    d_names: dataset names
    margin: number of units separating all  elements of the plot
    '''
    from matplotlib.collections import PatchCollection

    def generate_squares(zeropt, color_indices, n_tiles, tilesize, margin):
        square_list = []
        for ind, c in enumerate(color_indices):
            row = int(ind / n_tiles)
            col = int(ind - row*n_tiles)
            bottom_left = np.array((col*tilesize+margin,
                                    (tilesize * (n_tiles - row - 1))+margin))
            bottom_left = bottom_left + zeropt
            square_list.append(patches.Rectangle(bottom_left,
                                                 tilesize-margin,
                                                 tilesize-margin))
        return square_list

    ax.grid(False)
    ax.set_xlim([0, len(d_names)])
    ax.set_ylim([0, len(p_names)])
    ax.set_xticks(0.5+np.arange(len(d_names)))
    ax.set_xticklabels(d_names)
    ax.set_yticks(0.5+np.arange(len(p_names)))
    ax.set_yticklabels([str(x) for x in np.arange(len(p_names), 0, -1)])
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Algorithm ordering')
    ax.set_title('Order of pipeline performances')
    
    sea.despine(ax=ax, offset=10, trim=False)
    squares_per_side = np.ceil(
        np.sqrt(np.array([len(x) for x in array[:]]).max()))
    square_side = 0.5/squares_per_side - 2*margin

    for row in range(array.shape[0]):
        for col in range(array.shape[1]):
            if len(array[row,col]) > 0:
                zeropt = np.array([col + 0.5, array.shape[0]-row - 0.5]) - 0.25
                objs = generate_squares(zeropt,
                                        array[row, col],
                                        squares_per_side,
                                        square_side,
                                        margin)
                p = PatchCollection(objs, facecolors=[colors[c] for c in array[row,col]])
                ax.add_collection(p)
    
    return ax


def ordering_plot(data, d_list, p_threshold=0.05):
    '''
    Input:
    data: df
    d_list: list of names for datasets with significant differences
    '''

    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111)
    pipelines = data['pipeline'].unique()
    datasets = d_list
    array = np.ndarray((len(pipelines),len(datasets)),dtype='object')
    for ind_d, d in enumerate(datasets):
        for ind_p in range(len(pipelines)):
            array[ind_p,ind_d] = []
        reduced_data = data[data['dataset'] == d]
        scores = np.array([reduced_data[reduced_data['pipeline'] ==p]['score'] for p in pipelines])
        compare_order = np.argsort(scores.mean(axis=1))
        ordinal = 0
        losers = []
        for ind_comp in range(len(compare_order)-1,0,-1):
            current = losers + [compare_order[ind_comp]]
            pval = permutation_pairedttest(scores[compare_order[ind_comp-1]],
                                             scores[compare_order[ind_comp]])
            if pval < p_threshold:
                array[ordinal,ind_d].extend(current)
                losers = []
            else:
                losers.append(compare_order[ind_comp])
            ordinal += 1
        array[-1,ind_d].extend(losers)
        array[-1,ind_d].append(compare_order[0])
    orderplot(ax, array, pipelines, datasets)
    amounts =  []
    for ind_c in range(len(datasets)):
        for ind_r in range(array.shape[0] - 1):
            if len(array[ind_r,ind_c]) != 0:
                   amounts.extend([pipelines[i] for i in array[ind_r, ind_c]])
                   break
    amounts = np.array(amounts)
    bar_fig = plt.figure()
    ax = bar_fig.add_subplot(111)
    ax.bar(np.arange(1,len(pipelines)+1),
           np.array([np.array(amounts==p).sum() for p in pipelines]),
           tick_label=pipelines)
    ax.set_xlabel('Pipeline')
    ax.set_ylabel('# of times')
    ax.set_title('How often each pipeline performed best over datasets')
                

    return fig, bar_fig



def time_line_plot(data):
    '''
    plot data entries per timepoint
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    data['n_entries'] = data['samples']*data['channels']

    for p in data['pipeline'].unique():
        ax.scatter(data[data['pipeline'] == p]['n_entries'],
                   data[data['pipeline'] == p]['time'])
    ax.legend(data['pipeline'].unique())
    ax.set_xlabel('Entries in training matrix')
    ax.set_ylabel('Time to fit decoding model')
    return fig


def time_plot(data):
    '''
    Input:
    data: dataframe

    Out:
    ax: pyplot Axes reference
    '''
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for p in data['pipeline'].unique():
        ax.scatter(data[data['pipeline'] == p]['channels'],
                   data[data['pipeline'] == p]['samples'],
                   data[data['pipeline'] == p]['time'])
    ax.legend(data['pipeline'].unique())

    ax.set_xlabel('Number of channels')
    ax.set_ylabel('Training samples')
    ax.set_zlabel('Time to fit decoding model')

    return fig
