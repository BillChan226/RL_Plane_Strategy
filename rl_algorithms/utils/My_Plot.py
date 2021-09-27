import seaborn as sns
import pandas as pd
import matplotlib
from ipdb import set_trace as tt
# matplotlib.use('AGG')  # 或者PDF, SVG或PS
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import json
import os
import os.path as osp
import numpy as np

DIV_LINE_WIDTH = 50

# Global vars for tracking and labeling data at load time.
exp_idx = 0
units = dict()

def plot_data_ours(data, environments, xaxis='Epoch', value="AverageEpRet", condition="Condition1", smooth=1, **kwargs):
    # plot_main

    if smooth > 1:
        """
        smooth data with moving window average.
        that is,
            smoothed_y[t] = average(y[t-k], y[t-k+1], ..., y[t+k-1], y[t+k])
        where the "smooth" param is width of that window (2k+1)
        """
        y = np.ones(smooth)
        for datum in data:
            x = np.asarray(datum[value])
            z = np.ones(len(x))
            smoothed_x = np.convolve(x,y,'same') / np.convolve(z,y,'same')
            datum[value] = smoothed_x
    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)
        try:
            datastd = data.groupby(['Epoch', 'Condition1', 'Condition3']).std().reset_index()
            datamean = data.groupby(['Epoch', 'Condition1', 'Condition3']).mean().reset_index()
        except KeyError:
            pass

    plot_data20 = {}
    plot_data50 = {}
    for e in environments:
        std = datastd[datastd.Condition1==e][datastd.Condition3==str(20)].AverageEpRet.tolist()
        mean = datamean[datamean.Condition1==e][datamean.Condition3==str(20)].AverageEpRet.tolist()
        plot_data20[e] = np.array([std, mean])
        std = datastd[datastd.Condition1 == e][datastd.Condition3 == str(50)].AverageEpRet.tolist()
        mean = datamean[datamean.Condition1 == e][datamean.Condition3 == str(50)].AverageEpRet.tolist()
        plot_data50[e] = np.array([std, mean])

    return plot_data20, plot_data50

def data_with_suffix(root, suffix, environments, axis='TotalEnvInteracts'):
    if root[-1] != '/':
        root += "/"
    logdirs = [root for env in environments]
    #logdirs = [root+env for env in environments]
    res = {}
    for env, dir in zip(environments, logdirs):
        print('dir',dir)
        df_env = pd.concat(get_datasets(dir), ignore_index=True)
        RR = df_env.groupby(axis)
        std = df_env.groupby(axis).std().reset_index().AverageEpRet
        df_mean = df_env.groupby(axis).mean().reset_index()
        mean = df_mean.AverageEpRet
        xaxis = df_mean[axis]
        res[env] = np.array([xaxis, mean, std])
    return res


def get_datasets(logdir, condition=None):
    """
    Recursively look through logdir for output files produced by
    spinup.logx.Logger. 

    Assumes that any file "progress.txt" is a valid hit. 
    """
    global exp_idx
    global units
    if condition and '-' in condition:
        condition, condition3 = condition.split("-")
    else:
        condition3 = ""
    datasets = []
    for root, _, files in os.walk(logdir):
        if 'progress.txt' in files:
            exp_name = None
            try:
                config_path = open(os.path.join(root,'config.json'))
                config = json.load(config_path)
                if 'exp_name' in config:
                    exp_name = config['exp_name']
            except:
                print('No file named config.json')
            condition1 = condition or exp_name or 'exp'
            condition2 = condition1 + '-' + str(exp_idx)
            exp_idx += 1
            if condition1 not in units:
                units[condition1] = 0
            unit = units[condition1]
            units[condition1] += 1
            exp_data = pd.read_table(os.path.join(root, 'progress.txt'))
            performance = 'AverageTestEpRet' if 'AverageTestEpRet' in exp_data else 'AverageEpRet'
            exp_data.insert(len(exp_data.columns),'Unit',unit)
            exp_data.insert(len(exp_data.columns),'Condition1',condition1)
            exp_data.insert(len(exp_data.columns),'Condition2',condition2)
            exp_data.insert(len(exp_data.columns),'Condition3', condition3)
            exp_data.insert(len(exp_data.columns),'Performance',exp_data[performance])
            datasets.append(exp_data)
           # print(exp_data)
    return datasets


def get_all_datasets(all_logdirs, legend=None, select=None, exclude=None):
    """
    For every entry in all_logdirs,
        1) check if the entry is a real directory and if it is, 
           pull data from it; 

        2) if not, check to see if the entry is a prefix for a 
           real directory, and pull data from that.
    """
    logdirs = []
    for logdir in all_logdirs:
        if osp.isdir(logdir) and logdir[-1]==os.sep:
            logdirs += [logdir]
        else:
            basedir = osp.dirname(logdir)
            fulldir = lambda x : osp.join(basedir, x)
            prefix = logdir.split(os.sep)[-1]
            listdir= os.listdir(basedir)
            logdirs += sorted([fulldir(x) for x in listdir if prefix in x])

    """
    Enforce selection rules, which check logdirs for certain substrings.
    Makes it easier to look at graphs from particular ablations, if you
    launch many jobs at once with similar names.
    """
    if select is not None:
        logdirs = [log for log in logdirs if all(x in log for x in select)]
    if exclude is not None:
        logdirs = [log for log in logdirs if all(not(x in log) for x in exclude)]

    # Verify logdirs
    print('Plotting from...\n' + '='*DIV_LINE_WIDTH + '\n')
    for logdir in logdirs:
        print(logdir)
    print('\n' + '='*DIV_LINE_WIDTH)

    # Make sure the legend is compatible with the logdirs
    assert not(legend) or (len(legend) == len(logdirs)), \
        "Must give a legend title for each set of experiments."

    # Load data from logdirs
    data = []
    if legend:
        for log, leg in zip(logdirs, legend):
            data += get_datasets(log, leg)
    else:
        for log in logdirs:
            data += get_datasets(log)
    return data

def plot_data(data_to_plot, environments):
    num_env = len(environments)
    legends = data_to_plot.keys()
    # colors = ['red', 'blue', 'teal', 'green', 'lightskyblue']
    colors = ['blue', 'green']
    figure = plt.figure(figsize=(30, 5))

    def million_formatter(x, pos):
        return '%.1fM' % (x * 1e-6)
    formatter = FuncFormatter(million_formatter)
    #axs = [plt.subplot(161 ), plt.subplot(162 ), plt.subplot(163),
          # plt.subplot(164 ), plt.subplot(165 ), plt.subplot(166 )]
    axs = [plt.subplot(111 )]
    font = font1 = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': 20,
        }

    for i, ax in enumerate(axs):
        ax.set_facecolor((0.95, 0.95, 0.95))
        ax.xaxis.set_major_formatter(formatter)
        ax.set_xlabel(environments[i], fontdict=font)
        ppo = data_to_plot['WithDemon'][environments[i]][1]
        #max_PPO = max(ppo)
        #z = 0
        #while ppo[z] < 0.2 * max_PPO:
        #  z+=1
        # print(z)
        our = data_to_plot["NoDemon"][environments[i]][1]
        #mm = 0
        #while our[mm] < 0.2 * max_PPO:
        #  mm += 1
        # print(mm)
        #print(environments[i], mm / z)

        ax.grid(color='white', linestyle='-', linewidth=2, alpha=0.6)
        for j, k in enumerate(data_to_plot.keys()):
            method = data_to_plot[k]
            x = method[environments[i]][0]
            y = method[environments[i]][1]
            std = method[environments[i]][2]
            cut = 10000
            x, y ,std = x[:cut], y[:cut], std[:cut]
            ax.plot(x, y, color=colors[j], linewidth=2, label=k)
            ax.fill_between(x, y+std, y-std, facecolor=colors[j], alpha=0.3)
        ax.legend(ncol=3, fontsize=16)

    plt.plot()
    plt.show()





def make_plots(root_path, environments, legend=None, xaxis=None, values=None, count=False,
               font_scale=1.5, smooth=1, select=None, exclude=None, estimator='mean'):
    # all_data_dict = {}
    all_env_path = []
    legend = []
    for e in environments:
        # all_data_dict[e] = {}
        for k in [100, 200, 500]:
            acdf_suffix = 'ACDF_pi{}_vf{}'.format(k, k)
            all_env_path.append(root_path+e+acdf_suffix)
            legend.append(e+'-'+str(k))
        # all_data_dict[e][k] = get_all_datasets(env_path)
    suffix = [] #, 'ACDF_pi50_vf50']


    data_to_plot = {}


    # data_to_plot['OursWithReward5'] = data_with_suffix(root_path, 'ACDF_pi50_vf50', environments)
    #data_to_plot['WithDemon'] = data_with_suffix(root_path + 'PPO_With_Demons/', suffix='', environments=environments)
    data_to_plot['WithDemon'] = data_with_suffix(root_path + 'PPO_InMoov_With_Demons/', suffix='', environments=environments)
    # data_to_plot['Ours'] = data_with_suffix(root_path, 'ACDF_pi20_vf20', environments)
    #data_to_plot['NoDemon'] = data_with_suffix(root_path + 'PPO_No_Demons/', suffix='', environments=environments)
    data_to_plot['NoDemon'] = data_with_suffix(root_path + 'PPO_Inmoov_No_Demons/', suffix='', environments=environments)
    # data_to_plot['Ours50'] = data_with_suffix(root_path + 'no_reward/', suffix='ACDF_pi50_vf50',environments=environments)
    # data_to_plot['20'] = data_with_suffix(root_path + 'no_reward/', suffix='ACDF_pi50_vf50', environments=environments)
    # data_to_plot['50'] = data_with_suffix(root_path, suffix='ACDF_pi50_vf50', environments=environments)
    # data_to_plot['100'] = data_with_suffix(root_path, suffix='ACDF_pi20_vf20', environments=environments)
    # data_to_plot['200'] = data_with_suffix(root_path, suffix='ACDF_pi200_vf200', environments=environments)
    # data_to_plot['500'] = data_with_suffix(root_path, suffix='ACDF_pi500_vf500', environments=environments)
    # import pickle
    # with open("data_to_plot.pickle", "wb") as fp:
    #     pickle.dump(data_to_plot, fp, protocol=pickle.HIGHEST_PROTOCOL)
    # with open("data_to_plot.pickle", "rb") as fp:
    #     data_to_plot = pickle.load(fp)


    table_dict = {}
    for m in data_to_plot:
        print(m)
        table_dict[m] = []
        for env in data_to_plot[m]:
            print(env)
            tmp = data_to_plot[m][env]
            # table_dict[m].append((tmp[1][0], tmp[2][0]))
            index = np.argmax(tmp[1])

            table_dict[m].append((tmp[1][index], tmp[2][index]))

    df = pd.DataFrame(table_dict)
    df.to_excel('high_without_reward_result.xlsx')
    plot_data(data_to_plot, environments)

    # plot_with_suffix(root_path, suffix, environments)
    # data = get_all_datasets(all_env_path, legend)
    # # data = get_all_datasets(all_logdirs, legend, select, exclude)
    # values = values if isinstance(values, list) else [values]
    # condition = 'Condition2' if count else 'Condition1'
    # estimator = getattr(np, estimator)      # choose what to show on main curve: mean? max? min?
    # for value in values:
    #     plt.figure()
    #     plot20, plot50 = plot_data_ours(data, environments, xaxis=xaxis, value=value, condition=condition, smooth=smooth, estimator=estimator)
    # plt.show()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', nargs='*')
    parser.add_argument('--legend', '-l', nargs='*')
    parser.add_argument('--xaxis', '-x', default='TotalEnvInteracts')
    parser.add_argument('--value', '-y', default='Performance', nargs='*')
    parser.add_argument('--count', action='store_true')
    parser.add_argument('--smooth', '-s', type=int, default=1)
    parser.add_argument('--select', nargs='*')
    parser.add_argument('--exclude', nargs='*')
    parser.add_argument('--est', default='mean')
    args = parser.parse_args()
    #environments = ['Hopper', 'Walker2d', 'Ant', 'Humanoid', 'Swimmer', 'HalfCheetah']
    environments = ['Ant']
    environments.sort()
    #args.logdir='/home/tete/work/new/data_plot'
   # root_path = args.logdir[-1] if args.logdir[-1][-1] == '/' else args.logdir[-1] + '/'
    #root_path="/home/tete/work/new/data_plot/K_means/"
    root_path="/home/tete/work/new/data/"

    make_plots(root_path, environments, args.legend, args.xaxis, args.value, args.count,
               smooth=args.smooth, select=args.select, exclude=args.exclude,
               estimator=args.est)






if __name__ == "__main__":
    main()
