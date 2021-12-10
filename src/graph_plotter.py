import argparse
import json
import matplotlib.pyplot as pyplot
import numpy as np

from CythonFunctions import CGlobalEnvironmentCurves


def scenario_to_curve(scenario, i):

    if i > 1000:
        return 1.0

    if scenario == 'scenario1':
        return 1.0
    elif scenario == 'scenario2':
        return CGlobalEnvironmentCurves.linear_lerp(0., 1., i)
    elif scenario == 'scenario3':
        return CGlobalEnvironmentCurves.cosine_lerp(0., 1., i, 16)
    else:
        return CGlobalEnvironmentCurves.linear_modified_dsinusoidal_lerp(0., 1., i, 16, 1, 1)


def get_scenario_data(runs: {}, scenario):

    placeholder = np.zeros((2000, len(runs)), dtype=float)

    index = 0
    for seed in runs:
        for i in range(len(runs[seed]['population']['total'])):
            placeholder[i][index] = runs[seed]['population']['total'][i]

        index += 1

    data = np.zeros((6, 2000), dtype=float)

    for i in range(2000):
        data[0][i] = np.mean(placeholder[i])
        data[1][i] = np.std(placeholder[i])
        data[2][i] = (data[1][i] / data[0][i]) * 100.0
        lower, upper = data[0][i] - data[1][i],  data[0][i] + data[1][i]
        data[3][i] = len([s for s in placeholder[i] if lower < s < upper]) / 50.0
        data[4][i] = len([s for s in placeholder[i] if lower - data[1][i] < s < upper + data[1][i]]) / 50.0

        # Resistance Metric
        if i != 0:
            data[5][i] = (data[0][i] - data[0][0]) / data[0][i] * 100.0

    return data


def write_plot(agent_types: [], scenario, filename, data, title: str, index: int, x_axis: str, y_axis: str,
               legend: str = 'lower right'):

    fig, ax = pyplot.subplots()
    ax.set_title(title)
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)

    iterations = np.arange(2000)

    for agent_type in agent_types:
        ax.plot(iterations, data[agent_type][scenario][index], label=agent_type)
        if index == 0:
            color = [l for l in ax.lines if l._label == agent_type][0]._color
            ax.fill_between(iterations, data[agent_type][scenario][0] - data[agent_type][scenario][1],
                        data[agent_type][scenario][0] + data[agent_type][scenario][1], color=color, alpha=0.2)

    ax.legend(loc=legend)
    ax.set_aspect('auto')

    fig.savefig(filename)
    pyplot.close(fig)


def main():
    # Process the params
    print("Parsing arguments...")
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='The file containing the processed data', type=str)
    parser.add_argument('store_loc', help='location to store images', type=str)
    parser.add_argument('-v', '--verbose', help='Will print out informative information to the terminal.',
                        action='store_true')

    parser = parser.parse_args()

    data = {}

    with open(parser.file) as json_file:
        data = json.load(json_file)

    reordered_data = {}

    for agent_type in data:
        reordered_data[agent_type] = {}
        for scenario in data[agent_type]:
            # Get all values from all runs
            reordered_data[agent_type][scenario] = get_scenario_data(data[agent_type][scenario], scenario)

    for scenario in data['Traditional']:
        write_plot([a for a in data], scenario, '%s/population_%s' % (parser.store_loc, scenario), reordered_data,
                   'Total Population of Agent Types for %s averaged over 50 simulation runs.' % scenario,
                   0, 'iterations', 'Population')
        write_plot([a for a in data], scenario, '%s/SD_%s' % (parser.store_loc, scenario), reordered_data,
                   'SD of Population of Agent Types for %s averaged over 50 simulation runs.' % scenario,
                   1, 'iterations', 'SD')
        write_plot([a for a in data], scenario, '%s/RSD_%s' % (parser.store_loc, scenario), reordered_data,
                   'RSD of Population of Agent Types for %s averaged over 50 simulation runs.' % scenario,
                   2, 'iterations', 'RSD(%)')
        write_plot([a for a in data], scenario, '%s/onestd_%s' % (parser.store_loc, scenario), reordered_data,
                   '%s of simulation runs within 1 STD of the mean\nfor %s averaged over 50 simulation runs.' % ('%', scenario),
                   3, 'iterations', '%')
        write_plot([a for a in data], scenario, '%s/twostd_%s' % (parser.store_loc, scenario), reordered_data,
                   '%s of simulation runs within 2 STD of the mean\nfor %s averaged over 50 simulation runs.' % ('%', scenario),
                   4, 'iterations', '%')
        write_plot([a for a in data], scenario, '%s/resistance_%s' % (parser.store_loc, scenario), reordered_data,
                   'Relative Resistance of population \nfor %s averaged over 50 simulation runs.' % scenario,
                   5, 'iterations', '%')


if __name__ == '__main__':
    main()
