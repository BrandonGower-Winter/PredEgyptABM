import argparse
import json
import os
import statistics

import numpy as np

from Progress import progress


def gini(x):

    # Mean Absolute Difference
    mad = np.abs(np.subtract.outer(x, x)).mean()
    # Relative Mean Absolute difference
    rmad = mad / np.mean(x)

    return 0.5 * rmad


def get_json_iteration(filename: str) -> int:
    return int(filename[filename.index('_')+1:-5])


def load_json_files(folder_path: str, sort: bool = True, key=get_json_iteration) -> []:

    json_snapshots = []

    for root, _, files in os.walk(folder_path, topdown=True):

        json_files = [f for f in files if f[-4:] == 'json']
        if sort:
            json_files.sort(key=key)

        for file in json_files:
            with open(os.path.join(root, file)) as json_file:
                json_snapshots.append(json.load(json_file))

    return json_snapshots


def generate_composite_val(prop: str, snapshot: dict, comp_func, sort: bool = False):

    ls = [agent[prop] for agent in snapshot]

    if len(ls) == 0:
        return 0

    if sort:
        ls.sort()

    return comp_func(ls)


def get_composite_property_as_dict(snapshots: [[dict]], property: str, comp_funcs: [(str, any)],
                                   over_range: (int, int) = (0, -1), sort: bool = False) -> dict:

    prop_dict = {}

    over_range = over_range if over_range[1] != -1 else (over_range[0], len(snapshots))

    for i in range(over_range[0], over_range[1]):
        for func in comp_funcs:

            val = generate_composite_val(property, snapshots[i], func[1], sort)

            if func[0] in prop_dict:
                prop_dict[func[0]].append(val)
            else:
                prop_dict[func[0]] = [val]

    return prop_dict


def main():

    # Process the params
    print("Parsing arguments...")
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='The path to the folder containing all of the generated data', type=str)
    parser.add_argument('-v', '--verbose', help='Will print out informative information to the terminal.',
                    action='store_true')

    parser = parser.parse_args()

    if not os.path.isdir(parser.path):
        print('Please make sure path specified is a directory...')
        return

    # Get all settlement files
    print("Loading all json files...")

    # For all folders in path

    # File Format: agent_type/scenario/seed/resources+population
    to_write = {}

    for agent_type in [d for d in os.listdir(parser.path) if os.path.isdir(os.path.join(parser.path, d))]:
        print('Agent Type: %s:' % agent_type)
        to_write[agent_type] = {}
        agent_path = os.path.join(parser.path, agent_type)

        scenarios = [s for s in os.listdir(agent_path) if os.path.isdir(os.path.join(agent_path, s))]
        print('\t- Found %s scenarios...' % len(scenarios))
        for scenario in scenarios:
            to_write[agent_type][scenario] = {}
            print('\t- Scenario: %s' % scenario)
            scenario_path = os.path.join(agent_path, scenario)

            runs = [r for r in os.listdir(scenario_path) if os.path.isdir(os.path.join(scenario_path, r))]
            run_len = len(runs)
            print('\t\t- Found %s Simulation Runs...' % run_len)
            for i in range(run_len):
                progress(i, run_len)
                to_write[agent_type][scenario][runs[i]] = {}
                # Get all agent json files in this simulation run
                agent_snapshots = load_json_files(str(scenario_path) + '/' + runs[i] + '/agents')

                to_write[agent_type][scenario][runs[i]]['resources'] = get_composite_property_as_dict(agent_snapshots, 'resources',
                                                                 [('mean', statistics.mean),
                                                                  ('median', statistics.median),
                                                                  ('min', min),
                                                                  ('max', max),
                                                                  ('total', sum),
                                                                  ('gini', gini)], sort=True)

                to_write[agent_type][scenario][runs[i]]['population'] = get_composite_property_as_dict(agent_snapshots, 'occupants',
                                               [('mean', statistics.mean),
                                                ('median', statistics.median),
                                                ('min', min),
                                                ('max', max),
                                                ('total', sum)], sort=True)

            print()
    print('Writing data to output file: %s:' % (parser.path + 'processed_agents.json'))
    with open(parser.path + 'processed_agents.json', 'w') as outfile:
        json.dump(to_write, outfile, indent=4)


if __name__ == '__main__':
    main()
