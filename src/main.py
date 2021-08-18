import argparse

import ECAgent.Environments as env
import ECAgent.Visualization as vis

from VegetationModel import *
from Agents import *

# Default Decoder file path
default_path = './resources/decoder_file.json'


class EgyptModel(Model, IDecodable):

    def __init__(self, width: int, height: int, iterations: int,  heightmap: [[float]], water_map: [[float]],
                 soil_map: [[float]], cellSize: int, debug: bool = True):
        super().__init__(None)

        if debug: print('Creating Gridworld')

        self.debug = debug
        self.environment = env.GridWorld(width, height, self)
        self.cellSize = cellSize
        self.iterations = iterations

        if debug: print('Loading Heightmap')

        def elevation_generator_functor(pos, cells):
            return heightmap[pos[0]][pos[1]]

        def is_water_generator(pos, cells):
            return water_map[pos[0]][pos[1]] != 0.0

        def soil_generator(pos, cells):
            return soil_map[pos[0]][pos[1]]

        self.environment.addCellComponent('height', elevation_generator_functor)

        if debug:
            print('Generating Watermap')

        self.environment.addCellComponent('isWater', is_water_generator)

        # Generate slope data

        if debug:
            print('Generating Slopemap')

        def slopemap_generator(pos, cells):
            id = env.discreteGridPosToID(pos[0], pos[1], width)

            if cells['isWater'][id]:
                return 0
            else:
                maxSlope = 0.0  # Set base slope value
                heights = self.environment.cells['height']
                for neighbour in self.environment.getNeighbours(pos):
                    slopeVal = np.degrees(
                        np.arctan(
                            abs(heights[neighbour] - heights[id]) / self.cellSize))
                    maxSlope = max(maxSlope, slopeVal)

                # Set slopVal
                return maxSlope

        self.environment.addCellComponent('slope', slopemap_generator)

        if debug:
            print('Generating Soil Data')

        self.environment.addCellComponent('sand_content', soil_generator)

    @staticmethod
    def decode(params: dict):

        #Get heightmap
        from PIL import Image

        im = Image.open(params['heightmap_url']).convert('L')
        water_im = Image.open(params['water_mask']).convert('L')
        soil_mask = Image.open(params['soil_mask']).convert('L')

        width, height = params['img_width'], params['img_height']
        max_height, min_height = params['max_height'], params['min_height']
        height_diff = max_height - min_height

        heightmap = []
        water_map = []
        soil_map = []

        start_x, start_y = 0, 0

        if 'start_x' in params:
            start_x = params['start_x']

        if 'start_y' in params:
            start_y = params['start_y']

        for x in range(start_x, start_x + width):
            row = []
            water_row = []
            soil_row = []

            for y in range(start_y, start_y + height):
                row.append(min_height + (im.getpixel((x, y))/255.0 * height_diff))
                water_row.append(water_im.getpixel((x, y)))
                soil_row.append((soil_mask.getpixel((x, y))/255.0 * 100))

            heightmap.append(row)
            water_map.append(water_row)
            soil_map.append(soil_row)

        return EgyptModel(width, height, params['iterations'], heightmap,water_map, soil_map, params['cell_dim'])


class CellComponent(Component):

    def __init__(self, agent, model):
        super().__init__(agent, model)

        self.height = 0
        self.slope = 0
        self.isWater = False
        self.waterAvailability = 0


def parseArgs():
    """Create the EgyptModel Parser"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', help='Path to decoder json file.', default=default_path)
    parser.add_argument('-v', '--visualize', help='Will start a dash applet to display a summary of the simulation.',
                        action='store_true')
    parser.add_argument('-d', '--debug', help='Sets the model to debug mode. Output printed to terminal will be verbose',
                        action='store_true')

    return parser.parse_args()


layout_dict = {
        'height': 1000,
        'xaxis': dict(title="xCoord"),
        'yaxis': dict(title="yCoord")
    }

if __name__ == '__main__':

    parser = parseArgs()

    print("Creating Model...")
    model = JsonDecoder().decode(parser.file)

    model.debug = parser.debug

    # Run Initialization here
    if 'APS' in model.systemManager.systems.keys() and 'RAS' in model.systemManager.systems.keys():

        # Assign Random Positions to the Settlements
        for sID in model.environment[SettlementRelationshipComponent].settlements:

            while True:
                pos_x = model.random.randrange(0, model.environment.width)
                pos_y = model.random.randrange(0, model.environment.height)
                unq_id = env.discreteGridPosToID(pos_x, pos_y, model.environment.width)
                if model.environment.cells['isSettlement'][unq_id] == -1 and not model.environment.cells['isWater'][unq_id]:
                    model.environment[SettlementRelationshipComponent].settlements[sID].pos.append(unq_id)
                    model.environment.cells.at[unq_id, 'isSettlement'] = sID
                    break

        # Update num_households to APS
        model.systemManager.systems['APS'].num_households = len(model.environment.agents)

        # Assign Households to Settlements
        for agent in model.environment.getAgents():
            s = model.random.choice(model.environment[SettlementRelationshipComponent].settlements)  # Get ID
            model.environment[SettlementRelationshipComponent].add_household_to_settlement(agent, s.id)  # Add household

            # Set household position
            h_pos = model.environment.cells['pos'][model.environment[SettlementRelationshipComponent].settlements[s.id].pos[-1]]

            agent[PositionComponent].x = h_pos[0]
            agent[PositionComponent].y = h_pos[1]

        # Clean up empty settlements
        for sID in [s for s in model.environment[SettlementRelationshipComponent].settlements]:

            if len(model.environment[SettlementRelationshipComponent].settlements[sID].occupants) == 0:
                model.environment[SettlementRelationshipComponent].remove_settlement(sID)

    for i in range(model.iterations):
        print('Iteration: {} : {}%'.format(i, i/model.iterations * 100))
        model.systemManager.executeSystems()
        # print(model.systemManager.systems['GES'])

    if parser.visualize:
        webApp = vis.VisualInterface("Predynastic Egypt", model)

        heightArr = []
        vegetationArr = []
        moistureArr = []
        landOwnershipArr = []
        settlementOwnershipArr = []

        xdim = [i for i in range(model.iterations)]

        for x in range(model.environment.width):
            heightRow = []
            vegRow = []
            moistRow = []
            ownRow = []
            settleRow = []

            for y in range(model.environment.height):
                id = env.discreteGridPosToID(x, y, model.environment.width)
                heightRow.append(model.environment.cells['height'][id])
                vegRow.append(model.environment.cells['vegetation'][id])
                moistRow.append(model.environment.cells['moisture'][id])
                houseID = model.environment.cells['isOwned'][id]
                ownRow.append(str(houseID))
                settleRow.append(str(houseID if houseID == -1 else model.environment.getAgent(houseID)[HouseholdRelationshipComponent].settlementID))

            heightArr.append(heightRow)
            vegetationArr.append(vegRow)
            moistureArr.append(moistRow)
            landOwnershipArr.append(ownRow)
            settlementOwnershipArr.append(settleRow)

        webApp.addDisplay(vis.createGraph('height_heatmap', vis.createHeatMapGL(
            "Heightmap", heightArr,
            heatmap_kwargs={'colorbar': dict(title="Height(m)")},
            layout_kwargs=layout_dict)
                                          )
                          )

        webApp.addDisplay(vis.createGraph('vegetation_heatmap', vis.createHeatMapGL(
            "Final Vegetation Data", vegetationArr,
            heatmap_kwargs={'colorbar': dict(title="Vegetation(Kg)")},
            layout_kwargs=layout_dict)
                                          )
                          )

        webApp.addDisplay(vis.createGraph('vegetation_scatter', vis.createScatterGLPlot(
            'Mean Vegetation', [
                [xdim, model.systemManager.systems['collector'].records[1], {'name': 'Vegetation(Kg)'}]
            ], layout_kwargs=layout_dict
        )))

        webApp.addDisplay(vis.createGraph('moisture_heatmap', vis.createHeatMapGL(
            "Final Moisture Data", moistureArr,
            heatmap_kwargs={'colorbar': dict(title="Soil Moisture (mm)")},
            layout_kwargs=layout_dict)
                                          )
                          )

        webApp.addDisplay(vis.createGraph('moisture_scatter', vis.createScatterGLPlot(
            'Mean Soil Moisture', [
                [xdim, model.systemManager.systems['collector'].records[0], {'name': 'Moisture(mm)'}]
            ], layout_kwargs=layout_dict
        )))

        agent_col_records = model.systemManager.systems['agent_collector'].records
        pop_plots = {}

        for x in range(len(agent_col_records)):
            for key in agent_col_records[x]:
                if not key in pop_plots:
                    pop_plots[key] = {}
                    pop_plots[key]['xdim'] = [x]
                    pop_plots[key]['resources'] = [agent_col_records[x][key]['resources']]
                    pop_plots[key]['population'] = [agent_col_records[x][key]['population']]
                    pop_plots[key]['satisfaction'] = [agent_col_records[x][key]['satisfaction']]
                    pop_plots[key]['load'] = [agent_col_records[x][key]['load']]
                else:
                    pop_plots[key]['xdim'].append(x)
                    pop_plots[key]['resources'].append(agent_col_records[x][key]['resources'])
                    pop_plots[key]['population'].append(agent_col_records[x][key]['population'])
                    pop_plots[key]['satisfaction'].append(agent_col_records[x][key]['satisfaction'])
                    pop_plots[key]['load'].append(agent_col_records[x][key]['load'])

        webApp.addDisplay(vis.createGraph('resources_scatter', vis.createScatterGLPlot(
            'Household Resources', [
                [pop_plots[key]['xdim'], pop_plots[key]['resources'], {'name': ('Household %d' % key if isinstance(key, int) else key)}]
                for key in pop_plots], layout_kwargs=layout_dict
        )))

        webApp.addDisplay(vis.createGraph('load_scatter', vis.createScatterGLPlot(
            'Household Load', [
                [pop_plots[key]['xdim'], pop_plots[key]['load'],
                 {'name': ('Household %d' % key if isinstance(key, int) else key)}]
                for key in pop_plots], layout_kwargs=layout_dict
        )))

        webApp.addDisplay(vis.createGraph('population_scatter', vis.createScatterGLPlot(
            'Household Population', [
                [pop_plots[key]['xdim'], pop_plots[key]['population'],
                 {'name': ('Household %d' % key if isinstance(key, int) else key)}]
                for key in pop_plots], layout_kwargs=layout_dict
        )))

        webApp.addDisplay(vis.createGraph('satisfaction_scatter', vis.createScatterGLPlot(
            'Household Satisfaction', [
                [pop_plots[key]['xdim'], pop_plots[key]['satisfaction'],
                 {'name': ('Household %d' % key if isinstance(key, int) else key)}]
                for key in pop_plots], layout_kwargs=layout_dict
        )))

        action_records = model.systemManager.systems['action_collector'].records

        webApp.addDisplay(vis.createGraph('action_scatter', vis.createScatterGLPlot(
            'Household Actions', [
                [xdim, [ac['farm_count'] for ac in action_records], {'name': 'farm_count'}],
                [xdim, [ac['forage_count'] for ac in action_records], {'name': 'forage_count'}],
                [xdim, [ac['loan_count'] for ac in action_records], {'name': 'loan_count'}]]
            , layout_kwargs=layout_dict
        )))

        land_map = dict(
            source="./resources/Qena_Rescaled.png",
            xref="x",
            yref="y",
            x=0,
            y=3,
            sizex=model.environment.width,
            sizey=model.environment.height,
            sizing="stretch",
            opacity=1.0,
            layer="below")

        colors = [[0, 'rgba(0,0,255, 0)'], [0.1, 'rgba(0,255,0, 255)'], [0.6, 'rgba(255,0,0,255)'], [1.0, 'rgba(0,0,255, 255)']]

        fig_to_add = vis.createHeatMapGL(
            "Final Ownership Heatmap", landOwnershipArr,
            heatmap_kwargs={'colorbar': dict(title="Ownership (id)"), 'colorscale': colors},
            layout_kwargs=layout_dict)

        fig_to_add.add_layout_image(land_map)
        webApp.addDisplay(vis.createGraph('ownership_landmap', fig_to_add))

        webApp.addDisplay(vis.createGraph('settlement_landmap', vis.createHeatMapGL(
            "Final Settlement Heatmap", settlementOwnershipArr,
            heatmap_kwargs={'colorbar': dict(title="Settlement (id)")},
            layout_kwargs=layout_dict)
                                          )
                          )

        print('Number of Households: ' + str(len(model.environment.getAgents())))
        print('Number of Settlements: ' + str(len(model.environment[SettlementRelationshipComponent].settlements)))

        for agent in model.environment.getAgents():
            print(agent)

        webApp.app.run_server()