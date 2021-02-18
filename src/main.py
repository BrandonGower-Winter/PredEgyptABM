import numpy as np
import ECAgent.Environments as env

from VegetationModel import *


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

        for x in range(width):
            row = []
            water_row = []
            soil_row = []

            for y in range(height):
                row.append(min_height + (im.getpixel((x,y))/255.0 * height_diff))
                water_row.append(water_im.getpixel((x,y)))
                soil_row.append((soil_mask.getpixel((x,y))/255.0 * 100))

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


print("Creating Model...")
model = JsonDecoder().decode('../resources/decoder_file.json')

if __name__ == '__main__':

    for i in range(model.iterations):
        model.systemManager.executeSystems()