import math
import numpy as np

from ECAgent.Core import *
from ECAgent.Environments import *
from ECAgent.Decode import *


def lerp(start, end, percentage):
    return start + (end - start) * percentage


class GlobalEnvironmentComponent(Component):

    def __init__(self, agent, model: Model, start_temp: [int], end_temp: [int], start_rainfall: [int],
                 end_rainfall: [int], start_flood: [int], end_flood: [int], soil_depth: float):
        super().__init__(agent, model)

        self.start_temp = start_temp
        self.end_temp = end_temp
        self.start_rainfall = start_rainfall
        self.end_rainfall = end_rainfall
        self.start_flood = start_flood
        self.end_flood = end_flood

        self.soil_depth = soil_depth

        self.temp = []
        self.rainfall = []
        self.flood = -1


class SoilMoistureComponent(Component):

    def __init__(self, agent, model: Model, L: int, N: int, I: float):
        super().__init__(agent, model)

        self.L = L
        self.N = N
        self.I = I  # Heat Index

        # Get avg water cell height
        isWaterArr = model.environment.cells['isWater'].tolist()
        sum = 0.0
        count = 0
        for x in range(len(isWaterArr)):
            if isWaterArr[x]:
                sum += model.environment.cells['height'][x]
                count += 1

        self.avgWaterHeight = sum / count if count != 0 else 0


class GlobalEnvironmentSystem(System, IDecodable):

    def __init__(self, id: str, model: Model, start_temp: [int], end_temp: [int], start_rainfall: [int],
                 end_rainfall: [int], start_flood: [int], end_flood: [int], soil_depth: float,
                 priority=0, frequency=1, start=0, end=maxsize):
        super().__init__(id, model, priority, frequency, start, end)

        model.environment.addComponent(GlobalEnvironmentComponent(model.environment, model, start_temp, end_temp,
                                                                  start_rainfall, end_rainfall, start_flood, end_flood,
                                                                  soil_depth))

    @staticmethod
    def decode(params: dict):
        return GlobalEnvironmentSystem(params['id'], params['model'], params['start_temp'], params['end_temp'],
                                       params['start_rainfall'], params['end_rainfall'], params['start_flood'],
                                       params['end_flood'], params['soil_depth'], priority=params['priority'])

    @staticmethod
    def calcMinMaxGlobalVals(startArr, endArr, percentage):
        min = lerp(startArr[0], endArr[0], percentage)
        max = lerp(startArr[1], endArr[1], percentage)
        return min, max

    def execute(self):

        print("Executing...")
        env_comp = self.model.environment.getComponent(GlobalEnvironmentComponent)

        env_comp.temp.clear()
        env_comp.rainfall.clear()

        percentage = (self.model.systemManager.timestep * 1.0)/self.model.iterations

        min_t, max_t = GlobalEnvironmentSystem.calcMinMaxGlobalVals(env_comp.start_temp, env_comp.end_temp, percentage)

        # Set Rainfall
        min_r, max_r = GlobalEnvironmentSystem.calcMinMaxGlobalVals(env_comp.start_rainfall, env_comp.end_rainfall,
                                                                    percentage)
        # Set Flooding
        min_f, max_f = GlobalEnvironmentSystem.calcMinMaxGlobalVals(env_comp.start_flood, env_comp.end_flood,
                                                                    percentage)
        env_comp.flood = self.model.random.uniform(min_f, max_f)

        for i in range(12):

            env_comp.temp.append(self.model.random.uniform(min_t, max_t))
            env_comp.rainfall.append(self.model.random.uniform(min_r, max_r))

        if self.model.debug:
            print('Global_Properties:\n\n%: {}\nAvg River Height: {}\nTemperatures: {}C\nRainfall: {}mm\nFlood: {}m\n'
                  .format(percentage * 100, self.model.environment.getComponent(SoilMoistureComponent).avgWaterHeight,
                          self.model.environment.getComponent(GlobalEnvironmentComponent).temp,
                          self.model.environment.getComponent(GlobalEnvironmentComponent).rainfall,
                          self.model.environment.getComponent(GlobalEnvironmentComponent).flood))


class SoilMoistureSystem(System, IDecodable):

    def __init__(self, id: str, model: Model, L: int, N: int, I: float, priority=0, frequency=1, start=0, end=maxsize):
        super().__init__(id, model, priority, frequency, start, end)

        model.environment.addComponent(SoilMoistureComponent(model.environment, model, L, N, I))

        def moisture_generator(pos, cells):
            cellID = discreteGridPosToID(pos[0], pos[1] , model.environment.width)
            return SoilMoistureSystem.wfc(model.environment.getComponent(GlobalEnvironmentComponent).soil_depth,
                                          cells['sand_content'][cellID])

        model.environment.addCellComponent('moisture', moisture_generator)

        self.lastAvgMoisture = 0.0


    @staticmethod
    def decode(params: dict):
        return SoilMoistureSystem(params['id'], params['model'], params['L'], params['N'], params['I'],
                                  priority=params['priority'])

    @staticmethod
    def thornthwaite(day_length : int, days: int, avg_temp: int, heat_index : float):
        return 16 * (day_length/12) * (days/30) * (10*avg_temp/heat_index)

    @staticmethod
    def calcRdr(sand_content, soil_m, soil_depth):
        a = SoilMoistureSystem.alpha(sand_content)
        b = SoilMoistureSystem.beta(sand_content)
        return (1 + a)/(1 + a * math.pow(soil_m/soil_depth, b))

    @staticmethod
    def alpha(sand_content):
        clay = 100 - sand_content
        sand_sqrd = math.pow(sand_content, 2)
        return math.exp(-4.396 - 0.0715 * clay - 0.000488 * sand_sqrd -
                        0.00004258 * sand_sqrd * clay) * 100

    @staticmethod
    def beta(sand_content):
        clay = 100 - sand_content
        return -3.140 - 0.000000222 * math.pow(clay, 2) - 0.00003484 * math.pow(sand_content, 2) * clay

    @staticmethod
    def wfc(soil_depth, sand_content):
        return soil_depth * lerp(0.3, 0.7, 1 - (sand_content/100.0))

    def execute(self):

        sm_comp = self.model.environment.getComponent(SoilMoistureComponent)
        global_env_comp = self.model.environment.getComponent(GlobalEnvironmentComponent)

        floodCheckVal = global_env_comp.flood + sm_comp.avgWaterHeight

        soilVals = self.model.environment.cells['moisture'].tolist()
        for i in range(12):

            PET = SoilMoistureSystem.thornthwaite(sm_comp.L, sm_comp.N, global_env_comp.temp[i], sm_comp.I)

            for x in range(self.model.environment.width):
                for y in range(self.model.environment.height):

                    cellID = discreteGridPosToID(x, y, self.model.environment.width)

                    if self.model.environment.cells['isWater'][cellID]:
                        pass

                    if self.model.environment.cells['height'][cellID] < floodCheckVal:
                        soilVals[cellID] = SoilMoistureSystem.wfc(global_env_comp.soil_depth,
                                                                  self.model.environment.cells['sand_content'][cellID])
                        pass

                    if PET > global_env_comp.rainfall[i]:
                        rdr = SoilMoistureSystem.calcRdr(self.model.environment.cells['sand_content'][cellID],
                                                         soilVals[cellID] + global_env_comp.rainfall[i],
                                                         global_env_comp.soil_depth)
                        moisture = soilVals[cellID] - (PET - global_env_comp.rainfall[i]) * rdr
                        soilVals[cellID] = moisture if moisture > 0 else 0

                    else:
                        wfc = SoilMoistureSystem.wfc(global_env_comp.soil_depth,
                                                     self.model.environment.cells['sand_content'][cellID])

                        moisture = self.model.environment.cells['moisture'][cellID] + (global_env_comp.rainfall[i] - PET)
                        soilVals[cellID] = moisture if moisture < wfc else wfc

        self.model.environment.cells.update({'moisture': soilVals})

        if self.model.debug:
            sum = 0.0
            for x in range(self.model.environment.width):
                for y in range(self.model.environment.height):
                    cellID = discreteGridPosToID(x, y, self.model.environment.width)
                    sum += self.model.environment.cells['moisture'][cellID]

            avg_PET = np.mean([SoilMoistureSystem.thornthwaite(sm_comp.L, sm_comp.N, global_env_comp.temp[i], sm_comp.I) for i in range(12)])

            avg = np.mean(self.model.environment.cells['moisture'])
            variance = np.std(self.model.environment.cells['moisture'])
            print('PET: {} Mean Moisture: {} std: {} Delta: {}'.format(avg_PET, avg, variance,
                                                                       avg - self.lastAvgMoisture))
            self.lastAvgMoisture = avg
