import math
import numpy as np

from ECAgent.Core import *
from ECAgent.Environments import *
from ECAgent.Decode import *
from ECAgent.Collectors import Collector


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

        self.avgWaterHeight = sum / count if count != 0 else 1.0


class VegetationGrowthComponent(Component):

    def __init__(self, agent, model: Model, init_pop: int, carry_pop: int, growth_rate: float, decay_rate: float,
                 ideal_moisture: float):
        super().__init__(agent, model)

        self.init_pop = init_pop
        self.carry_pop = carry_pop
        self.growth_rate = growth_rate
        self.decay_rate = decay_rate
        self.ideal_moisture = ideal_moisture


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

        if self.model.debug:
            print("Generating Global Data Variables...")

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
                   cells['sand_content'][cellID]) * min(1.0,
                        math.pow(cells['height'][cellID] / model.environment.getComponent(SoilMoistureComponent).avgWaterHeight, 2))

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

        for x in range(self.model.environment.width):
            for y in range(self.model.environment.height):

                cellID = discreteGridPosToID(x, y, self.model.environment.width)

                if self.model.environment.cells['isWater'][cellID]:
                    pass

                if self.model.environment.cells['height'][cellID] < floodCheckVal:
                    soilVals[cellID] = SoilMoistureSystem.wfc(global_env_comp.soil_depth,
                                                              self.model.environment.cells['sand_content'][cellID])
                    pass

                for i in range(12):
                    PET = SoilMoistureSystem.thornthwaite(sm_comp.L, sm_comp.N, global_env_comp.temp[i], sm_comp.I)

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


class VegetationGrowthSystem(System, IDecodable):

    def __init__(self, id: str, model: Model, init_pop: int, carry_pop: int, growth_rate: float, decay_rate: float,
                 ideal_moisture, priority=0, frequency=1, start=0, end=maxsize):

        super().__init__(id, model, priority=priority, frequency=frequency, start=start, end=end)

        model.environment.addComponent(VegetationGrowthComponent(self.model.environment, model, init_pop,
                                                                 carry_pop, growth_rate, decay_rate, ideal_moisture))

        # Create a random range of values for the inititial vegetation population
        max_carry = min(carry_pop, int(init_pop + (init_pop * init_pop/carry_pop)))
        min_carry = max(0, int(init_pop - (init_pop * init_pop/carry_pop)))

        def vegetation_generator(pos, cells):
            return self.model.random.uniform(min_carry, max_carry)

        model.environment.addCellComponent('vegetation', vegetation_generator)

    @staticmethod
    def Logistic_Growth(pop: float, carry_cap: int, growth_rate: float):
        return growth_rate * pop * ((carry_cap - pop)/carry_cap) if carry_cap > pop else 0

    @staticmethod
    def decay(val: float, rate: float):
        return val * rate

    @staticmethod
    def waterPenalty(moisture: float, moisture_ideal: float, capacity_ratio: float):

        moisture_req = capacity_ratio * moisture_ideal

        if moisture < moisture_req:
            return moisture/moisture_req, 0.0
        else:
            return 1.0, moisture - moisture_req

    @staticmethod
    def tOpt(temp: float):
        return -0.0005 * math.pow(temp - 20.0, 2) + 1

    @staticmethod
    def tempPenalty(temperature: float):
        topt = VegetationGrowthSystem.tOpt(temperature)
        return 0.8 + 0.02 * topt - 0.0005 * math.pow(topt,2)

    def execute(self):

        veg_vals = self.model.environment.cells['vegetation'].tolist()
        soil_vals = self.model.environment.cells['moisture'].tolist()

        vg_comp = self.model.environment.getComponent(VegetationGrowthComponent)

        for x in range(self.model.environment.width):
            for y in range(self.model.environment.height):

                cellID = discreteGridPosToID(x, y, self.model.environment.width)

                if self.model.environment.cells['isWater'][cellID]:
                    pass

                # Check for cell refill given neighbour's vegetation density
                if veg_vals[cellID] < 1.0:
                    veg_sum = 0.0
                    count = 0
                    for neighbour in self.model.environment.getNeighbours(self.model.environment.cells['pos'][cellID]):
                        veg_sum += veg_vals[neighbour]
                        count += 1

                    veg_avg = veg_sum/count
                    if self.model.random.random() < veg_avg/self.model.environment.getComponent(VegetationGrowthComponent).carry_pop:
                        veg_vals[cellID] = veg_avg

                else:
                    capacity_ratio = veg_vals[cellID]/self.model.environment.getComponent(
                        VegetationGrowthComponent).carry_pop

                    r, soil_vals[cellID] = VegetationGrowthSystem.waterPenalty(soil_vals[cellID],
                                                                           self.model.environment.getComponent(
                                                                               VegetationGrowthComponent
                                                                           ).ideal_moisture, capacity_ratio)

                    r *= VegetationGrowthSystem.tempPenalty(
                        np.mean(self.model.environment.getComponent(GlobalEnvironmentComponent).temp)
                    )
                    veg_vals[cellID] -= VegetationGrowthSystem.decay(veg_vals[cellID], vg_comp.decay_rate)
                    veg_vals[cellID] += VegetationGrowthSystem.Logistic_Growth(veg_vals[cellID], vg_comp.carry_pop * r,
                                                                           vg_comp.growth_rate)

        self.model.environment.cells.update({'moisture': soil_vals, 'vegetation': veg_vals})

        if self.model.debug:
            print('...Vegetation System...')
            print('Vegetation: {} Mean Moisture: {}'.format(np.mean(self.model.environment.cells['vegetation']),
                                                            np.mean(self.model.environment.cells['moisture'])))

    @staticmethod
    def decode(params: dict):
        return VegetationGrowthSystem(params['id'], params['model'], params['init_pop'], params['carry_pop'],
                                      params['growth_rate'], params['decay_rate'], params['ideal_moisture'],
                                      priority=params['priority'])


class SoilContentSystem(System, IDecodable):
    def __init__(self, id: str, model: Model, sand_content_range, priority=0, frequency=1, start=0, end=maxsize):
        super().__init__(id, model, priority, frequency, start, end)

        self.sand_content_range = sand_content_range

    def execute(self):

        sand_cells = self.model.environment.cells['sand_content'].tolist()
        vegetation_cells = self.model.environment.cells['vegetation']

        for i in range(len(sand_cells)):
            sand_cells[i] = lerp(self.sand_content_range[0], self.sand_content_range[1],
                                  1.0 - vegetation_cells[i]/self.model.environment.getComponent(VegetationGrowthComponent).carry_pop)
            if(1.0 - vegetation_cells[i]/self.model.environment.getComponent(VegetationGrowthComponent).carry_pop < 0.0):
                print(vegetation_cells[i])

        self.model.environment.cells.update({'sand_content': sand_cells})

    @staticmethod
    def decode(params: dict):
        return SoilContentSystem(params['id'], params['model'], params['sand_content_range'],
                                 priority=params['priority'])


class VegetationCollector(Collector, IDecodable):

    def __init__(self, id: str, model: Model):
        super().__init__(id, model)

        self.records.append([])
        self.records.append([])

    def collect(self):

        wh = self.model.environment.width * self.model.environment.height
        self.records[1].append(np.mean(
            [
                self.model.environment.cells['vegetation'][i] for i in range(wh)
                if not self.model.environment.cells['isWater'][i]
            ]
        ))
        self.records[0].append(np.mean([
                self.model.environment.cells['moisture'][i] for i in range(wh)
                if not self.model.environment.cells['isWater'][i]
            ]
        ))

    @staticmethod
    def decode(params: dict):
        return VegetationCollector(params['id'], params['model'])
