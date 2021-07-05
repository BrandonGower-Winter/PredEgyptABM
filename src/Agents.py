import VegetationModel as VM
import math

from ECAgent.Core import *
from ECAgent.Decode import IDecodable
from ECAgent.Environments import PositionComponent, discreteGridPosToID
from ECAgent.Collectors import Collector


class Individual:
    def __init__(self, id, age=0):
        self.id = id
        self.age = age


class ResourceComponent(Component):

    carrying_capacity = 0
    consumption_rate = 0
    age_of_maturity = 0

    def __init__(self, agent, model: Model):
        super().__init__(agent, model)

        self.occupants = {}
        self.occupant_counter = 0
        self.resources = 0
        # Not really a measure of hunger but a measure of how much of the total a family resources requirement was met
        self.hunger = 0
        self.ownedLand = []

    def add_occupant(self, id, age: int = 0):
        self.occupants[id] = Individual(id, age)
        self.occupant_counter += 1

    def claim_land(self, id):
        self.ownedLand.append(id)

    def release_land(self, id):
        self.ownedLand.remove(id)

    def able_workers(self):
        return len([o for o in self.occupants if self.occupants[o].age >= ResourceComponent.age_of_maturity])

    def get_next_id(self):
        return '%d_%d' % (self.agent.id, self.occupant_counter)


class RelationshipComponent(Component):

    def __init__(self, agent, model: Model, settlementID : int):
        super().__init__(agent, model)

        self.settlementID = settlementID


class Household(Agent, IDecodable):

    def __init__(self, id: str, model: Model, settlementID: int):
        super().__init__(id, model)

        self.addComponent(ResourceComponent(self, model))
        self.addComponent(RelationshipComponent(self, model, settlementID))

    @staticmethod
    def decode(params: dict):

        ResourceComponent.age_of_maturity = params['params']['age_of_maturity']
        ResourceComponent.consumption_rate = params['params']['consumption_rate']
        ResourceComponent.carrying_capacity = params['params']['carrying_capacity']

        agent = Household(params['agent_index'], params['model'], -1)
        for i in range(params['params']['init_occupants']):
            age = agent.model.random.randrange(params['params']['init_age_range'][0], params['params']['init_age_range'][1])
            agent[ResourceComponent].add_occupant(agent[ResourceComponent].get_next_id(), age)
        return agent


class AgentResourceAcquisitionSystem(System, IDecodable):

    farms_per_patch = 0
    land_buffer = 0
    max_acquisition_distance = 0.0

    moisture_consumption_rate = 0.0
    crop_gestation_period = 0
    farming_production_rate = 0.0
    forage_consumption_rate = 0.0
    forage_production_multiplier = 0.0

    @staticmethod
    def decode(params: dict):
        AgentResourceAcquisitionSystem.farms_per_patch = params['farms_per_patch']
        AgentResourceAcquisitionSystem.land_buffer = params['land_buffer']
        AgentResourceAcquisitionSystem.max_acquisition_distance = params['max_acquisition_distance']
        AgentResourceAcquisitionSystem.moisture_consumption_rate = params['moisture_consumption_rate']
        AgentResourceAcquisitionSystem.crop_gestation_period = params['crop_gestation_period']
        AgentResourceAcquisitionSystem.farming_production_rate = params['farming_production_rate']
        AgentResourceAcquisitionSystem.forage_consumption_rate = params['forage_consumption_rate']
        AgentResourceAcquisitionSystem.forage_production_multiplier = params['forage_production_multiplier']

        return AgentResourceAcquisitionSystem(params['id'], params['model'], params['priority'])

    def __init__(self, id: str, model: Model,priority):

        super().__init__(id, model, priority=priority)

        def owned_generator(pos, cells):
            return -1

        model.environment.addCellComponent('isOwned', owned_generator)

    def acquire_land(self, household: Household, target: int):
        new_land = int(target * AgentResourceAcquisitionSystem.land_buffer) - len(household[ResourceComponent].ownedLand)
        id = discreteGridPosToID(household[PositionComponent].x, household[PositionComponent].y, self.model.environment.width)

        # Get a list of all the available patches of land
        own_cells = self.model.environment.cells['isOwned']

        available_land = []
        for land_id in household[ResourceComponent].ownedLand:

            available_land +=[x for x in self.model.environment.getNeighbours(self.model.environment.cells['pos'][land_id])
                 if x not in available_land and own_cells[x] == -1 and not self.model.environment.cells['isWater'][x]]
        #available_land = [x for x in self.model.environment.getNeighbours((household[PositionComponent].x, household[PositionComponent].y),
            #radius=AgentResourceAcquisitionSystem.max_acquisition_distance) if own_cells[x] == -1 and not self.model.environment.cells['isWater'][x]]
        if len(available_land) == 0:  # Return if there is simply no land left to allocate
            return

        # Randomly remove land patches
        while new_land < len(available_land):
            available_land.pop(self.model.random.randrange(0, len(available_land)))

        for land_id in available_land:
            household[ResourceComponent].claim_land(land_id)
            self.model.environment.cells.at[land_id, 'isOwned'] = household.id

        # If we did not allocate enough land we can call the function again
        if len(household[ResourceComponent].ownedLand) < target:
            self.acquire_land(household, target)

    def farm(self, patch_id, workers):
        # Calculate penalties
        tmp_Penalty = VM.VegetationGrowthSystem.tempPenalty(
            sum(self.model.environment.getComponent(VM.GlobalEnvironmentComponent).temp)/12.0
        )
        wtr_penalty, moisture_remain = VM.VegetationGrowthSystem.waterPenalty(self.model.environment.cells['moisture'][patch_id],
            AgentResourceAcquisitionSystem.moisture_consumption_rate/AgentResourceAcquisitionSystem.crop_gestation_period, 1.0)
        # Calculate Crop Yield
        crop_yield = AgentResourceAcquisitionSystem.farming_production_rate * wtr_penalty * tmp_Penalty * workers
        # Adjust soil moisture
        self.model.environment.cells.at[patch_id, 'moisture'] = moisture_remain
        return int(crop_yield)

    def gather_resources(self, household: Household, patch_id: int, workers: int,farm: bool):
        if farm:
            household[ResourceComponent].resources += self.farm(patch_id, workers)
        else:
            distance = math.sqrt((self.model.environment.cells['pos'][patch_id][0] - household[PositionComponent].x) ** 2
                                 + (self.model.environment.cells['pos'][patch_id][1] - household[PositionComponent].y) ** 2)
            dst_multiplier = (1.0 - distance/AgentResourceAcquisitionSystem.max_acquisition_distance)
            veg_diff = max(self.model.environment.cells['vegetation'][patch_id]
                           - AgentResourceAcquisitionSystem.forage_consumption_rate * workers/AgentResourceAcquisitionSystem.farms_per_patch
                           , 0.0)
            household[ResourceComponent].resources += (self.model.environment.cells['vegetation'][patch_id] - veg_diff) \
                * AgentResourceAcquisitionSystem.forage_production_multiplier * dst_multiplier
            self.model.environment.cells.at[patch_id, 'vegetation'] = veg_diff

    def execute(self):
        for household in self.model.environment.getAgents():
            # Determine how many patches a household can farm
            able_workers = household[ResourceComponent].able_workers()
            numToFarm = able_workers / AgentResourceAcquisitionSystem.farms_per_patch
            # If ownedLand < patches to farm allocate more land to farm
            if len(household[ResourceComponent].ownedLand) < numToFarm:
                self.acquire_land(household, numToFarm)

            hPos = (household[PositionComponent].x, household[PositionComponent].y)

            # Select land patches
            farmableLand = [x for x in household[ResourceComponent].ownedLand]
            foragableLand = [x for x in self.model.environment.getNeighbours(hPos,
                 radius=AgentResourceAcquisitionSystem.max_acquisition_distance)
                             if (self.model.environment.cells['isOwned'][x] == -1 or self.model.environment.cells['isOwned'][x] == household.id)
                                 and not self.model.environment.cells['isWater'][x]]

            for i in range(math.ceil(numToFarm)):

                worker_diff = max(able_workers - AgentResourceAcquisitionSystem.farms_per_patch, 0)
                workers = able_workers - worker_diff
                able_workers = worker_diff

                if self.model.random.random() < self.model.systemManager.timestep / self.model.iterations:
                    patchID = farmableLand.pop(self.model.random.randrange(0, len(farmableLand)))  # Remove patches of land randomly
                    self.gather_resources(household, patchID, workers, True)  # The choice of farming or foraging is random
                else:
                    patchID = foragableLand.pop(self.model.random.randrange(0, len(foragableLand)))
                    self.gather_resources(household, patchID, workers, False)


class AgentResourceConsumptionSystem(System, IDecodable):

    def __init__(self, id: str, model: Model,priority):
        super().__init__(id, model, priority=priority)

    @staticmethod
    def decode(params: dict):
        return AgentResourceConsumptionSystem(params['id'], params['model'], params['priority'])

    def execute(self):
        for agent in self.model.environment.getAgents():
            resComp = agent[ResourceComponent]
            resComp.hunger = min(1.0, resComp.resources / (ResourceComponent.consumption_rate * len(resComp.occupants)))
            resComp.resources = max(0,
                 resComp.resources - (ResourceComponent.consumption_rate * len(resComp.occupants)))


class AgentPopulationSystem(System, IDecodable):
    """This system is responsible for managing agent reproduction, death and aging"""
    def __init__(self, id: str, model: Model, priority, birth_rate, death_rate):
        super().__init__(id, model, priority=priority)

        self.birth_rate = birth_rate
        self.death_rate = death_rate

    def execute(self):

        toRem = []
        for household in self.model.environment.getAgents():

            # Reallocation check
            if household[ResourceComponent].hunger < 1.0 and self.model.random.random() < (1.0 - household[ResourceComponent].hunger):
                self.reallocate_agent(household)

            # Birth Chance
            for i in range(household[ResourceComponent].able_workers()//2):
                if self.model.random.random() * household[ResourceComponent].hunger <= self.birth_rate:
                    household[ResourceComponent].add_occupant(household[ResourceComponent].get_next_id())
                    # TODO Split households based on household cap here


            # Death Chance
            occuRem = []
            for occupant in household[ResourceComponent].occupants:
                # Random death rate
                if self.model.random.random() * household[ResourceComponent].hunger < self.death_rate:
                    occuRem.append(occupant)
                else:  # If occupant does not die, they age by 1
                    household[ResourceComponent].occupants[occupant].age += 1

            for o in occuRem:
                household[ResourceComponent].occupants.pop(o)

            if household[ResourceComponent].able_workers() == 0: # Add households to the delete list
                toRem.append(household)

        # Delete empty households here
        for household in toRem:
            # Remove all land from ownership
            self.model.environment.cells.loc[(self.model.environment.cells.isOwned == household.id), 'isOwned'] = -1
            # Delete Agent
            self.model.environment.removeAgent(household.id)

    def reallocate_agent(self, household: Household):
        # Get rid of all land ownership that household has
        self.model.environment.cells.loc[(self.model.environment.cells.isOwned == household.id), 'isOwned'] = -1
        household[ResourceComponent].ownedLand.clear()

        # Assign new household position
        new_x, new_y = (self.model.random.randrange(0, self.model.environment.width), self.model.random.randrange(0, self.model.environment.height))
        new_unq_id = discreteGridPosToID(new_x, new_y, self.model.environment.width)

        # This just ensures that an agent doesn't reallocate to a cell that is owned by another agent. (Will need to be reworked)
        while self.model.environment.cells['isOwned'][new_unq_id] != -1:
            new_x, new_y = (self.model.random.randrange(0, self.model.environment.width),
                            self.model.random.randrange(0, self.model.environment.height))
            new_unq_id = discreteGridPosToID(new_x, new_y, self.model.environment.width)

        household[PositionComponent].x = new_x
        household[PositionComponent].y = new_y
        household[ResourceComponent].claim_land(new_unq_id)
        self.model.environment.cells.at[new_unq_id, 'isOwned'] = household.id

    @staticmethod
    def decode(params: dict):
        return AgentPopulationSystem(params['id'], params['model'], params['priority'], params['birth_rate'],
                                     params['death_rate'])


class AgentCollector(Collector, IDecodable):

    def __init__(self, id: str, model: Model):
        super().__init__(id, model)

    def collect(self):

        agents = self.model.environment.getAgents()
        self.records.append({})
        for agent in agents:
            self.records[self.model.systemManager.timestep][agent.id] = {
                    'resources': agent[ResourceComponent].resources,
                    'population': len(agent[ResourceComponent].occupants)
                }
        self.records[self.model.systemManager.timestep]['total'] = {
            'resources': sum([self.records[self.model.systemManager.timestep][x]['resources'] for x in self.records[self.model.systemManager.timestep]]),
            'population': sum([self.records[self.model.systemManager.timestep][x]['population'] for x in self.records[self.model.systemManager.timestep]])
        }

    @staticmethod
    def decode(params: dict):
        return AgentCollector(params['id'], params['model'])