import statistics

import VegetationModel
import math
import numpy as np
import json

from ECAgent.Core import *
from ECAgent.Decode import IDecodable
from ECAgent.Environments import PositionComponent, discreteGridPosToID
from ECAgent.Collectors import Collector, FileCollector
from VegetationModel import VegetationGrowthSystem, SoilMoistureSystem, SoilMoistureComponent, GlobalEnvironmentComponent

from Logging import ILoggable

# Cython Functions
from CythonFunctions import CAgentResourceConsumptionSystemFunctions, CAgentResourceAcquisitionFunctions, CVegetationGrowthSystemFunctions


class Individual:
    def __init__(self, id, age=0):
        self.id = id
        self.age = age


class ResourceComponent(Component):

    carrying_capacity = 0
    consumption_rate = 0
    child_factor = 0.0
    age_of_maturity = 0
    vision_square = 0

    def __init__(self, agent, model: Model):
        super().__init__(agent, model)

        self.occupants = {}
        self.occupant_counter = 0
        self.resources = 0
        # Not really a measure of hunger but a measure of how much of the total a family resources requirement was met
        self.hunger = 0
        self.satisfaction = 0.0
        self.ownedLand = []

    def add_occupant(self, id, age: int = 0):
        # TODO Manage IDs properly
        self.occupants[id] = Individual(id, age)
        self.occupant_counter += 1

    def move_occupant(self, id, other_household):
        other_household[ResourceComponent].occupants[id] = self.occupants[id]
        self.occupants.pop(id)

    def claim_land(self, id):
        self.ownedLand.append(id)

    def release_land(self, id):
        self.ownedLand.remove(id)

    def able_workers(self):
        return len([o for o in self.occupants if self.occupants[o].age >= ResourceComponent.age_of_maturity])

    def average_age(self):
        return statistics.mean([self.occupants[o].age for o in self.occupants
                                if self.occupants[o].age >= ResourceComponent.age_of_maturity])

    def get_next_id(self):
        return '%d_%d' % (self.agent.id, self.occupant_counter)

    def required_resources(self):
        occu_length = len(self.occupants)
        return ResourceComponent.consumption_rate * (self.able_workers() + (ResourceComponent.child_factor * (occu_length - self.able_workers())))

    def excess_resources(self):
        return max(0.0, self.resources - self.required_resources())


class HouseholdRelationshipComponent(Component):

    load_difference = 0

    def __init__(self, agent, model: Model, settlementID : int):
        super().__init__(agent, model)

        self.settlementID = settlementID
        self.load = 0

        # Randomize the resource trading personalities
        self.peer_resource_transfer_chance = self.model.random.random()
        self.sub_resource_transfer_chance = self.model.random.random()

    def is_aquaintance(self, h):
        """Returns true if household h is in the same settlement as household h"""
        return self.settlementID == h[HouseholdRelationshipComponent].settlementID

    def is_auth(self, h):
        """Returns true if household self has an authority relationship over household h"""

        s_status = self.agent.social_status()
        if h is self.agent or s_status == 0:
            return False

        h_status = h.social_status()
        return self.is_aquaintance(h) and ((s_status - h_status)/max(s_status, h_status)
                                           > HouseholdRelationshipComponent.load_difference)

    def is_sub(self, h):
        """Returns true if household self has a subordinate relationship with household h"""
        # Household self is a subordinate if the household h has an auth relationship
        return h[HouseholdRelationshipComponent].is_auth(self.agent)

    def is_peer(self, h):
        """Returns true if household self has a peer relationship with household h"""

        if h is self.agent or self.is_auth(h):
            return False

        h_status = h.social_status()
        s_status = self.agent.social_status()

        if h_status == 0 and s_status == 0:
            return True  # Ensure no division by error

        return self.is_aquaintance(h) and abs(s_status - h_status)/max(s_status, h_status) \
               < HouseholdRelationshipComponent.load_difference


class HouseholdPreferenceComponent(Component):

    learning_rate_range = [0.0, 1.0]

    def __init__(self, agent, model: Model, init_preference: float = 0.0):
        super().__init__(agent, model)

        self.forage_utility = init_preference
        self.farm_utility = 0.0
        self.learning_rate = model.random.uniform(HouseholdPreferenceComponent.learning_rate_range[0],
                                                 HouseholdPreferenceComponent.learning_rate_range[1])

        self.prev_hunger = 1.0


class HouseholdRBAdaptiveComponent(Component):

    yrs_to_look_back = 1
    yr_look_back_weights = [1.0]

    max_labour_adapt_size = 6.0

    risk_elasticity = 1.0
    cognitive_bias = 0.0

    adaptation_intention_threshold = 0.5
    learning_rate = 0.05

    def __init__(self, agent: Agent, model: Model):
        super().__init__(agent, model)

        self.rainfall_memory = [0.0] * HouseholdRBAdaptiveComponent.yrs_to_look_back
        self.flood_memory = [0.0] * HouseholdRBAdaptiveComponent.yrs_to_look_back

        self.percentage_to_farm = 0.0  # Probability of Agent choosing to farm

    def update_flood_memory(self, val: float):
        self.flood_memory.append(val)

        # Remove Oldest Memory
        if len(self.flood_memory) > HouseholdRBAdaptiveComponent.yrs_to_look_back:
            self.flood_memory.pop(0)

    def update_rainfall_memory(self, val: float):
        self.rainfall_memory.append(val)

        # Remove Oldest Memory
        if len(self.rainfall_memory) > HouseholdRBAdaptiveComponent.yrs_to_look_back:
            self.rainfall_memory.pop(0)


class Household(Agent, IDecodable):

    def __init__(self, id: str, model: Model, settlementID: int):
        super().__init__(id, model)

        self.addComponent(ResourceComponent(self, model))
        self.addComponent(HouseholdRelationshipComponent(self, model, settlementID))

    @staticmethod
    def decode(params: dict):

        ResourceComponent.age_of_maturity = params['age_of_maturity']
        ResourceComponent.consumption_rate = params['consumption_rate']
        ResourceComponent.carrying_capacity = params['carrying_capacity']
        ResourceComponent.vision_square = params['vision_square']
        ResourceComponent.child_factor = params['child_factor']

        HouseholdRelationshipComponent.load_difference = params['load_difference']

        agent = Household(params['agent_index'], params['model'], -1)
        for i in range(params['init_occupants']):
            age = agent.model.random.randrange(params['init_age_range'][0], params['init_age_range'][1])
            agent[ResourceComponent].add_occupant(agent[ResourceComponent].get_next_id(), age)
        return agent

    def __str__(self):
        return 'Household {}:\n\tOccupants: {}\n\tResources: {}\n\tLoad: {}'.format(
            self.id, len(self[ResourceComponent].occupants), self[ResourceComponent].resources, self[HouseholdRelationshipComponent].load)

    def social_status(self):
        """ Returns the agents social status which is the sum of its resources and load. """
        return self[ResourceComponent].resources + self[HouseholdRelationshipComponent].load

    def jsonify(self) -> dict:
        created_dict = {}
        created_dict['id'] = self.id

        created_dict['occupants'] = len(self[ResourceComponent].occupants)
        created_dict['able_workers'] = self[ResourceComponent].able_workers()
        created_dict['resources'] = self[ResourceComponent].resources
        created_dict['hunger'] = self[ResourceComponent].hunger
        created_dict['satisfaction'] = self[ResourceComponent].satisfaction
        created_dict['owned_land'] = self[ResourceComponent].ownedLand

        created_dict['settlement_id'] = self[HouseholdRelationshipComponent].settlementID
        created_dict['load'] = self[HouseholdRelationshipComponent].load
        created_dict['peer_chance'] = self[HouseholdRelationshipComponent].peer_resource_transfer_chance
        created_dict['sub_chance'] = self[HouseholdRelationshipComponent].sub_resource_transfer_chance

        return created_dict


class RBAdaptiveHousehold(Household):
    """ This Agent-Type is heavily inspired by the agents described by Hailegiorgis, Crooks and Cioffi-Revilla in their
    paper titled 'An Agent-Based Model for Rural Households Adaptation to Climate Change"""

    def __init__(self, id: str, model: Model, settlementID: int):
        super().__init__(id, model, settlementID)

        self.addComponent(HouseholdRBAdaptiveComponent(self, model))

    @staticmethod
    def decode(params: dict):
        ResourceComponent.age_of_maturity = params['age_of_maturity']
        ResourceComponent.consumption_rate = params['consumption_rate']
        ResourceComponent.carrying_capacity = params['carrying_capacity']
        ResourceComponent.vision_square = params['vision_square']
        ResourceComponent.child_factor = params['child_factor']

        HouseholdRelationshipComponent.load_difference = params['load_difference']

        HouseholdRBAdaptiveComponent.yrs_to_look_back = params['yrs_to_look_back']
        HouseholdRBAdaptiveComponent.yr_look_back_weights = params['yr_look_back_weights']

        HouseholdRBAdaptiveComponent.max_labour_adapt_size = params['max_labour_adapt_size']

        HouseholdRBAdaptiveComponent.risk_elasticity = params['risk_elasticity']
        HouseholdRBAdaptiveComponent.cognitive_bias = params['cognitive_bias']

        HouseholdRBAdaptiveComponent.adaptation_intention_threshold = params['adaptation_intention_threshold']
        HouseholdRBAdaptiveComponent.learning_rate = params['learning_rate']

        agent = RBAdaptiveHousehold(params['agent_index'], params['model'], -1)
        for i in range(params['init_occupants']):
            age = agent.model.random.randrange(params['init_age_range'][0], params['init_age_range'][1])
            agent[ResourceComponent].add_occupant(agent[ResourceComponent].get_next_id(), age)
        return agent

    def __str__(self):
        return super().__str__() + '\n\tProbability to Farm: '.format(
            self[HouseholdRBAdaptiveComponent].percentage_to_farm)

    def jsonify(self) -> dict:
        created_dict = super().jsonify()

        created_dict['percentage_to_farm'] = self[HouseholdRBAdaptiveComponent].percentage_to_farm
        return created_dict


class PreferenceHousehold(Household):

    def __init__(self, id: str, model: Model, settlementID: int, init_preference):
        super().__init__(id, model, settlementID)

        self.addComponent(HouseholdPreferenceComponent(self, model, init_preference))

    @staticmethod
    def decode(params: dict):
        ResourceComponent.age_of_maturity = params['age_of_maturity']
        ResourceComponent.consumption_rate = params['consumption_rate']
        ResourceComponent.carrying_capacity = params['carrying_capacity']
        ResourceComponent.vision_square = params['vision_square']
        ResourceComponent.child_factor = params['child_factor']

        HouseholdRelationshipComponent.load_difference = params['load_difference']

        HouseholdPreferenceComponent.learning_rate_range = params['learning_rate_range']

        agent = PreferenceHousehold(params['agent_index'], params['model'], -1, params['init_preference'])
        for i in range(params['init_occupants']):
            age = agent.model.random.randrange(params['init_age_range'][0], params['init_age_range'][1])
            agent[ResourceComponent].add_occupant(agent[ResourceComponent].get_next_id(), age)
        return agent

    def __str__(self):
        return super().__str__() + '\n\tFarming Preference(f/F): ({}, {})'.format(
            self[HouseholdPreferenceComponent].forage_utility, self[HouseholdPreferenceComponent].farm_utility)

    def jsonify(self) -> dict:
        created_dict = super().jsonify()

        created_dict['forage_utility'] = self[HouseholdPreferenceComponent].forage_utility
        created_dict['farm_utility'] = self[HouseholdPreferenceComponent].farm_utility
        created_dict['prev_hunger'] = self[HouseholdPreferenceComponent].prev_hunger
        created_dict['learning_rate'] = self[HouseholdPreferenceComponent].learning_rate

        return created_dict


class Settlement:
    def __init__(self, id):
        self.id = id
        self.pos = []
        self.occupants = []

    def jsonify(self):
        return {
            'id': self.id,
            'pos': self.pos,
            'occupants': self.occupants
        }


class SettlementRelationshipComponent(Component):

    def __init__(self, agent: Agent, model: Model, yrs_per_move, cell_capacity):
        super().__init__(agent, model)
        self.settlement_count = 0
        self.settlements = {}

        self.yrs_per_move = yrs_per_move
        self.cell_capacity = cell_capacity

    def create_settlement(self) -> int:
        self.settlements[self.settlement_count] = Settlement(self.settlement_count)
        id = self.settlement_count
        self.settlement_count += 1
        return id

    def remove_settlement(self, settlementID):
        # Remove all settlement spots
        for unq_id in self.settlements[settlementID].pos:
            self.model.environment.cells.at[unq_id, 'isSettlement'] = -1
        self.settlements.pop(settlementID)

    def add_household_to_settlement(self, household: Household, settlementID):

        if len(self.settlements[settlementID].occupants) > 0 and len(self.settlements[settlementID].occupants) % self.cell_capacity == 0:
            # Acquire more land from a neighbouring cell
            plot_choices = []
            for coords in [self.model.environment.cells['pos'][x] for x in self.settlements[settlementID].pos]:
                plot_choices += [p for p in self.model.environment.getNeighbours(coords)
                                 if p not in plot_choices and not self.model.environment.cells['isWater'][p]
                                 and self.model.environment.cells['isSettlement'][p] == -1]

            choice = self.model.random.choice(plot_choices)
            self.settlements[settlementID].pos.append(choice)
            self.model.environment.cells.at[choice, 'isSettlement'] = settlementID

            # Remove house from owners list if settlement expands
            if self.model.environment.cells['isOwned'][choice] != -1:
                rem_h = self.model.environment.getAgent(self.model.environment.cells['isOwned'][choice])
                rem_h[ResourceComponent].release_land(choice)
                self.model.environment.cells.at[choice, 'isOwned'] = -1

        self.settlements[settlementID].occupants.append(household.id)
        household[HouseholdRelationshipComponent].settlementID = settlementID

    def remove_household(self, household: Household):
        sID = household[HouseholdRelationshipComponent].settlementID
        self.settlements[sID].occupants.remove(household.id)

        # Purge Settlement if it doesn't exist
        if len(self.settlements[sID].occupants) == 0:
            self.remove_settlement(sID)

        household[HouseholdRelationshipComponent].settlementID = -1

    def move_household(self, household: Household, new_settlementID):
        self.remove_household(household)
        self.add_household_to_settlement(household, new_settlementID)

    def merge_settlements(self, s1, s2):
        """Merges Settlement S2 into Settlement S1"""

        # Move all houses
        for houseID in self.settlements[s2].occupants:
            self.move_household(self.model.environment.getAgent(houseID), s1)

        # Delete Settlement s2
        self.remove_settlement(s2)

    def getSettlementWealth(self, settlementID):
        return sum([self.model.environment.getAgent(h)[ResourceComponent].resources for h in self.settlements[settlementID].occupants])

    def getSettlementLoad(self, settlementID):
        return sum([self.model.environment.getAgent(h)[HouseholdRelationshipComponent].load for h in
                    self.settlements[settlementID].occupants])

    def getSettlementPopulation(self, settlementID):
        return sum([len(self.model.environment.getAgent(h)[ResourceComponent].occupants) for h in
                    self.settlements[settlementID].occupants])

    def getSettlementFarmUtility(self, settlementID):
        return sum([self.model.environment.getAgent(h)[HouseholdPreferenceComponent].farm_utility for h in
                    self.settlements[settlementID].occupants
                   if self.model.environment.getAgent(h).hasComponent(HouseholdPreferenceComponent)]
                   ) / len(self.settlements[settlementID].occupants)

    def getSettlementForageUtility(self, settlementID):
        return sum([self.model.environment.getAgent(h)[HouseholdPreferenceComponent].forage_utility for h in
                    self.settlements[settlementID].occupants
                   if self.model.environment.getAgent(h).hasComponent(HouseholdPreferenceComponent)]
                   ) / len(self.settlements[settlementID].occupants)

    def getSettlementPercentageToFarm(self, settlementID):
        return sum([self.model.environment.getAgent(h)[HouseholdRBAdaptiveComponent].percentage_to_farm for h in
                    self.settlements[settlementID].occupants
                    if self.model.environment.getAgent(h).hasComponent(HouseholdRBAdaptiveComponent)]
                   ) / len(self.settlements[settlementID].occupants)

    def getAverageSettlementWealth(self, settlementID):
        return self.getSettlementWealth(settlementID) / len(self.settlements[settlementID].occupants)

    def getEmptySettlementNeighbours(self, settlementID, ensure=False, exclude=[]):
        emptyCells = []
        count = 1
        while (ensure and len(emptyCells) == 0) or len(emptyCells) == 0:
            for coords in [self.model.environment.cells['pos'][x] for x in self.settlements[settlementID].pos]:

                emptyCells += [p for p in self.model.environment.getNeighbours(coords, radius=count)
                                 if p not in emptyCells and not self.model.environment.cells['isWater'][p]
                                 and self.model.environment.cells['isSettlement'][p] == -1 and self.model.environment.cells['isOwned'][p] == -1
                                and p not in exclude]
            count += 1
        return emptyCells

    def get_all_auth(self, h: Household):
        """Returns all of the households which have an auth relationship over household h"""
        return [self.model.environment.getAgent(x) for x in self.settlements[h[HouseholdRelationshipComponent].settlementID].occupants
                if self.model.environment.getAgent(x)[HouseholdRelationshipComponent].is_auth(h)]

    def get_all_sub(self, h: Household):
        """Returns all households which have a subordinate relationship over household h"""
        return [self.model.environment.getAgent(x) for x in self.settlements[h[HouseholdRelationshipComponent].settlementID].occupants
                if self.model.environment.getAgent(x)[HouseholdRelationshipComponent].is_sub(h)]

    def get_all_peer(self, h: Household):
        """Returns all of the households which have a peer relationship over household h"""
        return [self.model.environment.getAgent(x) for x in
                self.settlements[h[HouseholdRelationshipComponent].settlementID].occupants
                if self.model.environment.getAgent(x)[HouseholdRelationshipComponent].is_peer(h)]


class AgentResourceAcquisitionSystem(System, IDecodable, ILoggable):

    farms_per_patch = 0
    land_buffer = 0
    max_acquisition_distance = 0

    moisture_consumption_rate = 0
    crop_gestation_period = 0
    farming_production_rate = 0
    forage_consumption_rate = 0
    forage_production_multiplier = 0.0

    delay_factor = 0

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

        if 'delay_factor' in params:
            AgentResourceAcquisitionSystem.delay_factor = params['delay_factor']

        return AgentResourceAcquisitionSystem(params['id'], params['model'], params['priority'])

    def __init__(self, id: str, model: Model,priority):

        System.__init__(self, id, model, priority=priority)
        IDecodable.__init__(self)
        ILoggable.__init__(self, 'model.RAS')

        def owned_generator(pos, cells):
            return -1

        model.environment.addCellComponent('isOwned', owned_generator)

    @staticmethod
    def num_to_farm(threshold, maxFarm, random):
        numToFarm = 0
        for index in range(maxFarm):
            if random.random() < threshold:
                numToFarm += 1
        return numToFarm

    @staticmethod
    def num_to_farm_phouse(threshold, maxFarm, random, farm_utility, forage_utility):
        numToFarm = 0
        for index in range(maxFarm):
            is_farm_max = farm_utility > forage_utility
            # A satisfied house has a hunger of 1.0
            if random.random() < threshold:
                if is_farm_max:
                    numToFarm += 1
            else:
                numToFarm += random.randint(0, 1)

        return numToFarm

    @staticmethod
    def generateNeighbours(xPos, yPos, width, height, radius):
        toReturn = []
        for x in range(max(xPos - radius, 0), min(xPos + radius, width)):
            for y in range(max(yPos - radius, 0), min(yPos + radius, height)):
                toReturn.append(discreteGridPosToID(x, y, width))
        return toReturn

    def acquire_land(self, household: Household, target: int, owned_cells):
        new_land = int(target * AgentResourceAcquisitionSystem.land_buffer) - len(household[ResourceComponent].ownedLand)

        # Get a list of all the available patches of land

        available_land = []
        past_iter = 0
        while len(available_land) < new_land:
            for land_id in household[ResourceComponent].ownedLand:

                available_land +=[x for x in self.model.environment.getNeighbours(self.model.environment.cells['pos'][land_id])
                    if x not in available_land and owned_cells[x] == -1 and not self.model.environment.cells['isWater'][x]
                    and self.model.environment.cells['isSettlement'][x] == -1]

            if len(available_land) == past_iter:
                available_land.append(self.model.random.choice(
                    [x for x in self.model.environment[SettlementRelationshipComponent].getEmptySettlementNeighbours(
                    household[HouseholdRelationshipComponent].settlementID, True, available_land)]
                ))

            past_iter = len(available_land)

        moist_cells = self.model.environment.cells['moisture']

        def getMoisture(loc):
            return moist_cells[loc]

        # Sort farm by moisture levels
        available_land.sort(key=getMoisture)

        # Remove least promising land patches
        while new_land < len(available_land):
            available_land.pop(0)

        for land_id in available_land:
            household[ResourceComponent].claim_land(land_id)
            self.model.environment.cells.at[land_id, 'isOwned'] = household.id

            self.logger.info('HOUSEHOLD.CLAIM: {} {}'.format(household.id, land_id))

        # If we did not allocate enough land we can call the function again
        if len(household[ResourceComponent].ownedLand) < target:
            self.acquire_land(household, target)

    def execute(self):
        # Instantiate numpy arrays of environment dataframe

        owned_cells = self.model.environment.cells['isOwned'].to_numpy()
        settlement_cells = self.model.environment.cells['isSettlement'].to_numpy()
        water_cells = self.model.environment.cells['isWater'].to_numpy()
        vegetation_cells = self.model.environment.cells['vegetation'].to_numpy()
        moisture_cells = self.model.environment.cells['moisture'].to_numpy()
        height_cells = self.model.environment.cells['height'].to_numpy()
        position_cells = self.model.environment.cells['pos']  # Not passed to Cython so it doesn't need to be np.array
        slope_cells = self.model.environment.cells['slope'].to_numpy()

        def getVegetation(location):  # Function used to sort land patches by vegetation density.
            return vegetation_cells[location]

        for household in self.model.environment.getAgents():
            # Determine how many patches a household can farm
            able_workers = household[ResourceComponent].able_workers()
            max_farm = math.ceil(able_workers / AgentResourceAcquisitionSystem.farms_per_patch)

            is_phouse = household.hasComponent(HouseholdPreferenceComponent)

            if is_phouse:
                farm_threshold = household[HouseholdPreferenceComponent].prev_hunger + (1 * self.model.systemManager.timestep / self.model.iterations)

                numToFarm = CAgentResourceAcquisitionFunctions.num_to_farm_phouse(farm_threshold, max_farm, self.model.random,
                          household[HouseholdPreferenceComponent].farm_utility, household[HouseholdPreferenceComponent].forage_utility)

            else:
                if not household.hasComponent(HouseholdRBAdaptiveComponent):
                    farm_threshold = (self.model.systemManager.timestep - AgentResourceAcquisitionSystem.delay_factor) / (self.model.iterations * 0.5)
                else:
                    farm_threshold = household[HouseholdRBAdaptiveComponent].percentage_to_farm
                numToFarm = CAgentResourceAcquisitionFunctions.num_to_farm(farm_threshold, max_farm, self.model.random)

            numToForage = max_farm - numToFarm
            hPos = (household[PositionComponent].x, household[PositionComponent].y)

            # Forage numToForage Cells
            totalForage = 0
            if numToForage > 0:

                foragableLand = CAgentResourceAcquisitionFunctions.generateNeighbours(hPos[0], hPos[1],
                                      self.model.environment.width, self.model.environment.height,
                                          AgentResourceAcquisitionSystem.max_acquisition_distance, owned_cells,
                                                                                  settlement_cells, water_cells)

                if len(foragableLand) < numToForage:
                    numToForage = len(foragableLand)

                # Sort Foraging land by how much vegetation it has
                foragableLand.sort(key=getVegetation, reverse=True)

                # Forage these patches
                for iForage in range(numToForage):
                    worker_diff = max(able_workers - AgentResourceAcquisitionSystem.farms_per_patch, 0)
                    workers = able_workers - worker_diff
                    able_workers = worker_diff

                    new_resources = CAgentResourceAcquisitionFunctions.forage(foragableLand[iForage], workers,
                                      vegetation_cells, AgentResourceAcquisitionSystem.forage_consumption_rate,
                                                          AgentResourceAcquisitionSystem.forage_production_multiplier,
                                                          AgentResourceAcquisitionSystem.farms_per_patch)

                    # Update Household Resources
                    household[ResourceComponent].resources += new_resources
                    totalForage += new_resources

                    self.logger.info(
                        'HOUSEHOLD.FORAGE: {} {} {}'.format(household.id, foragableLand[iForage], new_resources))

            # Forage numToFarm Cells
            totalFarm = 0
            if numToFarm > 0:
                # If ownedLand < patches to farm allocate more land to farm
                if len(household[ResourceComponent].ownedLand) < numToFarm * AgentResourceAcquisitionSystem.land_buffer:
                    self.acquire_land(household, numToFarm, owned_cells)

                # Select land patches
                farmableLand = [x for x in household[ResourceComponent].ownedLand]

                for i in range(numToFarm):

                    worker_diff = max(able_workers - AgentResourceAcquisitionSystem.farms_per_patch, 0)
                    workers = able_workers - worker_diff
                    able_workers = worker_diff

                    # Remove patches of land randomly
                    patchID = farmableLand.pop(self.model.random.randrange(0, len(farmableLand)))
                    new_resources = CAgentResourceAcquisitionFunctions.farm(patchID, workers, hPos, position_cells[patchID],
                                    sum(self.model.environment[GlobalEnvironmentComponent].temp)/12.0,
                                    AgentResourceAcquisitionSystem.max_acquisition_distance,
                                    AgentResourceAcquisitionSystem.moisture_consumption_rate,
                                    AgentResourceAcquisitionSystem.crop_gestation_period,
                                    AgentResourceAcquisitionSystem.farming_production_rate,
                                    AgentResourceAcquisitionSystem.farms_per_patch,
                                    height_cells, moisture_cells, slope_cells,
                                    self.model.environment[SoilMoistureComponent],
                                    self.model.environment[GlobalEnvironmentComponent], self.model.random)

                    household[ResourceComponent].resources += new_resources
                    totalFarm += new_resources

                    self.logger.info('HOUSEHOLD.FARM: {} {} {}'.format(household.id, patchID, new_resources))

            # Adjust the farm_preference based on number of resources acquired
            if is_phouse:
                AgentResourceAcquisitionSystem.adjust_farm_preference(household,
                                                                      totalForage + totalFarm,
                                                                      totalFarm/numToFarm if numToFarm != 0 else 0.0,
                                                                      totalForage/numToForage if numToForage != 0 else 0.0)

        # Update Environment Dataframe
        self.model.environment.cells.update({'vegetation': vegetation_cells, 'moisture': moisture_cells,
                                             'isOwned': owned_cells})

    @staticmethod
    def adjust_farm_preference(household: PreferenceHousehold, acquired_resources, farm_res_avg, forage_res_avg):
        # Set new prev_hunger
        household[HouseholdPreferenceComponent].prev_hunger = acquired_resources / household[ResourceComponent].required_resources()

        # Adjust forage utility
        household[HouseholdPreferenceComponent].forage_utility += household[HouseholdPreferenceComponent].learning_rate * (
            forage_res_avg - household[HouseholdPreferenceComponent].forage_utility
        )
        # Adjust Farm utility
        household[HouseholdPreferenceComponent].farm_utility += household[HouseholdPreferenceComponent].learning_rate * (
             farm_res_avg - household[HouseholdPreferenceComponent].farm_utility
        )


class AgentResourceTransferSystem(System, IDecodable, ILoggable):

    def __init__(self, id: str, model: Model, priority, load_decay):
        System.__init__(self, id, model, priority=priority)
        IDecodable.__init__(self)
        ILoggable.__init__(self, 'model.ARTS')

        self.load_decay = load_decay

    @staticmethod
    def decode(params: dict):
        return AgentResourceTransferSystem(params['id'], params['model'], params['priority'], params['load_decay'])

    def execute(self):

        # Decay Load
        for h in self.model.environment.getAgents():
            h[HouseholdRelationshipComponent].load *= self.load_decay

        # For each settlement:
        for settlement in [self.model.environment[SettlementRelationshipComponent].settlements[s] for s in
                           self.model.environment[SettlementRelationshipComponent].settlements]:

            for household in [self.model.environment.getAgent(h) for h in settlement.occupants]:

                # If resources < needed resources: ask for help
                if household[ResourceComponent].resources < household[ResourceComponent].required_resources():

                    resources_needed = household[ResourceComponent].required_resources() - household[ResourceComponent].resources
                    # Get auth relationships as primary providers
                    providers = self.model.environment[SettlementRelationshipComponent].get_all_auth(household)

                    # Get required resources
                    # Get help from superiors randomly
                    while len(providers) != 0 and resources_needed > 0:
                        provider = self.model.random.choice(providers)
                        providers.remove(provider)

                        if self.model.random.random() < provider[HouseholdRelationshipComponent].sub_resource_transfer_chance:
                            resource_given = AgentResourceTransferSystem.ask_for_resources(provider, resources_needed)
                            household[ResourceComponent].resources += resource_given

                            if resource_given > 0:
                                self.logger.info('HOUSEHOLD.RESOURCES.TRANSFER.SUCCESS.AUTH: {} {} {}'.format(
                                    household.id, provider.id, resource_given))
                            else:
                                self.logger.info('HOUSEHOLD.RESOURCES.TRANSFER.FAIL.AUTH: {} {}'.format(
                                    household.id, provider.id
                                ))

                            # Update amount of resources needed
                            resources_needed -= resource_given
                        else:
                            self.logger.info('HOUSEHOLD.RESOURCES.TRANSFER.REJECT.AUTH: {} {}'.format(
                                household.id, provider.id
                            ))

                    if resources_needed > 0:
                        # Get peers as secondary providers
                        providers = self.model.environment[SettlementRelationshipComponent].get_all_peer(household)

                        while len(providers) != 0 and resources_needed > 0:
                            provider = self.model.random.choice(providers)
                            providers.remove(provider)

                            if self.model.random.random() < provider[HouseholdRelationshipComponent].peer_resource_transfer_chance:

                                resource_given = AgentResourceTransferSystem.ask_for_resources(provider, resources_needed)
                                household[ResourceComponent].resources += resource_given

                                if resource_given > 0:
                                    self.logger.info('HOUSEHOLD.RESOURCES.TRANSFER.SUCCESS.PEER: {} {} {}'.format(
                                        household.id, provider.id, resource_given))
                                else:
                                    self.logger.info('HOUSEHOLD.RESOURCES.TRANSFER.FAIL.PEER: {} {}'.format(
                                        household.id, provider.id))

                                # Update amount of resources needed
                                resources_needed -= resource_given
                            else:
                                self.logger.info('HOUSEHOLD.RESOURCES.TRANSFER.REJECT.PEER: {} {}'.format(
                                    household.id, provider.id))

                    if resources_needed > 0:
                        # Get subordinates as tertiary providers
                        providers = self.model.environment[SettlementRelationshipComponent].get_all_sub(household)

                        while len(providers) != 0 and resources_needed > 0:
                            provider = self.model.random.choice(providers)
                            providers.remove(provider)

                            # Subordinates cannot say no to giving away excess resources

                            resource_given = AgentResourceTransferSystem.ask_for_resources(provider, resources_needed)
                            household[ResourceComponent].resources += resource_given

                            if resource_given > 0:
                                self.logger.info('HOUSEHOLD.RESOURCES.TRANSFER.SUCCESS.SUB: {} {} {}'.format(
                                    household.id, provider.id, resource_given))
                            else:
                                self.logger.info('HOUSEHOLD.RESOURCES.TRANSFER.FAIL.SUB: {} {}'.format(
                                    household.id, provider.id))

                            # Update amount of resources needed
                            resources_needed -= resource_given

    @staticmethod
    def ask_for_resources(h: Household, amount: int) -> int:

        # Return nothing if h does not have any resources to spare
        excess = h[ResourceComponent].excess_resources()
        if excess == 0:
            return 0

        # If excess does cover the
        if excess <= amount:
            h[ResourceComponent].resources -= excess
            h[HouseholdRelationshipComponent].load += excess

            return excess
        else:
            h[ResourceComponent].resources -= amount
            h[HouseholdRelationshipComponent].load += amount

            return amount


class AgentResourceConsumptionSystem(System, IDecodable, ILoggable):

    def __init__(self, id: str, model: Model,priority):
        System.__init__(self, id, model, priority=priority)
        IDecodable.__init__(self)
        ILoggable.__init__(self, 'model.RCS')

    @staticmethod
    def decode(params: dict):
        return AgentResourceConsumptionSystem(params['id'], params['model'], params['priority'])

    @staticmethod
    def consume(resources, required_resources) -> (int, float):
        # This is actually the inverse of hunger with 1.0 being completely 'full' and zero being 'starving'
        hunger = min(1.0, resources / required_resources)
        remaining_resources = max(0, resources - required_resources)

        return remaining_resources, hunger

    @staticmethod
    def ARCProcess(resComp: ResourceComponent) -> float:

        req_res = resComp.required_resources()
        rem_res, hunger = AgentResourceConsumptionSystem.consume(resComp.resources, req_res)
        resComp.hunger = hunger
        resComp.satisfaction += hunger
        resComp.resources = rem_res

        return req_res * hunger

    def execute(self):
        for stats in [(a.id, CAgentResourceConsumptionSystemFunctions.ARCProcess(a[ResourceComponent]))
                      for a in self.model.environment.getAgents()]:
            self.logger.info('HOUSEHOLD.CONSUME: {} {}'.format(stats[0], stats[1]))


class AgentPopulationSystem(System, IDecodable, ILoggable):
    """This system is responsible for managing agent reproduction, death and aging"""
    def __init__(self, id: str, model: Model, priority, birth_rate, death_rate, yrs_per_move, num_settlements,
                 cell_capacity):
        System.__init__(self, id, model, priority=priority)
        IDecodable.__init__(self)
        ILoggable.__init__(self, 'model.APS')

        self.birth_rate = birth_rate
        self.death_rate = death_rate

        self.model.environment.addComponent(SettlementRelationshipComponent(self.model.environment, model, yrs_per_move,
                                                                            cell_capacity))

        self.num_households = 0

        # Add Settlement Cell Map
        def settlement_generator(pos, cells):
            return -1

        model.environment.addCellComponent('isSettlement', settlement_generator)

        # Create the settlements
        for i in range(num_settlements):
            self.model.environment[SettlementRelationshipComponent].create_settlement()

    def split_household(self, household: Household):

        self.logger.info('HOUSEHOLD.SPLIT: {}'.format(household.id))

        if household.hasComponent(HouseholdPreferenceComponent):
            new_household = PreferenceHousehold(self.num_households, self.model, -1, 0.0)
        elif household.hasComponent(HouseholdRBAdaptiveComponent):
            new_household = RBAdaptiveHousehold(self.num_households, self.model, -1)
        else:
            new_household = Household(self.num_households, self.model, -1)

        self.model.environment.addAgent(new_household)
        # Add to settlement
        sID = household[HouseholdRelationshipComponent].settlementID
        self.model.environment[SettlementRelationshipComponent].add_household_to_settlement(new_household, sID)

        # Set household position
        h_pos = self.model.environment.cells['pos'][
            self.model.environment[SettlementRelationshipComponent].settlements[sID].pos[-1]]

        new_household[PositionComponent].x = h_pos[0]
        new_household[PositionComponent].y = h_pos[1]

        # Split Resources
        half_res = household[ResourceComponent].resources / 2.0
        household[ResourceComponent].resources -= half_res
        new_household[ResourceComponent].resources += half_res

        # Copy across hunger and satisfaction values
        new_household[ResourceComponent].hunger = household[ResourceComponent].hunger
        new_household[ResourceComponent].satisfaction = household[ResourceComponent].satisfaction

        # Split land
        num_to_split = len(household[ResourceComponent].ownedLand) // 2

        while num_to_split > 0:
            land_id = self.model.random.choice(
                household[ResourceComponent].ownedLand
            )

            household[ResourceComponent].release_land(land_id)
            new_household[ResourceComponent].claim_land(land_id)

            # Update environment layer
            self.model.environment.cells.at[land_id, 'isOwned'] = new_household.id

            num_to_split -= 1

        # Split Occupants
        able_count = household[ResourceComponent].able_workers()
        child_count = (len(household[ResourceComponent].occupants) - able_count) // 2
        able_count = able_count // 2

        able_individuals = [o for o in household[ResourceComponent].occupants
                if household[ResourceComponent].occupants[o].age >= ResourceComponent.age_of_maturity]

        children = [o for o in household[ResourceComponent].occupants
                if o not in able_individuals]

        # Move adults
        while able_count > 0:
            id = self.model.random.choice(
                able_individuals
            )

            household[ResourceComponent].move_occupant(id, new_household)
            able_individuals.remove(id)
            able_count -= 1

        # Move children
        while child_count > 0:
            id = self.model.random.choice(
                children
            )

            household[ResourceComponent].move_occupant(id, new_household)
            children.remove(id)
            child_count -= 1

        # Set Household Resource Trading Personality Types
        new_household[HouseholdRelationshipComponent].peer_resource_transfer_chance = household[
            HouseholdRelationshipComponent].peer_resource_transfer_chance
        new_household[HouseholdRelationshipComponent].sub_resource_transfer_chance = household[
            HouseholdRelationshipComponent].sub_resource_transfer_chance

        # Set Preference
        if new_household.hasComponent(HouseholdPreferenceComponent):
            new_household[HouseholdPreferenceComponent].prev_hunger = household[HouseholdPreferenceComponent].prev_hunger
            new_household[HouseholdPreferenceComponent].forage_utility = household[HouseholdPreferenceComponent].forage_utility
            new_household[HouseholdPreferenceComponent].farm_utility = household[HouseholdPreferenceComponent].farm_utility
            new_household[HouseholdPreferenceComponent].learning_rate = household[HouseholdPreferenceComponent].learning_rate

        if new_household.hasComponent(HouseholdRBAdaptiveComponent):
            new_household[HouseholdRBAdaptiveComponent].rainfall_memory = [
                x for x in household[HouseholdRBAdaptiveComponent].rainfall_memory
            ]

            new_household[HouseholdRBAdaptiveComponent].flood_memory = [
                x for x in household[HouseholdRBAdaptiveComponent].flood_memory
            ]

            new_household[HouseholdRBAdaptiveComponent].percentage_to_farm = household[HouseholdRBAdaptiveComponent].percentage_to_farm

        self.num_households += 1

        self.logger.info('CREATE.HOUSEHOLD: {} {} {}'.format(new_household.id, sID, self.model.environment[
            SettlementRelationshipComponent].settlements[sID].pos[-1]))

    def execute(self):
        toRem = []
        for household in self.model.environment.getAgents():

            # Reallocation check
            if self.model.systemManager.timestep != 0 and self.model.systemManager.timestep % self.model.environment[SettlementRelationshipComponent].yrs_per_move == 0:
                if self.model.random.random() > (household[ResourceComponent].satisfaction / (self.model.environment[SettlementRelationshipComponent].yrs_per_move-1)):

                    logging.debug('Moving Household: ' + str(household.id))
                    self.reallocate_agent(household)

                household[ResourceComponent].satisfaction = 0.0  # Reset hunger decision every 'yrs_per_move' steps

            # Birth Chance
            for i in range(household[ResourceComponent].able_workers()):
                if self.model.random.random() <= self.birth_rate:
                    household[ResourceComponent].add_occupant(household[ResourceComponent].get_next_id())
                    self.logger.info('HOUSEHOLD.BIRTH: {}'.format(household.id))

            # Split household if household reaches capacity
            if len(household[ResourceComponent].occupants) > ResourceComponent.carrying_capacity:
                logging.debug('Splitting Household: ' + str(household.id))
                self.split_household(household)


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
                self.logger.info('HOUSEHOLD.DEATH: {}'.format(household.id))

            if household[ResourceComponent].able_workers() == 0:  # Add households to the delete list
                toRem.append(household)

        # Delete empty households here
        for household in toRem:
            # Remove all land from ownership
            self.model.environment.cells.loc[(self.model.environment.cells.isOwned == household.id), 'isOwned'] = -1
            self.model.environment[SettlementRelationshipComponent].remove_household(household)
            # Delete Agent
            self.model.environment.removeAgent(household.id)

            if len(household[ResourceComponent].occupants) > 0:
                self.logger.info('REMOVE.HOUSEHOLD.ORPHANED: {}'.format(household.id))
            else:
                self.logger.info('REMOVE.HOUSEHOLD.EMPTY: {}'.format(household.id))

    def reallocate_agent(self, household: Household):
        # Get rid of all land ownership that household has
        self.model.environment.cells.loc[(self.model.environment.cells.isOwned == household.id), 'isOwned'] = -1
        household[ResourceComponent].ownedLand.clear()

        old_sid = household[HouseholdRelationshipComponent].settlementID
        if old_sid != -1:
            self.model.environment[SettlementRelationshipComponent].remove_household(household)

        if old_sid not in self.model.environment[SettlementRelationshipComponent].settlements:
            self.logger.info('REMOVE.SETTLEMENT.ABANDONED: {}'.format(old_sid))

        # Check for settlement with most wealth
        mostWealth = 0
        most_id = -1

        for settlement in [self.model.environment[SettlementRelationshipComponent].settlements[s] for s in
                           self.model.environment[SettlementRelationshipComponent].settlements
                           if s != old_sid]:

            # Ensure household can travel to the settlement before evaluating it
            dist = (household[PositionComponent].x - self.model.environment.cells['pos'][settlement.pos[0]][0]) ** 2 \
                   + (household[PositionComponent].y - self.model.environment.cells['pos'][settlement.pos[0]][1]) ** 2

            if dist > ResourceComponent.vision_square:
                continue

            avgWealth = self.model.environment[SettlementRelationshipComponent].getAverageSettlementWealth(settlement.id)
            if avgWealth > mostWealth:
                mostWealth = avgWealth
                most_id = settlement.id

        if mostWealth >= household[ResourceComponent].required_resources():
            # Move to this settlement
            self.model.environment.getComponent(SettlementRelationshipComponent).add_household_to_settlement(household, most_id)
            new_x, new_y = self.model.environment.cells['pos'][self.model.environment[SettlementRelationshipComponent].settlements[most_id].pos[-1]]
            household[PositionComponent].x = new_x
            household[PositionComponent].y = new_y

            self.logger.info('HOUSEHOLD.MOVE.SETTLEMENT: {} {} {} {}'.format(household.id, old_sid, most_id,
                                    self.model.environment[SettlementRelationshipComponent].settlements[most_id].pos[-1]))

        else:
            # Assign new household position if it chooses to not move to an existing settlement

            hPos = (household[PositionComponent].x, household[PositionComponent].y)
            viable_land = [x for x in self.model.environment.getNeighbours(hPos,
                     radius=int(ResourceComponent.vision_square ** .5))
                             if self.model.environment.cells['isOwned'][x] == -1
                                 and not self.model.environment.cells['isWater'][x]
                                 and self.model.environment.cells['isSettlement'][x] == -1
                                 and self.model.environment.cells['slope'][x] > 0.0]

            vegetation_cells = self.model.environment.cells['vegetation']
            def getVegetation(loc):
                return vegetation_cells[loc]

            viable_land.sort(key=getVegetation, reverse=True)

            new_unq_id = viable_land[0]
            new_x, new_y = self.model.environment.cells['pos'][new_unq_id]

            # Create a new Settlement
            sttlID = self.model.environment.getComponent(SettlementRelationshipComponent).create_settlement()
            self.model.environment[SettlementRelationshipComponent].settlements[sttlID].pos.append(new_unq_id)
            self.logger.info('CREATE.SETTLEMENT: {} {}'.format(sttlID, new_unq_id))

            self.model.environment.cells.at[new_unq_id, 'isSettlement'] = sttlID

            # Move House and add it to settlement
            household[PositionComponent].x = new_x
            household[PositionComponent].y = new_y
            self.model.environment.getComponent(SettlementRelationshipComponent).add_household_to_settlement(household, sttlID)

            self.logger.info('HOUSEHOLD.MOVE.RANDOM: {} {} {} {}'.format(household.id, old_sid, sttlID, new_unq_id))

    @staticmethod
    def decode(params: dict):
        return AgentPopulationSystem(params['id'], params['model'], params['priority'], params['birth_rate'],
                                     params['death_rate'], params['yrs_per_move'], params['init_settlements'],
                                     params['cell_capacity'])


class AgentRBAdaptationSystem(System, IDecodable, ILoggable):

    # Cumulative Moving Average
    rainfall_CMA = 0.0
    flood_CMA = 0.0

    per_severity_index = [[0.7, 0.2, 0.5],
                          [0.4, 0.1, 0.1]]  # 1st is RAINFALL and 2nd is FLOOD

    def __init__(self,id: str, model: Model, priority: int):
        System.__init__(self, id, model, priority=priority)
        IDecodable.__init__(self)
        ILoggable.__init__(self, 'model.RBAS')

    def execute(self):

        # Update CMAs
        ge_comp = self.model.environment[GlobalEnvironmentComponent]
        rf_mean = np.mean(ge_comp.rainfall)
        AgentRBAdaptationSystem.flood_CMA += (ge_comp.flood - AgentRBAdaptationSystem.flood_CMA) / (self.model.systemManager.timestep + 1)
        AgentRBAdaptationSystem.rainfall_CMA += (rf_mean - AgentRBAdaptationSystem.rainfall_CMA) / (self.model.systemManager.timestep + 1)

        for agent in self.model.environment.getAgents():

            adapt_comp = agent[HouseholdRBAdaptiveComponent]

            # Update memories with some error margin
            adapt_comp.update_rainfall_memory(rf_mean + 0.05 * self.model.random.randrange(-1.0, 1.0) * rf_mean)
            adapt_comp.update_flood_memory(ge_comp.flood + 0.05 * self.model.random.randrange(-1.0, 1.0) * ge_comp.flood)

            # Need to build memory before making decisions.
            if self.model.systemManager.timestep < HouseholdRBAdaptiveComponent.yrs_to_look_back:
                return

            res_comp = agent[ResourceComponent]
            # Calculate Risk Appraisal

            deltas = [
                abs(sum([adapt_comp.rainfall_memory[x] * HouseholdRBAdaptiveComponent.yr_look_back_weights[x]
                    for x in range(HouseholdRBAdaptiveComponent.yrs_to_look_back)]) + (0.01 * self.model.random.random()
                                       ) - AgentRBAdaptationSystem.rainfall_CMA) / AgentRBAdaptationSystem.rainfall_CMA,
                abs(sum([adapt_comp.flood_memory[x] * HouseholdRBAdaptiveComponent.yr_look_back_weights[x]
                    for x in range(HouseholdRBAdaptiveComponent.yrs_to_look_back)]) + (0.01 * self.model.random.random()
                                           ) - AgentRBAdaptationSystem.flood_CMA) / AgentRBAdaptationSystem.flood_CMA
            ]

            # If Rainfall delta is less than the flood delta, use the flood delta
            sev_index = deltas[0] if deltas[0] > deltas[1] else deltas[1]
            index = 0 if deltas[0] > deltas[1] else 1
            # Set severity index
            if sev_index < 0:
                sev_index = 0
            elif sev_index > 0:
                sev_index = 2
            else:
                sev_index = 1

            # Now we can calculate the severity value
            severity = deltas[index] * AgentRBAdaptationSystem.per_severity_index[index][sev_index]

            if severity > 1.0:
                severity = 1.0
            elif severity < 0.0:
                severity = 0.0

            # Calculate risk appraisal
            risk_appraisal = 0.6 * severity + 0.4 * self.model.random.random()

            # Calculate Adaptation Appraisal
            w_age = 1.0 - (0.12 / (0.12 + (min(50.0, res_comp.average_age()) / 50.0) ** 3))
            # Here we use household capacity
            w_hh_size = 1.0 - (0.12 / (0.12 + (
                    min(ResourceComponent.carrying_capacity, res_comp.able_workers()
                        ) / ResourceComponent.carrying_capacity) ** 3))

            adaptation_efficancy = 0.55 * w_age + 0.45 * w_hh_size + (0.2 - 0.3 * self.model.random.random())

            w_wealth = 1.0 / (1.0 + math.exp(-3.0 * ((res_comp.resources / res_comp.required_resources()) - 0.5)))

            self_efficancy = 0.3 * w_wealth + 0.6 * adapt_comp.percentage_to_farm + (0.1 - 0.2 * self.model.random.random())

            adaptation_appraisal = 0.5 * (adaptation_efficancy + self_efficancy)

            if adaptation_appraisal < 0.0:
                adaptation_appraisal = 0.0
            elif adaptation_appraisal > 1.0:
                adaptation_appraisal = 1.0

            # Calculate Adaptation Intention
            # Note: There is no adaptation cost in this model
            r = HouseholdRBAdaptiveComponent.risk_elasticity * risk_appraisal
            p = adaptation_appraisal * (1 - HouseholdRBAdaptiveComponent.cognitive_bias)

            adaptation_intention = p - r

            adaptation_modifier = 0.0  # Assume Maladaptation
            # Now determine if successful adaptation vs. maladaptation occurs
            if adaptation_intention > HouseholdRBAdaptiveComponent.adaptation_intention_threshold:
                # Adaptation Occurs
                adaptation_modifier = HouseholdRBAdaptiveComponent.learning_rate
                self.logger.info('HOUSEHOLD.ADAPTATION.INTENDED: {}'.format(agent.id))
            elif self.model.random.random() < 0.01:  # Described as Ingenuity Change
                adaptation_modifier = HouseholdRBAdaptiveComponent.learning_rate * self.model.random.random()
                self.logger.info('HOUSEHOLD.ADAPTATION.INGENUITY: {}'.format(agent.id))

            # Get Experience
            h_ids = self.model.environment[SettlementRelationshipComponent].settlements[agent[HouseholdRelationshipComponent].settlementID].occupants
            hs = [self.model.environment.getAgent(h) for h in h_ids]

            adaptation_experience_modifier = statistics.mean([
                h[HouseholdRBAdaptiveComponent].percentage_to_farm for h in hs
            ]) * HouseholdRBAdaptiveComponent.learning_rate

            peer_trade_modifier = (statistics.mean(
                [h[HouseholdRelationshipComponent].peer_resource_transfer_chance for h in hs]
            ) - agent[HouseholdRelationshipComponent].peer_resource_transfer_chance) * HouseholdRBAdaptiveComponent.learning_rate

            sub_trade_modifier = (statistics.mean(
                [h[HouseholdRelationshipComponent].sub_resource_transfer_chance for h in hs]
            ) - agent[HouseholdRelationshipComponent].sub_resource_transfer_chance) * HouseholdRBAdaptiveComponent.learning_rate

            # Update adaptation value
            adapt_comp.percentage_to_farm += adaptation_modifier * adaptation_experience_modifier + 0.001 * self.model.random.random()

            if adapt_comp.percentage_to_farm < 0.0:
                adapt_comp.percentage_to_farm = 0.0
            elif adapt_comp.percentage_to_farm > 1.0:
                adapt_comp.percentage_to_farm = 1.0

            agent[HouseholdRelationshipComponent].peer_resource_transfer_chance += peer_trade_modifier
            agent[HouseholdRelationshipComponent].sub_resource_transfer_chance += sub_trade_modifier

    @staticmethod
    def decode(params: dict):
        return AgentRBAdaptationSystem(params['id'], params['model'], params['priority'])


# Collectors

class AgentCollector(Collector, IDecodable):

    def __init__(self, id: str, model: Model):
        super().__init__(id, model)

    @staticmethod
    def gini_coefficient(x):
        """Compute Gini coefficient of array of values"""
        diffsum = 0
        for i, xi in enumerate(x[:-1], 1):
            diffsum += np.sum(np.abs(xi - x[i:]))
        return diffsum / (len(x) ** 2 * np.mean(x))

    def collect(self):

        agents = self.model.environment.getAgents()
        self.records.append({})
        for agent in agents:
            self.records[self.model.systemManager.timestep][agent.id] = {
                    'resources': agent[ResourceComponent].resources,
                    'population': len(agent[ResourceComponent].occupants),
                    'satisfaction': agent[ResourceComponent].satisfaction,
                    'load': agent[HouseholdRelationshipComponent].load
                }
        self.records[self.model.systemManager.timestep]['total'] = {
            'resources': sum([self.records[self.model.systemManager.timestep][x]['resources'] for x in self.records[self.model.systemManager.timestep]]),
            'population': sum([self.records[self.model.systemManager.timestep][x]['population'] for x in self.records[self.model.systemManager.timestep]]),
            'satisfaction': sum([self.records[self.model.systemManager.timestep][x]['satisfaction'] for x in self.records[self.model.systemManager.timestep]]),
            'load': sum([self.records[self.model.systemManager.timestep][x]['load'] for x in self.records[self.model.systemManager.timestep]])
        }

    @staticmethod
    def decode(params: dict):
        return AgentCollector(params['id'], params['model'])


class ActionComponent(Component):

    def __init__(self, agent, model: Model):
        super().__init__(agent, model)

        self.forage_count = 0
        self.farm_count = 0
        self.loan_count = 0


class ActionCollector(Collector, IDecodable):

    def __init__(self, id: str, model: Model):
        super().__init__(id, model)

        self.model.environment.addComponent(ActionComponent(self.model.environment, self.model))

    def collect(self):
        ac = self.model.environment[ActionComponent]

        self.records.append({
            'forage_count': ac.forage_count,
            'farm_count': ac.farm_count,
            'loan_count': ac.loan_count
        })

        ac.forage_count = 0
        ac.farm_count = 0
        ac.loan_count = 0

    @staticmethod
    def decode(params: dict):
        return ActionCollector(params['id'], params['model'])


class SettlementHouseholdCollector(FileCollector, IDecodable):

    def __init__(self, id: str, model: Model, filename: str, write: str):
        super().__init__(id, model, filename, clear_records_on_write=False)

        self.write = write

    def write_records(self):

        if self.model.systemManager.timestep % 5 != 0 or not self.write:
            return

        file = open(self.filename, self.filemode)

        isOwnedCells = self.model.environment.cells['isOwned']
        isSettleCells = self.model.environment.cells['isSettlement']
        isWaterCells = self.model.environment.cells['isWater']
        toWrite = ''
        for i in range(len(isOwnedCells)):

            if isWaterCells[i]:
                toWrite += '-1 '
            elif isOwnedCells[i] > -1:
                toWrite += '1 '
            elif isSettleCells[i] > -1:
                toWrite += '2 '
            else:
                toWrite += '0 '

        file.write(toWrite + '\n')
        file.close()

    @staticmethod
    def decode(params: dict):
        return SettlementHouseholdCollector(params['id'], params['model'], params['filename'], params['write'])


class AgentSnapshotCollector(Collector):

    def __init__(self, id: str, model, file_name, frequency=1):
        super().__init__(id, model, frequency=frequency)

        self.file_name = file_name

    def collect(self):

        toWrite = []

        for household in self.model.environment.getAgents():
            toWrite.append(household.jsonify())

        with open(self.file_name + '/iteration_{}.json'.format(self.model.systemManager.timestep), 'w') as outfile:
            json.dump(toWrite, outfile, indent=4)


class SettlementSnapshotCollector(Collector):

    def __init__(self, id: str, model, file_name, frequency: int = 1):
        super().__init__(id, model, frequency=frequency)

        self.file_name = file_name

    def collect(self):

        toWrite = []

        for sid in self.model.environment[SettlementRelationshipComponent].settlements:
            srComp = self.model.environment[SettlementRelationshipComponent]
            generated_dict = srComp.settlements[sid].jsonify()
            generated_dict['wealth'] = srComp.getSettlementWealth(sid)
            generated_dict['load'] = srComp.getSettlementLoad(sid)
            generated_dict['population'] = srComp.getSettlementPopulation(sid)
            generated_dict['farm_utility'] = srComp.getSettlementFarmUtility(sid)
            generated_dict['forage_utility'] = srComp.getSettlementForageUtility(sid)
            generated_dict['percentage_to_farm'] = srComp.getSettlementPercentageToFarm(sid)
            toWrite.append(generated_dict)

        with open(self.file_name + '/iteration_{}.json'.format(self.model.systemManager.timestep), 'w') as outfile:
            json.dump(toWrite, outfile, indent=4)

