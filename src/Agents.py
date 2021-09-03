import VegetationModel as VM
import math
import numpy as np

from ECAgent.Core import *
from ECAgent.Decode import IDecodable
from ECAgent.Environments import PositionComponent, discreteGridPosToID
from ECAgent.Collectors import Collector, FileCollector


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

    def is_aquaintance(self, h):
        """Returns true if household h is in the same settlement as household h"""
        return self.settlementID == h[HouseholdRelationshipComponent].settlementID

    def is_auth(self, h):
        """Returns true if household self has an authority relationship over household h"""

        if h is self or self.load == 0:
            return False

        h_load = h[HouseholdRelationshipComponent].load
        return self.is_aquaintance(h) and ((self.load - h_load)/max(self.load, h_load) - HouseholdRelationshipComponent.load_difference) > 0

    def is_peer(self, h):
        """Returns true if household self has a peer relationship with household h"""

        if h is self:
            return False

        h_load = h[HouseholdRelationshipComponent].load

        if h_load == 0 and self.load == 0:
            return True  # Ensure no division by error

        return self.is_aquaintance(h) and (abs(self.load - h_load)/max(self.load, h_load) + HouseholdRelationshipComponent.load_difference) > 0


class HouseholdPreferenceComponent(Component):

    learning_rate_range = [0.0, 1.0]

    def __init__(self, agent, model: Model, init_preference: float = 0.0):
        super().__init__(agent, model)

        self.forage_utility = init_preference
        self.farm_utility = 0.0
        self.learning_rate = model.random.uniform(HouseholdPreferenceComponent.learning_rate_range[0],
                                                 HouseholdPreferenceComponent.learning_rate_range[1])

        self.prev_hunger = 1.0


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


class Settlement:
    def __init__(self, id):
        self.id = id
        self.pos = []
        self.occupants = []


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

    def get_all_peer(self, h: Household):
        """Returns all of the households which have a peer relationship over household h"""
        return [self.model.environment.getAgent(x) for x in
                self.settlements[h[HouseholdRelationshipComponent].settlementID].occupants
                if self.model.environment.getAgent(x)[HouseholdRelationshipComponent].is_peer(h)]


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

        # Get a list of all the available patches of land
        own_cells = self.model.environment.cells['isOwned']

        available_land = []
        past_iter = 0
        while len(available_land) < new_land:
            for land_id in household[ResourceComponent].ownedLand:

                available_land +=[x for x in self.model.environment.getNeighbours(self.model.environment.cells['pos'][land_id])
                    if x not in available_land and own_cells[x] == -1 and not self.model.environment.cells['isWater'][x]
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

        # If we did not allocate enough land we can call the function again
        if len(household[ResourceComponent].ownedLand) < target:
            self.acquire_land(household, target)

    def farm(self, patch_id, workers, house_pos: (int, int)):
        # Calculate penalties

        coords = self.model.environment.cells['pos'][patch_id]

        dst = max(abs(coords[0] - house_pos[0]), abs(coords[1] - house_pos[1]))
        dst_penalty = 1.0 if dst <= AgentResourceAcquisitionSystem.max_acquisition_distance else 1.0 / (dst-AgentResourceAcquisitionSystem.max_acquisition_distance)

        tmp_Penalty = VM.VegetationGrowthSystem.tempPenalty(
            sum(self.model.environment.getComponent(VM.GlobalEnvironmentComponent).temp)/12.0, self.model
        )
        wtr_penalty, moisture_remain = VM.VegetationGrowthSystem.waterPenalty(self.model.environment.cells['moisture'][patch_id],
            AgentResourceAcquisitionSystem.moisture_consumption_rate/AgentResourceAcquisitionSystem.crop_gestation_period, 1.0)
        # Calculate Crop Yield

        if self.model.systemManager.systems['SMS'].is_flooded(patch_id):
            wtr_penalty = 1.0

        crop_yield = AgentResourceAcquisitionSystem.farming_production_rate * wtr_penalty * tmp_Penalty * (workers/AgentResourceAcquisitionSystem.farms_per_patch)

        # Adjust soil moisture if cell not flooded
        if not self.model.systemManager.systems['SMS'].is_flooded(patch_id):
            self.model.environment.cells.at[patch_id, 'moisture'] = moisture_remain

        return int(crop_yield * dst_penalty)

    def forage(self, patch_id, workers) -> (int, int):
        veg_diff = max(self.model.environment.cells['vegetation'][patch_id]
                       - AgentResourceAcquisitionSystem.forage_consumption_rate * workers / AgentResourceAcquisitionSystem.farms_per_patch
                       , 0.0)
        return ((self.model.environment.cells['vegetation'][patch_id] - veg_diff) \
                * AgentResourceAcquisitionSystem.forage_production_multiplier, veg_diff)

    def gather_resources(self, household: Household, patch_id: int, workers: int,farm: bool) -> int:
        if farm:
            new_resources = self.farm(patch_id, workers, (household[PositionComponent].x, household[PositionComponent].y))
            household[ResourceComponent].resources += new_resources
            self.model.environment[ActionComponent].farm_count += 1
        else:
            new_resources, veg_diff = self.forage(patch_id, workers)
            household[ResourceComponent].resources += new_resources
            self.model.environment.cells.at[patch_id, 'vegetation'] = veg_diff
            self.model.environment[ActionComponent].forage_count += 1

        return new_resources

    def execute(self):
        for household in self.model.environment.getAgents():
            # Determine how many patches a household can farm
            able_workers = household[ResourceComponent].able_workers()
            max_farm = math.ceil(able_workers / AgentResourceAcquisitionSystem.farms_per_patch)
            numToFarm = 0
            res_before = household[ResourceComponent].resources

            is_phouse = household.hasComponent(HouseholdPreferenceComponent)

            if is_phouse:
                farm_threshold = household[HouseholdPreferenceComponent].prev_hunger

                for index in range(max_farm):
                    is_farm_max = household[HouseholdPreferenceComponent].farm_utility > household[HouseholdPreferenceComponent].forage_utility

                    # A satisfied house has a hunger of 1.0
                    if self.model.random.random() > farm_threshold:
                        if is_farm_max:
                            numToFarm += 1
                    else:
                        numToFarm += self.model.random.randint(0, 1)

            else:
                # TODO Allow curve specialization
                farm_threshold = self.model.systemManager.timestep / self.model.iterations
                for index in range(max_farm):
                    if self.model.random.random() < farm_threshold:
                        numToFarm +=1

            numToForage = max_farm - numToFarm
            hPos = (household[PositionComponent].x, household[PositionComponent].y)

            foragableLand = []
            if numToForage > 0:
                foragableLand = [x for x in self.model.environment.getNeighbours(hPos, radius=AgentResourceAcquisitionSystem.max_acquisition_distance)
                                 if self.model.environment.cells['isOwned'][x] == -1
                                 and not self.model.environment.cells['isWater'][x]
                                 and self.model.environment.cells['isSettlement'][x] == -1]

                if len(foragableLand) < numToForage:
                    numToForage = len(foragableLand)
                    numToFarm = max_farm - numToForage

            # If ownedLand < patches to farm allocate more land to farm
            if len(household[ResourceComponent].ownedLand) < numToFarm * AgentResourceAcquisitionSystem.land_buffer:
                self.acquire_land(household, numToFarm)

            # Select land patches
            farmableLand = [x for x in household[ResourceComponent].ownedLand]

            totalFarm = 0
            for i in range(numToFarm):

                worker_diff = max(able_workers - AgentResourceAcquisitionSystem.farms_per_patch, 0)
                workers = able_workers - worker_diff
                able_workers = worker_diff

                patchID = farmableLand.pop(self.model.random.randrange(0, len(farmableLand)))  # Remove patches of land randomly
                totalFarm += self.gather_resources(household, patchID, workers, True)  # The choice of farming or foraging is random

            totalForage = 0
            if numToForage > 0:

                vegetation_cells = self.model.environment.cells['vegetation']

                def getVegetation(location):
                    return vegetation_cells[location]

                # Sort Foraging land by how much vegetation it has
                foragableLand.sort(key=getVegetation, reverse=True)

                # Forage these patches
                for iForage in range(max_farm - numToFarm):
                    worker_diff = max(able_workers - AgentResourceAcquisitionSystem.farms_per_patch, 0)
                    workers = able_workers - worker_diff
                    able_workers = worker_diff

                    totalForage += self.gather_resources(household, foragableLand[iForage], workers, False)

            # Adjust the farm_preference based on number of resources acquired

            if household.hasComponent(HouseholdPreferenceComponent):
                AgentResourceAcquisitionSystem.adjust_farm_preference(household,
                                                                      totalForage + totalFarm,
                                                                      totalFarm/numToFarm if numToFarm != 0 else 0.0,
                                                                      totalForage/numToForage if numToForage != 0 else 0.0)

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


class AgentResourceTransferSystem(System, IDecodable):

    def __init__(self, id: str, model: Model, priority):
        super().__init__(id, model, priority=priority)

    @staticmethod
    def decode(params: dict):
        return AgentResourceTransferSystem(params['id'], params['model'], params['priority'])

    def execute(self):
        # For each settlement:
        for settlement in [self.model.environment[SettlementRelationshipComponent].settlements[s] for s in
                           self.model.environment[SettlementRelationshipComponent].settlements]:

            for household in [self.model.environment.getAgent(h) for h in settlement.occupants]:

                # If resources < needed resources: ask for help
                if household[ResourceComponent].resources < household[ResourceComponent].required_resources():

                    resources_needed = household[ResourceComponent].required_resources() - household[ResourceComponent].resources
                    loan_flag = resources_needed
                    # Get auth relationships as primary providers
                    providers = self.model.environment[SettlementRelationshipComponent].get_all_auth(household)

                    # Get required resources
                    # Get help from superiors randomly
                    while len(providers) != 0 and resources_needed > 0:
                        provider = self.model.random.choice(providers)
                        providers.remove(provider)

                        resource_given = AgentResourceTransferSystem.ask_for_resources(provider, resources_needed)
                        household[ResourceComponent].resources += resource_given

                        # Update amount of resources needed
                        resources_needed -= resource_given

                    if resources_needed > 0:
                        # Get peers as secondary providers
                        providers = self.model.environment[SettlementRelationshipComponent].get_all_peer(household)

                        while len(providers) != 0 and resources_needed > 0:
                            provider = self.model.random.choice(providers)
                            providers.remove(provider)

                            resource_given = AgentResourceTransferSystem.ask_for_resources(provider, resources_needed)
                            household[ResourceComponent].resources += resource_given

                            # Update amount of resources needed
                            resources_needed -= resource_given

                    if loan_flag != resources_needed:
                        self.model.environment[ActionComponent].loan_count += 1

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


class AgentResourceConsumptionSystem(System, IDecodable):

    def __init__(self, id: str, model: Model,priority):
        super().__init__(id, model, priority=priority)

    @staticmethod
    def decode(params: dict):
        return AgentResourceConsumptionSystem(params['id'], params['model'], params['priority'])

    def execute(self):
        for agent in self.model.environment.getAgents():
            resComp = agent[ResourceComponent]
            # This is actually the inverse of hunger with 1.0 being completely 'full' and zero being 'starving'
            resComp.hunger = min(1.0, resComp.resources / resComp.required_resources())
            resComp.satisfaction += resComp.hunger
            resComp.resources = max(0,
                 resComp.resources - resComp.required_resources())


class AgentPopulationSystem(System, IDecodable):
    """This system is responsible for managing agent reproduction, death and aging"""
    def __init__(self, id: str, model: Model, priority, birth_rate, death_rate, yrs_per_move, num_settlements,
                 cell_capacity):
        super().__init__(id, model, priority=priority)

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

        if household.hasComponent(HouseholdPreferenceComponent):
            new_household = PreferenceHousehold(self.num_households, self.model, -1, 0.0)
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


        # Set Preference
        if new_household.hasComponent(HouseholdPreferenceComponent):
            new_household[HouseholdPreferenceComponent].prev_hunger = household[HouseholdPreferenceComponent].prev_hunger
            new_household[HouseholdPreferenceComponent].forage_utility = household[HouseholdPreferenceComponent].forage_utility
            new_household[HouseholdPreferenceComponent].farm_utility = household[HouseholdPreferenceComponent].farm_utility
            new_household[HouseholdPreferenceComponent].learning_rate = household[HouseholdPreferenceComponent].learning_rate

        self.num_households += 1

    def execute(self):
        toRem = []
        for household in self.model.environment.getAgents():

            # Reallocation check
            if self.model.systemManager.timestep != 0 and self.model.systemManager.timestep % self.model.environment[SettlementRelationshipComponent].yrs_per_move == 0:
                if self.model.random.random() > (household[ResourceComponent].satisfaction / (self.model.environment[SettlementRelationshipComponent].yrs_per_move-1)):
                    if self.model.debug:
                        print('Moving Household: ' + str(household.id))
                    self.reallocate_agent(household)
                household[ResourceComponent].satisfaction = 0.0  # Reset hunger decision every 'yrs_per_move' steps

            # Birth Chance
            for i in range(household[ResourceComponent].able_workers()):
                if self.model.random.random() <= self.birth_rate:
                    household[ResourceComponent].add_occupant(household[ResourceComponent].get_next_id())

            # Split household if household reaches capacity
            if len(household[ResourceComponent].occupants) > ResourceComponent.carrying_capacity:
                if self.model.debug:
                    print('Splitting Household: ' + str(household.id))
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

            if household[ResourceComponent].able_workers() == 0:  # Add households to the delete list
                toRem.append(household)

        # Delete empty households here
        for household in toRem:
            # Remove all land from ownership
            self.model.environment.cells.loc[(self.model.environment.cells.isOwned == household.id), 'isOwned'] = -1
            self.model.environment[SettlementRelationshipComponent].remove_household(household)
            # Delete Agent
            self.model.environment.removeAgent(household.id)

    def reallocate_agent(self, household: Household):
        # Get rid of all land ownership that household has
        self.model.environment.cells.loc[(self.model.environment.cells.isOwned == household.id), 'isOwned'] = -1
        household[ResourceComponent].ownedLand.clear()

        old_sid = household[HouseholdRelationshipComponent].settlementID
        if old_sid != -1:
            self.model.environment[SettlementRelationshipComponent].remove_household(household)

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

        else:
            # Assign new household position if it chooses to not move to an existing settlement
            print('Moving Household to random location')
            hPos = (household[PositionComponent].x, household[PositionComponent].y)
            viable_land = [x for x in self.model.environment.getNeighbours(hPos,
                     radius=int(ResourceComponent.vision_square ** .5))
                             if self.model.environment.cells['isOwned'][x] == -1
                                 and not self.model.environment.cells['isWater'][x]
                                 and self.model.environment.cells['isSettlement'][x] == -1]

            vegetation_cells = self.model.environment.cells['vegetation']
            def getVegetation(loc):
                return vegetation_cells[loc]

            viable_land.sort(key=getVegetation, reverse=True)

            new_unq_id = viable_land[0]
            new_x, new_y = self.model.environment.cells['pos'][new_unq_id]

            # Create a new Settlement
            sttlID = self.model.environment.getComponent(SettlementRelationshipComponent).create_settlement()
            self.model.environment[SettlementRelationshipComponent].settlements[sttlID].pos.append(new_unq_id)

            # Move House and add it to settlement
            household[PositionComponent].x = new_x
            household[PositionComponent].y = new_y
            self.model.environment.getComponent(SettlementRelationshipComponent).add_household_to_settlement(household, sttlID)



    @staticmethod
    def decode(params: dict):
        return AgentPopulationSystem(params['id'], params['model'], params['priority'], params['birth_rate'],
                                     params['death_rate'], params['yrs_per_move'], params['init_settlements'],
                                     params['cell_capacity'])


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