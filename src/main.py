import numpy as np
import ECAgent.Visualization as vis
import ECAgent.Environments as env
from ECAgent.Core import *


class EgyptModel(Model):

    def __init__(self, dimensions: int):
        super().__init__(None)
        self.environment = env.GridWorld(dimensions, dimensions, self)


class CellComponent(Component):

    def __init__(self, agent, model):
        super().__init__(agent, model)

        self.height = 0
        self.slope = 0


dim = 720
cellSize = 200
print("Creating Model...")
model = EgyptModel(dim)
print("Loading Elevation Data")
# Get elevation Data
elevation_data = np.loadtxt("./resources/quena_map.asc", skiprows=5)


# Set elevation Data
def elevation_generator_functor(cell: Agent):
    cell.addComponent(CellComponent(cell, cell.model))
    cell[CellComponent].height = elevation_data[cell[env.PositionComponent].x][cell[env.PositionComponent].y]


model.environment.addCellComponent(elevation_generator_functor)

print("Setting Slope Data...")
# Set Slope data
for cell in model.environment.cells:
    maxSlope = 0.0  # Set base slope value
    for neighbour in model.environment.getNeighbours(cell):
        slopeVal = np.degrees(np.arctan(abs(neighbour[CellComponent].height - cell[CellComponent].height)/cellSize))
        maxSlope = max(maxSlope, slopeVal)
    # Set slopVal
    cell[CellComponent].slope = maxSlope

print("Calculating aquifer data...")
# Set aquifer data
# Visualize

webApp = vis.VisualInterface("Predynastic Egypt", model)

# Create heightmap data

heightmap = []
slopemap = []
canFarm = []
for x in range(dim):
    row = []
    slopeRow = []
    canSlopeRow = []
    for y in range(dim):
        row.append(model.environment.getCell(x, y)[CellComponent].height)
        slopeRow.append(model.environment.getCell(x, y)[CellComponent].slope)
        if model.environment.getCell(x, y)[CellComponent].slope <= 45:
            canSlopeRow.append(1)
        else:
            canSlopeRow.append(0)
    heightmap.append(row)
    slopemap.append(slopeRow)
    canFarm.append(canSlopeRow)

vis.addGraph(webApp, 'elevation_heatmap',
             vis.createHeatMap("Elevation Data", heightmap, 'xCoord', 'yCoord', 'Elevation', height=1000))

vis.addGraph(webApp, 'slope_heatmap',
             vis.createHeatMap("Slope Data", slopemap, 'xCoord', 'yCoord', 'Slope (degrees)', height=1000))

vis.addGraph(webApp, 'farm_heatmap',
             vis.createHeatMap("Can Farm", canFarm, 'xCoord', 'yCoord', 'Can Farm', height=1000))

if __name__ == '__main__':
    webApp.app.run_server()
