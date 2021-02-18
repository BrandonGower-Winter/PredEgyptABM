from main import *

import ECAgent.Visualization as vis


if __name__ == '__main__':
    # Visualize

    webApp = vis.VisualInterface("Predynastic Egypt", model, 1.0)

    # Create heightmap data

    heightmap = []
    slopemap = []
    canFarm = []
    soilmap = []
    soilmoisture = []

    print('Generating Data Points')

    xdim = model.environment.width
    ydim = model.environment.height

    for x in range(xdim):
        row = []
        slopeRow = []
        canSlopeRow = []
        soilRow = []
        moistRow = []

        for y in range(ydim):

            id = env.discreteGridPosToID(x, y, xdim)
            row.append(model.environment.cells['height'][id])
            slopeRow.append(model.environment.cells['slope'][id])

            if model.environment.cells['isWater'][id]:
                canSlopeRow.append(-1)
            elif model.environment.cells['slope'][id] > 45:
                canSlopeRow.append(0)
            else:
                canSlopeRow.append(1)

            soilRow.append(model.environment.cells['sand_content'][id])
            moistRow.append(model.environment.cells['moisture'][id])

        heightmap.append(row)
        slopemap.append(slopeRow)
        canFarm.append(canSlopeRow)
        soilmap.append(soilRow)
        soilmoisture.append(moistRow)

    layout_dict = {
        'height': 1000,
        'xaxis': dict(title="xCoord"),
        'yaxis': dict(title="yCoord"),
    }


    def update_soil_heatmap(figure):
        moisture = []
        for x in range(model.environment.width):
            row = []
            for y in range(model.environment.height):
                id = discreteGridPosToID(x, y, model.environment.width)
                row.append(model.environment.cells['moisture'][id])

            moisture.append(row)
        return vis.createHeatMapGL('Soil Moisture', moisture,
                                   heatmap_kwargs={'colorbar': dict(title="Soil Moisture(mm)")},
                                   layout_kwargs=layout_dict)

    webApp.addDisplay(vis.createLiveGraph('soil_moisture', vis.createHeatMapGL(
        "Soil Moisture", soilmoisture,
        heatmap_kwargs={'colorbar': dict(title="Soil Moisture(mm)")},
        layout_kwargs=layout_dict), webApp, update_soil_heatmap))

    webApp.addDisplay(vis.createGraph('elevation_heatmap', vis.createHeatMapGL(
        "Elevation Data", heightmap,
        heatmap_kwargs={'colorbar': dict(title="Elevation(m)")},
        layout_kwargs=layout_dict)
                                      )
                      )

    webApp.addDisplay(vis.createGraph('soil_content', vis.createHeatMapGL(
        'Sand Content', soilmap,
        heatmap_kwargs={'colorbar': dict(title="Sand Content(%)")},
        layout_kwargs=layout_dict)
                                      )
                      )
    # Run WebApp
    webApp.app.run_server()
