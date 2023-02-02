import numpy as np
from PIL import Image
from data_import import data_import_emissions_companies

# Interpolate company emissions
data = 'company_emission'
dataset = data_import_emissions_companies([1990, 1995, 2000, 2005, 2010, 2015, 2020])
interpolated = []
ds_matrix = [v for k,v in dataset[data].items()]
for y in range(len(ds_matrix[0])):
    row = []
    for x in range(len(ds_matrix[0][0])): #iterate over columns
        row.append(np.interp(range(36), [0,5,10,20,25,30,35], [d[y][x] for d in ds_matrix])) #interpolate values for each pixel over time
    interpolated.append(row) #add row to result
interpolated = np.transpose(interpolated, (2, 0, 1)) #transpose to get correct shape

# # Save interpolated emissions
year = 1990
for img in interpolated:
    result = Image.fromarray(img.astype(np.uint8))
    result.save(f"Results/interpolated_{data}_{year}.tif")
    year +=1

dataset['company_emission'][2020].max()
interpolated[10].max()
# Visualise
from matplotlib import pyplot as plt
# Create a figure and array of axes objects with two subplots
fig, ax = plt.subplots(nrows=1, ncols=2)
# Plot the first bar in the first subplot (interpolated)
ax[0].bar(range(1990, 2026), [x.mean() for x in interpolated])
ax[0].set_title(f'Interpolated {data}')
# Plot the second bar in the second subplot (original)
ax[1].bar(dataset[data].keys(), [x.mean() for x in dataset[data].values()])
ax[1].set_title(f'Original {data}')
plt.show()