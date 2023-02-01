import numpy as np
from PIL import Image
from data_import import data_import_emissions_companies

# Interpolate company emissions
dataset = data_import_emissions_companies([1990, 1995, 2000, 2005, 2010, 2015, 2020])
interpolated = []
ds_matrix = [v for k,v in dataset['company_count'].items()]
for y in range(len(ds_matrix[0])):
    row = []
    for x in range(len(ds_matrix[0][0])): #iterate over columns
        row.append(np.interp(range(36), [0,5,10,20,25,30,35], [d[y][x] for d in ds_matrix])) #interpolate values for each pixel over time
    interpolated.append(row) #add row to result
interpolated = np.transpose(interpolated, (2, 0, 1)) #transpose to get correct shape

# Save interpolated emissions
year = 1990
for img in interpolated:
    result = Image.fromarray(img.astype(np.uint8))
    result.save(f"Results/company_count_{year}_1000x1000.tif")
    year +=1

# Visualise
from matplotlib import pyplot as plt
# Create a figure and array of axes objects with two subplots
fig, ax = plt.subplots(nrows=1, ncols=2)
# Plot the first bar in the first subplot (interpolated)
ax[0].bar(range(1990, 2026), [x.mean() for x in interpolated])
ax[0].set_title('Interpolated Emissions')
# Plot the second bar in the second subplot (original)
ax[1].bar(dataset['company_emission'].keys(), [x.mean() for x in dataset['company_emission'].values()])
ax[1].set_title('Original Emissions')
plt.show()