import numpy as np
from PIL import Image
from data_import import data_import_emissions_companies

dataset = data_import_emissions_companies([1990, 1995, 2000, 2005, 2010, 2015, 2020])

interpolated = []
ds_matrix = [v for k,v in dataset['company_emission'].items()]
for y in range(len(ds_matrix[0])):
    row = []
    for x in range(len(ds_matrix[0][0])):
        row.append(np.interp(range(36), [0,5,10,20,25,30,35], [d[y][x] for d in ds_matrix]))
    interpolated.append(row)

interpolated = np.transpose(interpolated, (2, 0, 1))

year = 1990
for img in interpolated:
    result = Image.fromarray(img.astype(np.uint8))
    result.save(f"Results/company_emission_{year}_1000x1000.tif")
    year +=1
