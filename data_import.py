# Setup
import numpy as np
# import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation
from sklearn.preprocessing import MinMaxScaler

"""_summary_

Betreft Kaart conc_no2_2020 en rwc_no2_2020 
Omschrijving: Jaargemiddelde grootschalige NO2-concentratie en lokale bijdragen van rijkswegen (rwc) in 2020 in Nederland   
Bron: Rijksinstituut voor Volksgezondheid en Milieu   
Datum fact sheet: 12 maart 2021 
A.  Indicator     
Jaar 2020   
Scenario feitelijke omstandigheden   
Component stikstofdioxide (NO2)   
Kengetal jaargemiddelde van 24-uurswaarden   
Eenheid μg/m³   
Nauwkeurigheid σ = groter dan 15% (groter dan vorige jaren door extra onzekerheid in emissies vanwege COVID maatregelen. 
Periode kalenderjaar

Returns:
    _type_: _description_
"""

# Import data from ascii grid, save as png and return as numpy array
def data_import_single_year(year, dataset, export = False) -> np.array:
    ascii_grid = np.loadtxt(f'Data/conc_no2_{year}/{dataset}_no2_{year}.asc', skiprows=6)
    ascii_grid[ascii_grid==-999] = 0
    ascii_grid[ascii_grid==-9999] = 0
    if export:
        plt.imshow(ascii_grid)
        # Create image
        plt.savefig(f"Results/Images/no2_{dataset}_{year}.png", dpi=300)
        plt.title(f"NO2 - {dataset} - {year}")
        plt.close('all')
    return ascii_grid

# Create animation from list of arrays and save as gif
def animation(data, dataset, years, plot_speed_ms = 1000) -> None:
    # Create a figure and axes for the plot
    fig, ax = plt.subplots()
    # Plot the data
    im = ax.imshow(data[0])
    # Function to update the plot for each frame of the animation
    def update(i):
        ax.set_title(label = f"NO2 - {dataset} - {i + min(years)}") #title
        im.set_data(data[i]) #update data for each frame
        return im,
    # Create the animation using FuncAnimation for n frames based on data length
    ani = FuncAnimation(fig, update, frames=range(len(data)), blit=True, interval = plot_speed_ms)
    # Save the animation as a gif
    ani.save(f"Results/Images/no2_{dataset}_animation_{plot_speed_ms}.gif")
    return ani

# Iterate over datasets (type + years) and create a list of arrays for country wide emissions
def data_import_emissions(years = range(2011,2022), datasets = ['rwc', 'conc'], plot_speed_ms = 500, export = False) -> dict:
    #rwc -> emissions from roads
    #conc -> emmissions whole country
    result = dict()
    for dataset in datasets:
        data = []
        for year in years:
            data_single_year = data_import_single_year(year, dataset, export = export)
            data.append(data_single_year) #create list of arrays from raw data import
        if export:
            animation(data, dataset, years, plot_speed_ms) #animate the emission data over time
        data = {year : data for year, data in zip(years, data)} #create dict of arrays
        result[dataset] = data #add to result dict
    result['meta'] = np.loadtxt(f'Data/conc_no2_2021/conc_no2_2021.asc', max_rows=6, dtype=str) #add latest metadata
    return result

# Iterate over datasets (type + years) and create a list of arrays for company emissions
def data_import_emissions_companies(years = [1990, 1995, 2000, 2005, 2010, 2015, 2019, 2020]) -> dict:
    result = {
        'company_emission' : {},
        'company_count': {}
    }
    for year in years:
        for dataset in ['emission', 'count']:
            photo = Image.open(f"Results/company_{dataset}_{year}_1000x1000.tif")
            result[f'company_{dataset}'][year] = np.array(photo)
    return result

# MAE 
def mean_absolute_error(y_true, y_pred) -> float:
    return np.mean(np.abs(y_true - y_pred))

# calculate baseline model scores
def data_import_baseline_metrics(start_year = 2011):
    data = data_import_emissions()
    df = pd.DataFrame()
    for year in range(start_year,2021):
        y_true = data['conc'].get(year+1)
        # simple forecast methods
        naive = data['conc'].get(year) #naive (last data point)
        sliding_window = list(data['conc'].values())[0:(year-start_year+1)] # sliding window average
        average = sum(data['conc'].values()) / len(data['conc'])
        baseline_metrics = {'year' : year+1,'naive' : mean_absolute_error(y_true, naive), 'average' : mean_absolute_error(y_true, average), 'metric' : 'MAE'}
        df_temp = pd.DataFrame([baseline_metrics])
        df = pd.concat([df, df_temp], ignore_index=True)
    return df

# retrieve the coordinates of the neighbour cells
def get_neighbours(array, coord, n=1) -> np.array:
    y, x = coord #get coordinates
    y_lim, x_lim = array.shape #get array dimensions
    neighbours = []
    neighbour_coords = []
    # iterate over the neighbour cells
    for i in range(y-n, y+n+1):
        for j in range(x-n, x+n+1):
            if i >= 0 and i < y_lim and j >= 0 and j < x_lim: #check if the neighbour cell is within the array
                neighbour_coords.append((i, j)) #add the coordinates of the neighbour cell to the list
                neighbours.append(array[i][j]) #add the value of the neighbour cell to the list
            else:
                neighbours.append(0) #if the neighbour cell is outside the array, add 0 to the list
    dim = (n*2) + 1 #calculate the dimension of the neighbour array
    neighbours = np.array(neighbours).reshape(dim, dim) #reshape the neighbour array (e.g. n=1 -> 3x3)
    return neighbours

# Convert array Y and list of X variables to keras friendly array
def to_keras_array(X: list[np.array], Y = np.array, n_neighbors=1) -> tuple:
    X_arrays = []
    Y_array = []
    for i in range(Y.shape[0]): #iterate over the rows y
        for j in range(Y.shape[1]): #iterate over the columns x
            X_arrays.append(np.array([get_neighbours(x, coord=(i,j), n=n_neighbors) for x in X])) #get the neighbours of the cell and add them to the array
            Y_array.append(Y[i][j]) #add the outcome value of the cell to the array
    # convert to numpy arrays
    X = np.array(X_arrays)
    Y = np.array(Y_array)
    return Y, X

# Import data for a single year 
def data_import(year, n_neighbors, normalize = True, pre_process_method = "interpolated") -> tuple:
    emissions = data_import_emissions() #all emissions from roads and country
    print(f"Importing data for {year} with {n_neighbors} neighbors, this can take a while...")
    # DATA IMPORT FOR TRAINING YEAR
    X1 = emissions['conc'][year] #this year (x)
    X2 = np.array(Image.open(f"Results/{pre_process_method}_company_emission_{year}.tif")) #emissions_companies['company_emission'][year]
    X3 = np.array(Image.open(f"Results/{pre_process_method}_company_count_{year}.tif")) #emissions_companies['company_count'][year]
    if year < 2021:
        Y = emissions['conc'][year+1] #next year (y)
    else:
        print("No data available beyond 2021, returning data for 2021 instead")
        Y = X1 #used for scaling purposes
    # normalize the values
    if normalize:
        X1, X2, X3 = normalize_array(X1), normalize_array(X2), normalize_array(X3) 
        Ys = normalize_array(Y)
    # Convert the arrays to mulitdimensional arrays
    Ys, Xs = to_keras_array(Y = Ys, X = [X1, X2, X3], n_neighbors=n_neighbors)
    return Y, Ys, Xs

# Min max scaler
def normalize_array(array):
    scaler = MinMaxScaler()
    scaler = scaler.fit(array)
    # Transform the data to the normalized form
    normalized_data = scaler.transform(array)
    return normalized_data

# Reverse min max scaler
def renormalize_array(array, original_array):
    scaler = MinMaxScaler()
    scaler = scaler.fit(original_array)
    # Transform the data to the renormalized form
    original_data = scaler.inverse_transform(array)
    return original_data

if __name__=="__main__":

    # Set the year range and animation speed for .gif
    emissions = data_import_emissions(export = True) #all emissions from roads and country
    emissions_companies = data_import_emissions_companies() #all company rasters in dict (sum of emission + count)
    dataset = {**emissions, **emissions_companies} #combine dicts
    dataset.keys() #rasters + metadata
    meta = dataset.pop('meta') #remove metadata from dict
    print(meta) #280x320 km -> 1x1 km grids
    
    # Left over is clean dataset for modelling. 
    # NOTE: We only have data for 2011-2021 for emissions
    # NOTE: We only have data for 1990, 1995, 2000, 2005, 2010, 2015, 2019, 2020 for company emissions
    # NOTE: Do we impute the missing data? Or we use 2015 + 2020 with 5 year interval?
    dataset['rwc'][2020]
    dataset['conc'][2020]
    print(dataset['company_emission'][2020])
    dataset['company_count'][2020]
    
    #read the .csv file into a pandas df
    # df = pd.read_csv("Data/ERCompanyLevel.csv", header=0, sep=";") #company agents
    # summed = df[['Jaar','Emissie', 'Bedrijf']].groupby(['Jaar', 'Bedrijf']).sum('Emissie').reset_index()
    # df[['Xcoord', 'Ycoord', 'Bedrijf']]    
    

    Y, Ys, Xs = data_import(2011, n_neighbors=1) # Import the data
    Y, Ys, Xs = data_import(2021, n_neighbors=1) # Import the data
    
    Y.min(), Y.max(), Ys.min(), Ys.max(), Xs.min(), Xs.max() # Check the min and max values for scaling if you want