import numpy as np
from data_import import data_import_emissions, data_import_emissions_companies, data_import_baseline_metrics
from TransferOptimizer import Tile, get_neighbours, get_avg_emission, get_no_companies, initialize, get_cost, update_theta, predict_Y
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

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
    # Transform the data to the normalized form
    original_data = scaler.inverse_transform(array)
    return original_data

# Import the data
emissions = data_import_emissions() #all emissions from roads and country
emissions_companies = data_import_emissions_companies() #all company rasters in dict (sum of emission + count)
year = 2020
normalize = True
Y = emissions['conc'][year+1]
X1 = emissions['conc'][year]
X2 = emissions_companies['company_emission'][year]
X3 = emissions_companies['company_count'][year]
# normalize the values
if normalize:
    Y = normalize_array(Y)
    X1, X2, X3 = normalize_array(X1), normalize_array(X2), normalize_array(X3) 

# create a grid of tiles
tiles = []
for y in range(len(X1)):
    row = []
    for x in range(len(X1[0])):
        row.append(Tile(X1[y,x], X3[y,x], X2[y,x]))
    tiles.append(row)

# Get neighborhood values
x_vals = []
Y_new = []
for y in range(len(tiles)):
    for x in range(len(tiles[y])):
        em = tiles[y][x].current_emission
        neighbours = get_neighbours(tiles, (y,x), n=1)
        nem = get_avg_emission(neighbours)
        ncomp = get_no_companies(neighbours)
        x_vals.append([em,nem,ncomp])
        Y_new.append(Y[y][x])
x_vals = np.array(x_vals)
x_vals[np.isnan(x_vals)] = 0
Y_new = np.where(np.isnan(Y_new), 0, Y_new)

# Train the model
losses = [] #empty list to store the losses
thetas = [] #empty list to store thetas
n_iter = 3 #number of iterations 
best_theta = np.array([np.inf for _ in range(3)]) #initialize best theta
for _ in range(n_iter): #run the model n_iter times
    l = []
    b,theta=initialize(3) #initialize b and theta 
    print("After initialization -Bias: ",b,"theta: ",theta)
    print(f"calculated loss before: {get_cost(Y_new,predict_Y(b,theta,x_vals))}")
    Y_hat=predict_Y(b,theta,x_vals) #predict the values
    #gradient descent
    for i in range(1000): 
        b,theta=update_theta(x_vals,Y_new,Y_hat,b,theta) #update the values
        loss = get_cost(Y_new,predict_Y(b,theta,x_vals)) #calculate the loss
        l.append(loss) #append the loss to the list
        print("After update -Bias: ",b,"theta: ",theta) #print the values
        print(f"calculated loss after: {loss}") #print the loss
        if loss < l[i-1]:
            best_theta = theta #update the best theta if the loss is lower
    losses.append(l) 
    thetas.append(best_theta)

# convert to final prediction
loss_min = min([min(l) for l in losses])==[min(l) for l in losses] # get the id of the lowest loss
theta_min = [t for t, l in zip(thetas, loss_min) if l][0] # get the theta with the lowest loss
b,theta=initialize(3) # reinitialize b and theta
y_pred = predict_Y(b,theta_min,x_vals) # predict the values with the lowest theta
y_pred.shape = (320, 280) # reshape to the original shape
if normalize:
    y_pred = renormalize_array(y_pred, Y) # renormalize the values

# Plot the losses
for i in range(len(losses)):
    plt.plot(range(len(losses[0])),losses[i],label = 'id %s'%i)
plt.legend()
plt.show()

# Plot the predicted values
from data_import import mean_absolute_error
plt.imshow(y_pred)
plt.title(f'Trained: {year}, Predicted {year+1}, MAE: {round(mean_absolute_error(Y, y_pred), 2)}')
plt.show()
