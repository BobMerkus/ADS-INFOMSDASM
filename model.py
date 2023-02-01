import numpy as np
from data_import import data_import_emissions, data_import_emissions_companies, data_import_baseline_metrics, mean_absolute_error
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
    # Transform the data to the renormalized form
    original_data = scaler.inverse_transform(array)
    return original_data

# Import the data
emissions = data_import_emissions() #all emissions from roads and country
emissions_companies = data_import_emissions_companies() #all company rasters in dict (sum of emission + count)
year = 2020 #year to train on
normalize = True #normalize the values
n_iter = 5 #number of iterations 
n_neighbors = 1 #number of neighbors to use
Y = emissions['conc'][year+1]
X1 = emissions['conc'][year]
X2 = emissions_companies['company_emission'][year]
X3 = emissions_companies['company_count'][year]
# normalize the values
if normalize:
    Ys = normalize_array(Y)
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
        neighbours = get_neighbours(tiles, (y,x), n=n_neighbors)
        nem = get_avg_emission(neighbours)
        ncomp = get_no_companies(neighbours)
        x_vals.append([em,nem,ncomp])
        Y_new.append(Ys[y][x])
x_vals = np.array(x_vals)
x_vals[np.isnan(x_vals)] = 0
Y_new = np.where(np.isnan(Y_new), 0, Y_new)

# Train the model
losses = [] #empty list to store the losses
thetas = [] #empty list to store thetas
errors = [] #empty list to store the errors
bias = []
best_theta = np.array([np.inf for _ in range(3)]) #initialize best theta
best_b = np.array([np.inf for _ in range(3)]) #initialize best theta
for _ in range(n_iter): #run the model n_iter times
    l = []
    e = []
    b,theta=initialize(3) #initialize b and theta 
    print("After initialization -Bias: ",b,"theta: ",theta)
    print(f"calculated loss before: {get_cost(Y_new,predict_Y(b,theta,x_vals))}")
    Y_hat=predict_Y(b,theta,x_vals) #predict the values
    #gradient descent
    for i in range(1000): 
        b,theta=update_theta(x_vals,Y_new,Y_hat,b,theta) #update the values
        pred = predict_Y(b,theta,x_vals) #predict the values
        loss = get_cost(Y_new,pred) #calculate the loss
        mae = mean_absolute_error(Y_new,pred) #calculate the mean absolute error
        l.append(loss) #append the loss to the list
        e.append(mae) #append the mae to the list
        print("After update -Bias: ",b,"theta: ",theta) #print the values
        print(f"calculated loss after: {loss}") #print the loss
        if loss < l[i-1]:
            best_b, best_theta = b, theta #update the best theta if the loss is lower
    losses.append(l)
    errors.append(e) 
    thetas.append(best_theta)
    bias.append(best_b)

# convert to final prediction
loss_min = min([min(l) for l in losses])==[min(l) for l in losses] # get the id of the lowest loss
bias_min, theta_min = [(b, t)for b, t, l in zip(bias, thetas, loss_min) if l][0] # get the theta with the lowest loss
#b,theta=initialize(3) # reinitialize b and theta
y_pred = predict_Y(bias_min,theta_min,x_vals) # predict the values with the lowest theta
y_pred.shape = (320, 280) # reshape to the original shape
if normalize:
    y_pred = renormalize_array(y_pred, Y) # renormalize the values

# Plot the losses
for i in range(len(losses)):
    plt.plot(range(len(losses[0])),losses[i],label = 'id %s'%i)
plt.legend()
plt.title("Loss")
plt.show()

# Plot the errors
for i in range(len(errors)):
    plt.plot(range(len(errors[0])),errors[i],label = 'id %s'%i)
plt.legend()
plt.title("Mean Absolute Error (Normalized))")
plt.show()

# Plot the predicted values
from data_import import mean_absolute_error
plt.imshow(y_pred)
plt.suptitle(f'Trained: {year}, Predicted: {year+1}')
plt.title(f'Neighbors: {n_neighbors}, Iterations: {n_iter}, Scaling: {normalize}, MAE: {round(mean_absolute_error(Y, y_pred), 2)}',
          fontdict={'fontsize': 10})
plt.savefig(f'./results/predicted_{year}_{n_iter}_{normalize}.png')
plt.show()

