import numpy as np
import math
import code
import random
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from data_import import data_import_single_year, data_import_emissions_companies


class Tile:
    def __init__(self, a=0, b=0, c=0):
        self.current_emission = a
        self.no_companies = b
        self.emission_comp = c

# load the value to predict (next years emission)
Ys = data_import_single_year(2021, 'conc')
row_sums = Ys.sum(axis=1)
Ys = Ys / row_sums[:, np.newaxis]


# load emissions
emissions = data_import_single_year(2020, 'conc')

# normalize the values
row_sums = emissions.sum(axis=1)
emissions = emissions / row_sums[:, np.newaxis]

# load company_data
companies = data_import_emissions_companies([2020])

company_count = companies['company_count'][2020]
company_count[np.isnan(company_count)] = 0
company_emission = companies['company_emission'][2020]
company_emission[np.isnan(company_emission)] = 0
# normalize the values
# row_sums = company_count.sum(axis=1)
# company_count = company_count / row_sums[:, np.newaxis]

print(company_count.min())

row_sums = company_emission.sum(axis=1)
company_emission = company_emission / row_sums[:, np.newaxis]


tiles = []
for y in range(len(emissions)):
    row = []
    for x in range(len(emissions[0])):
        row.append(Tile(emissions[y,x], company_count[y,x], company_emission[y,x]))
    tiles.append(row)


def get_neighbours(coord, n=1):
    neighbours = []
    for i in range(coord[0] - n, coord[0] + n + 1):
        if 0 <= i < len(tiles):
            for j in range(coord[1] - n, coord[1] + n + 1):
                if 0 <= j < len(tiles[0]):
                    if i != coord[0] or j != coord[1]:
                        neighbours.append(tiles[i][j])
    return neighbours


def get_avg_emission(neighbours):
    e = 0
    for nei in neighbours:
        e += nei.current_emission

    return e/len(neighbours)

def get_no_companies(neighbours):
    e = 0
    for nei in neighbours:
        e += nei.no_companies

    return e


x_vals = []
Y = []
for y in range(len(tiles)):
    for x in range(len(tiles[y])):
        em = tiles[y][x].current_emission
        neighbours = get_neighbours((y,x), n=1)
        nem = get_avg_emission(neighbours)
        ncomp = get_no_companies(neighbours)
        x_vals.append([em,nem,ncomp])
        Y.append(Ys[y][x])

x_vals = np.array(x_vals)
x_vals[np.isnan(x_vals)] = 0

Y = np.where(np.isnan(Y), 0, Y)


def initialize(dim):
    b=random.random()
    theta=np.random.rand(dim)
    return b,theta

def predict_Y(b,theta,X):
    return b + np.dot(X,theta)

def get_cost(Y,Y_hat):
    Y_resd=Y-Y_hat
    return np.sum(np.dot(Y_resd.T,Y_resd))/len(Y-Y_resd)

def update_theta(x,y,y_hat,b_0,theta_o,learning_rate):
    db=(np.sum(y_hat-y)*2)/len(y)
    dw=(np.dot((y_hat-y),x)*2)/len(y)
    b_1=b_0-learning_rate*db
    theta_1=theta_o-learning_rate*dw
    return b_1,theta_1

b,theta=initialize(3)
print("After initialization -Bias: ",b,"theta: ",theta)
print(f"calculated loss before: {get_cost(Y,predict_Y(b,theta,x_vals))}")
Y_hat=predict_Y(b,theta,x_vals)

# losses = []
# for _ in range(1000):
#     b,theta=update_theta(x_vals,Y,Y_hat,b,theta,0.01)
#     loss = get_cost(Y,predict_Y(b,theta,x_vals))
#     losses.append(loss)
#     print("After update -Bias: ",b,"theta: ",theta)
#     print(f"calculated loss after: {loss}")


plt.plot([1,2,3,4])