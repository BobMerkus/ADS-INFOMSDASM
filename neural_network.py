import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from data_import import mean_absolute_error, data_import, renormalize_array

    # MODEL DEFINITION
# TRANFORMER NEURAL NETWORK
def model_transformer(input_shape):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Permute((2, 3, 1))(inputs)
    x = tf.keras.layers.Reshape((49, 3))(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
# ARTIFICIAL NEURAL NETWORK
def model_ANN(input_shape):
    model = tf.keras.models.Sequential() # Sequential model
    model.add(tf.keras.layers.Flatten(input_shape=input_shape))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error') # Compile the model
    return model

# Runtime
if __name__=='__main__':
    # Hyperparameters
    train_years = [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
    test_year = 2020
    normalize = True #normalize the values
    n_neighbors = 3 #number of neighbors to use
    window_size = n_neighbors*2+1
    input_shape = (3, window_size, window_size) #input shape for the model (number of features, x, y)
    epochs=50
    batch_size=32
    
    pre_process_method = "interpolated"
    file_name = f'./Results/ANN_{max(train_years)}_{pre_process_method}_{n_neighbors}'
    image_name = f'./Results/Images/ANN_{max(train_years)}_{pre_process_method}_{n_neighbors}'
    #emissions_companies = data_import_emissions_companies() #all company rasters in dict (sum of emission + count)
    #year = train_years[0]

    # Model definition
    #model = model_transformer(input_shape) # Transformer model (Not tested yet)
    model = model_ANN(input_shape) # Artificial Neural Network
    model.summary()
    # Train the model for each year
    histories = [] # Initialize a list to store the training histories
    MAE = [] # Initialize a list to store the MAE
    # MODEL TRAINING
    for year in train_years:
        # Import the data
        Y, Ys, Xs = data_import(year = year, n_neighbors=n_neighbors, normalize=normalize) # scaled data (Ys) and unscaled data (Y)
        print("Data imported with size:")
        print(f'Y: {Ys.shape}, X: {Xs.shape}') #scaled data (Ys) and unscaled data (Y)
        history = model.fit(x = Xs, y = Ys, epochs=epochs, batch_size=batch_size) # Fit the model on the training data
        histories.append(history) # Add the history to the list of histories
        Y_pred = model.predict(Xs).flatten().reshape(Y.shape)
        if normalize:
            Y_pred = renormalize_array(Y_pred, Y)
        mae = mean_absolute_error(Y,Y_pred)
        MAE.append(mae) #calculate the mean absolute error
        # Plot the predicted values
        plt.imshow(Y_pred)
        plt.suptitle(f'Trained: {min(train_years)}-{year}, Predicted: {year+1}')
        plt.title(f'Neighbors: {n_neighbors}, MAE: {round(mae, 2)}kg NO2',
                fontdict={'fontsize': 10})
        plt.savefig(f'{image_name}_{year}.png', dpi=300)
        plt.close('all') # Close the plot
        
    # Save the model to disk
    model.save(f'{file_name}.h5')
    # Concatenate the histories into a single history object
    final_history = histories[0]
    for i in range(1, len(histories)):
        final_history.history['loss'] += histories[i].history['loss']
    # Divide the total loss by the number of fit calls to get the average loss
    final_history.history['loss'] = [x / len(histories) for x in final_history.history['loss']]
    average_loss = round(np.mean(final_history.history['loss']),6)
    average_mae = round(np.mean(MAE), 3)
    fig, ax = plt.subplots(2)
    fig.suptitle(f'Training Loss + Mean Absolute Error (MAE) for Artificial Neural Network')
    # Plot the training loss
    ax[0].plot(final_history.history['loss'])
    ax[0].set_title(f'Training Loss: {average_loss}, Neighbors: {n_neighbors}', fontdict={'fontsize': 10})
    ax[0].set_xlabel('Epoch', fontdict={'fontsize': 8})
    ax[0].set_ylabel('Loss', fontdict={'fontsize': 8})
    # Plot the MAE
    ax[1].bar(train_years, MAE)
    ax[1].set_title(f'MAE: {average_mae} kg NO2', fontdict={'fontsize': 10})
    ax[1].axhline(y=np.mean(MAE), color = 'r')
    ax[1].set_ylim(0, 2)
    
    fig.tight_layout()
    fig.savefig(f'{image_name}_train.png', dpi=1000)
    fig.show()
    plt.close('all') # Close the plot

    # NOTE:: Bias because of interpolation with time series, we should actually use lookforward with missing company data 
    # NOTE:: 10 neighbors (3, 21, 21) = 1.323 related cells -> 179.841 parameters in ANN to estimate
    # NOTE:: (89.600, 3, 21, 21) training dimension -> 320 x 260 * 3 * 21 * 21 = 118.540.800 data points per year
    # NOTE:: 10 years to train on: 1.185.408.000 data points
    # NOTE:: Warning after n_neighbors > 10 ?
    
    # Predict the values for the test years
    model = tf.keras.models.load_model(f'{file_name}.h5') # Model that is pre-trained
    model.summary() # Print the model summary
    Y, Ys, Xs = data_import(test_year, n_neighbors=n_neighbors) # Import the data
    
    Y_pred = model.predict(Xs).flatten().reshape(Y.shape) # Predict the values for next year
    if normalize:
        Y_pred = renormalize_array(Y_pred, Y) # Renormalize the values if they were normalized
    if test_year>=2021:
        mae_display = 'N/A' # If the model was tained 2021 or later, we cannot calculate the MAE
    else:
        mae_display = round(mean_absolute_error(Y, Y_pred), 2) # Calculate the MAE 
    plt.imshow(Y_pred)
    plt.suptitle(f'Artificial Neural Network {min(train_years)}-{max(train_years)}, Predicted: {test_year+1}') 
    plt.title(f'Neighbors: {n_neighbors}, MAE: {mae_display} kg NO2',
            fontdict={'fontsize': 10})
    plt.savefig(f'{image_name}_test_{test_year+1}.png', dpi=300)
    plt.show()
    plt.close('all') # Close the plot
    
    import pandas as pd
    #Y_pred.tofile(f'{file_name}_{test_year+1}_prediction.bin') # Save the predicted values to a binary file
    df = pd.DataFrame(Y_pred)
    # save the dataframe as a csv file
    df.to_csv(f'{file_name}_{test_year+1}_prediction.csv')