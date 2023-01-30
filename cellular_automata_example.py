# Set up the grid with a random initial state
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

class CellularAutomata():
    
    def _random_grid(self):
        self.grid = np.random.randint(2, size=self.grid_size)      
    
    def generate_grid(self):  
        self.evolution = 0 
        self.counter+=1
        if self.method=="random":
            self._random_grid()
        else:
            raise ValueError("No supported method found. Supported methods are: ('random')")
    
    def __init__(self, x=10, y=10, method = "random") -> None:
        self.iteration = 0 
        self.counter = 0
        self.evolution = 0 
        self.x = x
        self.y = y
        self.method = method
        self.grid_size = self.x*self.y
        self.generate_grid()
    
    # Define the rules for updating the cells
    def update_cell(self, cell, neighbors):
        num_neighbors = sum(neighbors) # Count the number of living neighbors
        # Apply the rules of the game of Life
        if cell == 1:
            if num_neighbors < 2 or num_neighbors > 3:
                return 0
            else:
                return 1
        else:
            if num_neighbors == 3:
                return 1
            else:
                return 0
    
    # Compute the new state of new evolution grid
    def iter(self):
        new_grid = np.zeros(self.grid_size)
        for j in range(1, self.grid_size-1):
            # Get the current cell and its neighbors
            cell = self.grid[j]
            neighbors = [self.grid[j-1], self.grid[j+1]]
            new_grid[j] = self.update_cell(cell, neighbors) # Update the cell based on the rules
        return new_grid
        
    def animate(self, i):       
        new_grid = self.iter() # Compute the new state of new evolution grid
        is_empty = all(self.grid==0) # Check if the grid has reached a emptry state
        is_stable = np.array_equal(self.grid, new_grid) # Check if the grid has reached a stable state
        self.grid[:] = new_grid # Update the grid with the new state
        self.im.set_data(self.grid.reshape((self.x, self.y))) # Update the image data
        #new grid needed
        if is_empty or is_stable:
            print(f"{self.counter}:Empty/stable grid found at evolution {self.evolution} & iteration {self.iteration}")
            self.generate_grid()
        # Update evolution
        self.iteration+=1
        self.evolution+=1
        return [self.im]
    
    def run(self, frames=100, interval=100):
        self.fig, self.ax = plt.subplots()
        self.im = self.ax.imshow(self.grid.reshape((self.x, self.y)), cmap="binary")
        # Run the animation
        anim = animation.FuncAnimation(self.fig, self.animate, frames=frames, interval=interval)
        plt.show()

test_automata = CellularAutomata(x = 100, y = 100)
test_automata.run(frames = 100, interval = 100)
