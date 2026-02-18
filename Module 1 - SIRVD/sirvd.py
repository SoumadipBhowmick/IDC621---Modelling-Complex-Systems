import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap , BoundaryNorm

grid_size = 100
N=grid_size
grid = np.zeros((grid_size, grid_size))

vaccination_percentage = 10 # Randomly vaccinate x% of the population

num_cells = grid_size * grid_size
num_vaccinated = int(num_cells * vaccination_percentage / 100)

# Get random positions for vaccinated individuals
vaccinated_positions = np.random.choice(num_cells, num_vaccinated, replace=False)
vaccinated_rows = vaccinated_positions // grid_size
vaccinated_cols = vaccinated_positions % grid_size
grid[vaccinated_rows, vaccinated_cols] = 1

n= grid_size // 2
grid[n][n] = 2
grid[n,n+1] = 2
grid[n,n-1] = 2
grid[n-1,n] = 2
grid[n+1,n] = 2

SUSCEPTIBLE = 0
VACCINATED = 1
INFECTED_START = 2
PARA = 5  # Peak infectiousness
REFRACTORY = 6
DEAD = 28
d_rate = 0.05

colors = ['white', 'green', 'red', 'blue', 'black']
cmap = ListedColormap(colors)
bounds = [SUSCEPTIBLE,VACCINATED,INFECTED_START, REFRACTORY, DEAD, DEAD + 1 ]
norm = BoundaryNorm(bounds, cmap.N)
#plt.imshow(grid, cmap = cmap, norm=norm)

def is_infected(site):
    " checks if the patient is infected (2-5) "
    return INFECTED_START <= site <= PARA

def sirvd(current_grid,i,j):

    infected_count = is_infected(current_grid[i,(j+1) % N]) + is_infected(current_grid[i,(j-1) % N]) + is_infected(current_grid[(i+1) % N,j]) + is_infected(current_grid[(i-1) % N,j]) 
    infection_prob = infected_count/4

    #susceptible
    if current_grid[i,j] == SUSCEPTIBLE:
        if np.random.random() < infection_prob:
            return INFECTED_START
        else:
            return SUSCEPTIBLE

    #vaccinated
    elif current_grid[i,j] == VACCINATED:
        return VACCINATED
    
    #infection
    elif  INFECTED_START <= current_grid[i,j] <= PARA - 1:
        return current_grid[i,j] + 1
    
    elif current_grid[i,j] == PARA:
        if np.random.random() < d_rate:
            return DEAD
        else:
            return REFRACTORY

    elif REFRACTORY <= current_grid[i,j] <= REFRACTORY + 20 :
        return current_grid[i,j] + 1

    elif current_grid[i,j] == REFRACTORY + 21:
        return SUSCEPTIBLE
    
    # Dead
    elif current_grid[i,j] == DEAD:
        return DEAD

    return current_grid[i,j]
    
    

plt.ion()
fig, ax = plt.subplots()
im = ax.imshow(grid, cmap=cmap, norm=norm)
cbar = plt.colorbar(im, ax=ax, shrink=0.7,ticks=[0.5, 1.5, 3.5, REFRACTORY+10, DEAD + 0.5])
cbar.ax.set_yticklabels(['Susceptible', 'Vaccinated', 'Infected', 'Refractory', 'Dead'])
plt.tight_layout()

lent=200
n_sus = np.zeros(lent)
n_inf = np.zeros(lent)
n_dead = np.zeros(lent)
n_refr = np.zeros(lent)
n_vac = np.zeros(lent)

for i in range(lent):
    plt.title(f'Timestep:{i}')

    new_grid = grid.copy()

    for j in range(grid_size):
        for k in range(grid_size):

            new_grid[j,k] = sirvd(grid,j,k)

    grid = new_grid        
    n_sus[i] = np.sum(grid==SUSCEPTIBLE)
    n_inf[i] = np.sum((grid >= INFECTED_START) & (grid<PARA))
    n_dead[i] = np.sum(grid == DEAD)
    n_refr[i] = np.sum((grid>=PARA) & (grid <= REFRACTORY))
    n_vac[i] = np.sum(grid==VACCINATED)

    im.set_data(grid)
    fig.canvas.draw_idle()
    plt.pause(0.0001)

plt.ioff()

fig, ax = plt.subplots(figsize=(6, 5))

ax.minorticks_on()
ax.tick_params(direction='in', top=True, right=True)
ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
ax.tick_params(labelsize=13)
ax.tick_params(which='minor', direction='in', length=5, bottom=True, top=True, left=True, right=True)
ax.tick_params(which='major', direction='in', length=10, bottom=True, top=True, left=True, right=True)


ax.plot(n_sus/(grid_size**2), label='Susceptible', color='gray')
ax.plot(n_inf/(grid_size**2), label='Infected', color='red', linewidth=2)
#ax.plot(n_vac, label='Vaccinated', color='green')
ax.plot(n_dead/(grid_size**2), label='Dead', color='black')
ax.plot(n_refr/(grid_size**2), label='Refractory', color='blue')
ax.set_xlabel('Time Step')
ax.set_ylabel('Population Count')
ax.set_title('SIRVD Model Population Dynamics')
ax.legend()
plt.tight_layout()
plt.show()