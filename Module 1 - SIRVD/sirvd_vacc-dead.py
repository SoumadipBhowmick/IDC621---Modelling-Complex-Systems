import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap , BoundaryNorm
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['font.size'] = 14
grid_size = 100

SUSCEPTIBLE = 0
VACCINATED = 1
INFECTED_START = 2
PARA = 5  # Peak infectiousness
REFRACTORY = 6
DEAD = 7
d_rate = 0.25

colors = ['white', 'green', 'red', 'blue', 'black']
cmap = ListedColormap(colors)
bounds = [SUSCEPTIBLE,VACCINATED,INFECTED_START, REFRACTORY, DEAD, DEAD + 1 ]
norm = BoundaryNorm(bounds, cmap.N)

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

lent = 100

vaccinated_percentage = np.linspace(0, 100, num=40)

total_dead = []
total_infec = [] 

for per in vaccinated_percentage:
    print(f"Running simulation for {per:.1f}% vaccination...")
    
    grid = np.zeros((grid_size, grid_size))

    num_cells = grid_size * grid_size
    num_vaccinated = int(num_cells * per / 100)

    # Get random positions for vaccinated individuals
    if num_vaccinated > 0:
        vaccinated_positions = np.random.choice(num_cells, num_vaccinated, replace=False)
        vaccinated_rows = vaccinated_positions // grid_size
        vaccinated_cols = vaccinated_positions % grid_size
        grid[vaccinated_rows, vaccinated_cols] = 1

    
    n = grid_size // 2
    grid[n, n] = 2
    grid[n, n+1] = 2
    grid[n, n-1] = 2
    grid[n-1, n] = 2
    grid[n+1, n] = 2

    n_sus = np.zeros(lent)
    n_inf = np.zeros(lent)
    n_dead = np.zeros(lent)
    n_refr = np.zeros(lent)
    n_vac = np.zeros(lent)

    for i in range(lent):
        new_grid = grid.copy()

        
        for j in range(grid_size):
            for k in range(grid_size):
                new_grid[j,k] = sirs(grid, j, k)

        grid = new_grid        
        n_sus[i] = np.sum(grid == SUSCEPTIBLE)
        n_inf[i] = np.sum((grid >= INFECTED_START) & (grid <= PARA))
        n_dead[i] = np.sum(grid == DEAD)
        n_refr[i] = np.sum(grid == REFRACTORY)
        n_vac[i] = np.sum(grid == VACCINATED)

    total_infec.append(n_dead[-1] + n_refr[-1])  # Everyone who got infected = dead + recovered
    total_dead.append(n_dead[-1])
    print(f"  Total Infected: {np.sum(n_inf):.0f}")
    print(f"  Total Dead: {np.sum(n_dead):.0f}")


pal = sns.color_palette("rocket", 2)

fig, ax1 = plt.subplots(figsize=(6, 5))

ax1.minorticks_on()
ax1.tick_params(which='minor', direction='in', length=5,  bottom=True, top=True, left=True,  right=False)
ax1.tick_params(which='major', direction='in', length=10, bottom=True, top=True, left=True,  right=False)
ax1.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False, labelsize=13)

ax1.plot(vaccinated_percentage, np.array(total_infec) / (grid_size**2),
         color=pal[0], linestyle='', marker='^', mfc='w', markersize=8, label='Total Infected')
ax1.set_xlabel('Vaccination Percentage [%]', fontsize=14)
ax1.set_ylabel(r'Total Infected Population', fontsize=14, color=pal[0])
ax1.set_xlim(-5, 105)
ax1.tick_params(axis='y', labelcolor=pal[0])

ax2 = ax1.twinx()

ax2.minorticks_on()
ax2.tick_params(which='minor', direction='in', length=5,  bottom=True, top=True, left=False, right=True)
ax2.tick_params(which='major', direction='in', length=10, bottom=True, top=True, left=False, right=True)
ax2.tick_params(labelbottom=False, labeltop=False, labelleft=False, labelright=True, labelsize=13)

ax2.plot(vaccinated_percentage, np.array(total_dead) / (grid_size**2),
         color=pal[1], linestyle='', marker='o', mfc='w', markersize=8, label='Total Dead')
ax2.set_ylabel(r'Total Dead Population', fontsize=14, color=pal[1])
ax2.tick_params(axis='y', labelcolor=pal[1])

plt.savefig('vaccvsdead.png', dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.show()