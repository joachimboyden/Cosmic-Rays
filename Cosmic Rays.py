import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import matplotlib.pylab as plb

plb.style.use("seaborn")
mpl.style.use('dark_background')

#-------------------------------------------------------------------------------------------
#  COSMIC RAY SIMULATION
#-------------------------------------------------------------------------------------------

#compares a simulated distribution of cosmic rays to the theoretical model 

simulation_range = 10000

def uni_dist(start, end):
    return random.uniform(start, end)

# Coin flip for 1 or -1 charge on ion
def charge_chance():
    a = random.random()
    if a <= 0.5:
        return 1
    elif a > 0.5:
        return -1

def log_norm_dist(mu, sigma):
    return [random.lognormal(mu, sigma) for i in range(simulation_range)]

def rand_cos_sq():
    while True:
        x = np.pi*random.random() - np.pi/2
        y = random.random()
        if y < (np.cos(x))**2:
            return x
        else:
            continue
        
def cos_sq_dist():
    a = [rand_cos_sq() for i in range(simulation_range)]
    return a

def plot_zenith_dist():
    hist, bins, patches = plt.hist(cos_sq_dist(), bins=50, density=True, 
                                   label="Zenith Angle Distribution")
    bin_centres = (bins[1:] + bins[:-1])/2
    plt.plot(bin_centres, 2*((np.cos(bin_centres))**2)/np.pi, label=r'$cos(\theta)**2$')
    
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$P(\theta)$')
    plt.legend()
    plt.show()
    
def plot_dist(dist):
    hist, bins, patches = plt.hist(dist, bins=100, density=True, 
                                   label="Distribution")
    
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$P(\theta)$')
    plt.legend()
    plt.show()
    
class Ray():
    
    def __init__(self):
        self.charge = charge_chance()
        self.energy = random.lognormal(6.55, 1.8)
        self.zenith = rand_cos_sq()
        self.azimuth = random.random()*2*np.pi
        self.x = random.random()*0.5
        self.y = random.random()*0.5
        self.z = 0
        
plot_zenith_dist()
        
#----------------------------------------------------------------------------------------
#  DRIFT CHAMBER SIMULATION
#----------------------------------------------------------------------------------------

# simulates the motion of a cosmic ray with a random zenith angle, described by
# the distribution above, in a drift chamber.

def find_near(array, value):
    index = (np.abs(array - value)).argmin()
    return index      
        
class Chamber():
    
    def __init__(self, pixel_size):
        self.height = 0.3
        self.length = 0.5
        self.width = 0.5
        self.pixel_size = pixel_size
        self.pixel_num_x = int((self.length*100)/(pixel_size*100)) + 1
        self.pixel_num_y = int(self.width/pixel_size) + 1
        self.pixel_num_z = int((self.height*100)/(pixel_size*100)) + 1
        
        self.grid = np.zeros((self.pixel_num_z, self.pixel_num_x))
        self.x_dist_list = pixel_size*np.arange(self.pixel_num_x)
        self.z_dist_list = pixel_size*np.arange(self.pixel_num_z)
        self.ionisation = 94
        
        self.ray = Ray()
        
    def init_ion(self):
        for i in range(self.pixel_num_z):
            x_coor = self.ray_func_x(self.z_dist_list[i])
            if 0 <= x_coor <= self.length:
                x_ind = find_near(self.x_dist_list, x_coor) 
                self.grid[i][x_ind] = 1
        
        for i in range(self.pixel_num_x):
            z_coor = self.ray_func_z(self.x_dist_list[i])
            if 0 <= z_coor <= self.height:
                z_ind = find_near(self.z_dist_list, z_coor)
                self.grid[z_ind][i] = 1
            
        
    def ray_func_z(self, x):
        z = (x - self.ray.x)/np.tan(self.ray.zenith)
        return z
    
    def ray_func_x(self, z):
        x = z*np.tan(self.ray.zenith) + self.ray.x
        return x
    
    def plot_grid(self):
        fig,(ax1,ax2) = plt.subplots(2)
        sns.heatmap( self.grid, ax=ax1, annot=False, square=True)
        
        x = np.linspace(0, self.length, 100)
        y = -1*self.ray_func_z(x)
        ax2.set_xticks(self.x_dist_list)
        ax2.set_yticks(-1*self.z_dist_list)
        plt.xlim([0,self.width])
        plt.ylim([-1*self.height,0])
        ax2.set_aspect('equal')
        ax2.plot(x,y)
        
        
chamber = Chamber(0.02)
chamber.init_ion()
chamber.plot_grid()


        
        
        
        
    


    


    
