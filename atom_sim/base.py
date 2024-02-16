import numpy as np
import matplotlib.pyplot as plt
from arc import *
from scipy.constants import k, hbar
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from functools import partial
from tqdm import tqdm
from Functions import polylog 
from scipy.integrate import trapezoid
# from mpmath import polylog, re,nstr

class Atom_sim(object):
    '''
    Creates a 6D data point for N atoms
    [x,y,z,px,py,pz]
    atom: ARC atom object
    deg: boson:0 , fermion: 1
    center_density: peak number density (units: 10^18 m^-3)
    sigma: std position of atoms
    temp: temperature of the atoms
    '''

    def __init__(self, atom, center_density=1, sigma=5, temp = 500E-9, omega=1000, deg=0,samples=1000):

        self.atom = atom
        self.deg = deg
        self.center_density=center_density
        self.sigma = sigma
        self.temp = temp
        self.omega=omega
        self.de_Broglie = np.sqrt((2*np.pi*hbar**2)/(self.atom.mass*k*self.temp))
        self.samples = np.linspace(0,2*sigma*1e-6,samples)
        self.density_distribution()
        self.build_density()

        
    # def build_density(self):
    #     '''
    #     Generates the Data points depending on the distribution

    #     returns: [6D array] slice at t=0 
    #     '''
    #     N = round(self.center_density * (2*np.pi*self.sigma**2)**1.5)
    #     pos = np.random.normal(0,self.sigma*1E-6,(N,3))
    #     width = np.sqrt(k*self.temp/self.atom.mass)
    #     mom = np.random.normal(0,width,(N,3))
    #     self.data_points = np.stack([np.append(pos,mom,axis=1)])

    def build_density(self):
        '''
        Generates the Data points depending on the distribution

        returns: [6D array] slice at t=0 
        '''
        N = round(self.center_density * (2*np.pi*self.sigma**2)**1.5)
        radius = np.random.choice(self.samples,size=N,p=self.density)
        angles = np.random.uniform(0,2*np.pi,(N,2))
        pos = np.column_stack((radius*np.cos(angles[:,0])*np.sin(angles[:,1]),radius*np.sin(angles[:,0])*np.sin(angles[:,1]),radius*np.cos(angles[:,1])))
        width = np.sqrt(k*self.temp/self.atom.mass)
        mom = np.random.normal(0,width,(N,3))
        self.data_points = np.stack([np.append(pos,mom,axis=1)])

    def occupation_number(E, mu, beta, deg=0):
        '''
        Parameters
        ----------
        E : Particle Energy
        mu : Chemical Potential
        beta : 1/kT
        deg : bosons: 0, fermions: 1
            DESCRIPTION. The default is 0.

        Returns
        -------
        occupation number
        '''
        return 1/(np.exp(beta*(E-mu))+(-1)**deg)
    
    def harmonic_potential(self, r):
        V = 0.5 * self.atom.mass * (self.omega**2 * r**2)
        return V
    
    def density_distribution(self,mu=0):
        total = (-1)**self.deg * (1/self.de_Broglie**3)*polylog(1.5, (-1)**self.deg * np.exp((mu-self.harmonic_potential(self.samples))/(k*self.temp)))
        self.density = total/np.sum(total) 

    def thermal_denstiy(self,r,mu=0):
        total = (-1)**self.deg * ((k*self.temp)/(hbar*self.omega))**3 * polylog(3, [(-1)**self.deg * np.exp(mu/(k*self.temp))])
        self.density = total/np.sum(total) 

    def plot_3d(self,n,position=True):
        '''
        plots the distribution with x axis and y axis

        input:
        n - time slice to take
        
        out:
        matplotlib object
        '''
        if position:
            data = self.data_points[n][:,:3]
            labels = ['x','y','z']
        else:
            data = self.data_points[n][:,3:]
            labels= ['vx','vy','vz']

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='b', marker='.')
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_zlabel(labels[2])
        plt.show()

    def plot_2d(self,n,x=0,y=1):
        '''
        plots the distribution with x axis and y axis

        input:
        x - index to plot on x axis
        y - index to plot on y axis

        out:
        matplotlib object
        '''
        fig, ax =plt.subplots()

        if (x<3 and y>=3) or (x>=3 and y<3):
            raise IndexError('Plotting position against momentum')

        labels = ['x','y','z','px','py','pz']
        ax.scatter(self.data_points[n][:,x],self.data_points[n][:,y],marker='.')
        ax.set_xlabel(labels[x])
        ax.set_ylabel(labels[y])
        plt.show()

    def update_accel(self):
        '''
        Update the acceleration of each atom assumig in a spherical potential
        V = 0.5*m*w**2(x**2+y**2+z**2)
        a = - grad V = -0.5*w**2[2x,2y,2z]

        omega: trapping strength

        accel : 3xN array of accelerations [x,y,z]
        '''
        pos = self.data_points[-1][:,:3]
        return -self.omega**2*pos


    def update_pos(self,t_step):
        '''
        updates the position of particles by their veloity
        t_step: time interval (s) to update position
        '''
        accel = self.update_accel()
        pos = self.data_points[-1][:,:3] + t_step*self.data_points[-1][:,3:]+accel*t_step**2/2
        vel = self.data_points[-1][:,3:] + t_step*accel
        full_coords = np.stack([np.append(pos,vel,axis=1)])
        self.data_points = np.append(self.data_points,full_coords,axis=0)
        return full_coords

    def solve(self, duration, t_step=1E-6, savefile = None, save=False, anim=False):
        ''' 
        solves the position of atoms over a period of time includes t=0 and t=duration

        inputs:
        duration: total time to solve over (s)
        t_step: time incriments (s)
        '''
        if savefile:
            try: 
                self.data_points = np.load(savefile+".npy")
                return self.data_points
            except:
                print("no file can be found: "+savefile)
        time =0
        frames = int(duration/t_step)
        for i in tqdm(range(frames)):
            self.update_pos(t_step)
        if anim:
            self.animated_2d(int(duration/t_step))
        if save:
            np.save(savefile,self.data_points)
        return self.data_points

    def animated_2d(self,frames,x=0,y=0):
        '''
        plots an animated 2d solution to the movement of atoms

        frames: frame count (duration/t_step)
        '''
        #TODO: fix this
        fig, ax = plt.subplots()
        def animate(n):
            ax.clear()
            ax.scatter(self.data_points[n][:,0], self.data_points[n][:,1],marker = '.')
        ani = FuncAnimation(fig, animate ,frames=len(self.data_points))
        plt.show()

    def get_density(self, t, r=1e-7):
        count=[]
        for atom in self.data_points[t]:
            initial_pos = atom[:3]
            distances = self.data_points[t][:,:3]-initial_pos
            radius = np.hypot(np.hypot(distances[:,0],distances[:,1]),distances[:,2])
            count.append(np.count_nonzero(radius-r<0))
        return np.array([count])
    
    def radii(self):
        return np.hypot(np.hypot(self.data_points[:,:,0],self.data_points[:,:,1]),self.data_points[:,:,2])
    
    def plot_density(self,t, r=1e-7):
        plt.scatter(self.radii()[t],self.get_density(t,r))
        plt.xlabel('r')
        plt.ylabel('local density')