import numpy as np
import matplotlib.pyplot as plt
from arc import *
from scipy.constants import k
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from functools import partial
from tqdm import tqdm

class Atom_sim(object):
    '''
    Creates a 6D data point for N atoms
    [x,y,z,px,py,pz]
    atom: ARC atom object
    center_density: peak number density (units: 10^18 m^-3)
    sigma: std position of atoms
    temp: temperature of the atoms
    '''

    def __init__(self, atom, center_density=1, sigma=10, temp = 500E-9, omega=1000):

        self.atom = atom
        self.center_density=center_density
        self.sigma = sigma
        self.temp = temp
        self.omega=omega
        self.build_density()

        
    def build_density(self):
        '''
        Generates the Data points depending on the distribution

        returns: [6D array] slice at t=0 
        '''
        N = round(self.center_density * (2*np.pi*self.sigma**2)**1.5)
        pos = np.random.normal(0,self.sigma*1E-6,(N,3))
        width = np.sqrt(k*self.temp/self.atom.mass)
        mom = np.random.normal(0,width,(N,3))
        self.data_points = np.stack([np.append(pos,mom,axis=1)])
    
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

    def solve(self, duration, t_step=1E-6, anim=False):
        ''' 
        solves the position of atoms over a period of time includes t=0 and t=duration

        inputs:
        duration: total time to solve over (s)
        t_step: time incriments (s)
        '''

        time =0
        frames = int(duration/t_step)
        for i in tqdm(range(frames)):
            self.update_pos(t_step)
        if anim:
            self.animated_2d(int(duration/t_step))
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