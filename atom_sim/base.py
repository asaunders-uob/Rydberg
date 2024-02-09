import numpy as np
import matplotlib.pyplot as plt
from arc import *
from scipy.constants import k

class Atom_sim(object):
    '''
    Creates a 6D data point for N atoms
    [x,y,z,px,py,pz]
    atom: ARC atom object
    center_density: peak number density (units: 10^18 m^-3)
    sigma: std position of atoms
    temp: temperature of the atoms
    '''

    def __init__(self, atom, center_density=1, sigma=10, temp = 500E-9):

        self.atom = atom
        self.center_density=center_density
        self.sigma = sigma
        self.temp = temp
        self.build_density()

        
    def build_density(self):
        '''
        Generates the Data points depending on the distribution

        returns: 6D array 
        '''
        N = round(self.center_density * (2*np.pi*self.sigma**2)**1.5)
        pos = np.random.normal(0,self.sigma*1E-6,(N,3))
        width = np.sqrt(k*self.temp/self.atom.mass)
        mom = np.random.normal(0,width,(N,3))
        self.data_points = np.append(pos,mom,axis=1)
    
    def show_data(self,position=True):
        '''
        plots the distribution with x axis and y axis

        input:
        x - index to plot on x axis
        y - index to plot on y axis

        out:
        matplotlib object
        '''
        if position:
            data = self.data_points[-1][:,:3]
            labels = ['x','y','z']
        else:
            data = self.data_points[-1][:,3:]
            labels= ['vx','vy','vz']

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='b', marker='.')
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_zlabel(labels[2])
        plt.show()

    def update_pos(self,t_step):
        '''
        updates the position of particles by their veloity
        t_step: time interval (s) to update position
        '''
        accel =0
        pos = self.data_points[:,:3] + t_step*self.data_points[:,3:]+accel*t_step**2/2
        vel = self.data_points[:,3:] + t_step*accel
        full_coords = np.append(pos,vel,axis=1)
        self.data_points = np.stack([self.data_points,full_coords])
        return full_coords
