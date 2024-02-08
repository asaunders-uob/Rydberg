import numpy as np
import matplotlib.pyplot as plt

class Atom_sim(object):
    '''
    Creates a 6D data point for N atoms
    [x,y,z,px,py,pz]
    '''

    def __init__(self, N=10000, dist='square'):

        self.build_density(N,dist)

        
    def build_density(self,N,dist):
        '''
        Generates the Data points depending on the distribution

        returns: 6D array 
        '''
        pos = np.random.normal(0,1,(N,3))
        mom = np.random.normal(0,1,(N,3))
        self.data_points = np.append(pos,mom,axis=1)  
        return self.data_points
    
    def show_data(self,x,y):
        '''
        plots the distribution with x axis and y axis

        input:
        x - index to plot on x axis
        y - index to plot on y axis

        out:
        matplotlib object
        '''
        if (x<3 and y>=3) or (x>=3 and y<3):
            raise IndexError('Plotting position against momentum')

        labels = ['x','y','z','px','py','pz']

        plt.scatter(self.data_points[:,x],self.data_points[:,y],marker='.')
        plt.xlabel(labels[x])
        plt.ylabel(labels[y])

