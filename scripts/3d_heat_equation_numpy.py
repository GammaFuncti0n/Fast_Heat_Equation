import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def main():
    task = HeatEquation(mu=mu)
    task.fit()

    # Create animation
    time_step = 1e-2
    task.plot_animation_2d(time_step, slice_axis='z', slice_index=task.z_num//2)
    #task.plot_animation_3d(time_step)

def mu(x, y, z):
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    return np.exp(-4*X)

class HeatEquation():
    def __init__(self, 
                 mu,
                 x_left=0, x_right=5, step_x=0.1,
                 y_left=-4, y_right=4, step_y=0.1,
                 z_left=-4, z_right=4, step_z=0.1,
                 T=100, tau=0.5,
                 k=10.5,
                 ):
        self.mu = mu
        self.x_left = x_left
        self.x_right = x_right
        self.step_x = step_x
        self.y_left = y_left
        self.y_right = y_right
        self.step_y = step_y
        self.z_left = z_left
        self.z_right = z_right
        self.step_z = step_z
        self.T = T 
        self.tau = tau
        self.k = k

        # Set grids
        self.grid_x = np.arange(self.x_left, self.x_right, self.step_x)
        self.grid_y = np.arange(self.y_left, self.y_right, self.step_y)
        self.grid_z = np.arange(self.z_left, self.z_right, self.step_z)
        self.grid_time = np.arange(0, self.T, self.tau)

        # Set lenght of grids
        self.x_num = len(self.grid_x)
        self.y_num = len(self.grid_y)
        self.z_num = len(self.grid_z)
        self.time_num = len(self.grid_time)

        # Set initial conditions
        self.u_x_y_z = self.mu(self.grid_x, self.grid_y, self.grid_z)

        # Set boundary conditions
        # Boundary conditions for X
        self.u_0_y_z = self.mu(self.x_left, self.grid_y, self.grid_z)
        self.u_a_y_z = self.mu(self.x_right, self.grid_y, self.grid_z)
        # Boundary conditions for Y
        self.u_x_0_z = self.mu(self.grid_x, self.y_left, self.grid_z)
        self.u_x_b_z = self.mu(self.grid_x, self.y_right, self.grid_z)
        # Boundary conditions for Z
        self.u_x_y_0 = self.mu(self.grid_x, self.grid_y, self.z_left)
        self.u_x_y_c = self.mu(self.grid_x, self.grid_y, self.z_right)

        # Set U
        U = np.zeros((self.time_num, self.x_num, self.y_num, self.z_num))

        # Initial condition
        U[0,:,:,:] = self.u_x_y_z
        # Boundary X
        U[:,0,:,:] = np.repeat(self.u_0_y_z, self.time_num, axis=0)
        U[:,-1,:,:] = np.repeat(self.u_a_y_z, self.time_num, axis=0)
        # Boundary Y
        U[:,:,0,:] = np.repeat(np.transpose(self.u_x_0_z, axes=(1,0,2)), self.time_num, axis=0)
        U[:,:,-1,:] = np.repeat(np.transpose(self.u_x_b_z, axes=(1,0,2)), self.time_num, axis=0)
        # Boundary Z
        U[:,:,:,0] = np.repeat(np.transpose(self.u_x_y_0, axes=(2,0,1)), self.time_num, axis=0)
        U[:,:,:,-1] = np.repeat(np.transpose(self.u_x_y_c, axes=(2,0,1)), self.time_num, axis=0)
        self.U = U
    
    def tridiag_mat(self, n):
        '''Function for make tridiagonal matrix
        [[-1,1,0,0,0],
         [1,-2,1,0,0],
         [0,1,-2,1,0],
         [0,0,1,-2,1],
         [0,0,0,1,-1]]
        '''
        L = np.diag([-2] * (n)) + np.diag([1] * (n - 1), 1) + np.diag([1] * (n - 1), -1)
        L[0, 0] = L[-1, -1] = -1
        return L
    
    def fit(self):
        Lx = self.tridiag_mat(self.x_num) * self.k/(self.step_x**2)
        Ly = self.tridiag_mat(self.y_num) * self.k/(self.step_y**2)
        Lz = self.tridiag_mat(self.z_num) * self.k/(self.step_z**2)
        
        Px = (np.eye(self.x_num) - (self.tau/2)*Lx)
        Py = (np.eye(self.y_num) - (self.tau/2)*Ly)
        Pz = (np.eye(self.z_num) - (self.tau/2)*Lz)

        for t_i in tqdm(range(self.time_num-1)):
            LU = np.einsum('ijk,il->ljk', self.U[t_i], Lx) +\
                 np.einsum('ijk,jl->ilk', self.U[t_i], Ly) +\
                 np.einsum('ijk,kl->ijl', self.U[t_i], Lz)

            w = np.einsum('kl,ijk->ijl', np.linalg.inv(Pz), LU)
            v = np.einsum('jl,ijk->ilk', np.linalg.inv(Py), w)
            du = np.einsum('il,ijk->ljk', np.linalg.inv(Px), v)

            self.U[t_i+1, 1:-1, 1:-1, 1:-1] = self.U[t_i, 1:-1, 1:-1, 1:-1] + self.tau*du[1:-1, 1:-1, 1:-1]
    
    def plot_animation_3d(self, time_step):
        '''Function create animation in 3d space with scatter plot
        There are should be a little number of point for example ~10 for one axis
        '''
        X, Y, Z = np.meshgrid(self.grid_x, self.grid_y, self.grid_z, indexing='ij')

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for i in range(1, self.grid_time.shape[0]):
            ax.clear()
            sc = ax.scatter(X, Y, Z, c=self.U[i], cmap='hot', alpha=0.6, vmin=self.U.min(), vmax=self.U.max())
            ax.set_title(f'Time {self.grid_time[i]:.2f} / {self.grid_time[-1]}')
            ax.set_xlim(self.x_left, self.x_right)
            ax.set_ylim(self.y_left, self.y_right)
            ax.set_zlim(self.z_left, self.z_right)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.grid(True)

            if i == 1:  # Создавать colorbar только один раз
                cbar = fig.colorbar(sc, ax=ax, shrink=0.5, aspect=10)
                cbar.set_label('Heat')

            plt.pause(time_step)
        
        plt.show()

    def plot_animation_2d(self, time_step, slice_axis='z', slice_index=0):
        '''Function create animation of slice in 2d space
        There are can be ~50 points for one axis
        params:
        :slice_axis (int or str): - axis where we create slice
        :slice_index (int): - index in array where we create slice
        '''
        if((slice_axis==0) or (slice_axis=='x')):
            actual_U = self.U[:,slice_index,:,:]
            extent=[self.y_left, self.y_right, self.z_left, self.z_right]
            xlabel = 'y'
            ylabel = 'z'
        elif((slice_axis==1) or (slice_axis=='y')):
            actual_U = self.U[:,:,slice_index,:]
            extent=[self.x_left, self.x_right, self.z_left, self.z_right]
            xlabel = 'x'
            ylabel = 'z'
        elif((slice_axis==2) or (slice_axis=='z')):
            actual_U = self.U[:,:,:,slice_index,]
            extent=[self.x_left, self.x_right, self.y_left, self.y_right]
            xlabel = 'x'
            ylabel = 'y'
        else:
            raise ValueError(f"There isn't slice_axis = {slice_axis}")

        plt.figure()
        for i in range(1, self.grid_time.shape[0]):
            plt.clf()
            plt.imshow(actual_U[i], origin="lower", extent=extent, cmap='hot', vmin=self.U.min(), vmax=self.U.max())
            plt.title(f'Time {self.grid_time[i]:.2f} / {self.grid_time[-1]}')
            plt.colorbar(label="Heat")
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.pause(time_step)
        
        plt.show()

if __name__=='__main__':
    main()
