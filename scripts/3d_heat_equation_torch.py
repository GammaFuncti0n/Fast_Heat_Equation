import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #device = 'cpu'
    print(f'Device is {device}')
    task = HeatEquation(mu=mu, device=device)
    task.fit()

    # Create animation
    time_step = 1e-2
    #task.plot_animation_2d(time_step, slice_axis='z', slice_index=task.z_num//2)
    task.plot_animation_3d(time_step, resolution=20)

def mu(x, y, z):
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    return torch.exp(-X**2-Y**2-Z**2)

class HeatEquation():
    def __init__(self, 
                 mu,
                 x_left=-3, x_right=3, step_x=0.05,
                 y_left=-3, y_right=3, step_y=0.05,
                 z_left=-3, z_right=3, step_z=0.05,
                 T=1, tau=0.1,
                 k=0.1,
                 device='cuda'
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
        self.device = device

        # Set grids
        self.grid_x = torch.arange(self.x_left, self.x_right, self.step_x, dtype=torch.float32)
        self.grid_y = torch.arange(self.y_left, self.y_right, self.step_y, dtype=torch.float32)
        self.grid_z = torch.arange(self.z_left, self.z_right, self.step_z, dtype=torch.float32)
        self.grid_time = torch.arange(0, self.T, self.tau, dtype=torch.float32)

        # Set lenght of grids
        self.x_num = len(self.grid_x)
        self.y_num = len(self.grid_y)
        self.z_num = len(self.grid_z)
        self.time_num = len(self.grid_time)

        # Set initial conditions
        u_x_y_z = self.mu(self.grid_x, self.grid_y, self.grid_z)

        # Set boundary conditions
        # Boundary conditions for X
        u_0_y_z = self.mu(torch.tensor([self.x_left], dtype=torch.float32), self.grid_y, self.grid_z)
        u_a_y_z = self.mu(torch.tensor([self.x_right], dtype=torch.float32), self.grid_y, self.grid_z)
        # Boundary conditions for Y
        u_x_0_z = self.mu(self.grid_x, torch.tensor([self.y_left], dtype=torch.float32), self.grid_z)
        u_x_b_z = self.mu(self.grid_x, torch.tensor([self.y_right], dtype=torch.float32), self.grid_z)
        # Boundary conditions for Z
        u_x_y_0 = self.mu(self.grid_x, self.grid_y, torch.tensor([self.z_left], dtype=torch.float32))
        u_x_y_c = self.mu(self.grid_x, self.grid_y, torch.tensor([self.z_right], dtype=torch.float32))

        # Set U
        U = torch.zeros((self.time_num, self.x_num, self.y_num, self.z_num), dtype=torch.float32)

        # Initial condition
        U[0,:,:,:] = u_x_y_z
        # Boundary X
        U[:,0,:,:] = u_0_y_z.repeat((self.time_num,1,1))
        U[:,-1,:,:] = u_a_y_z.repeat((self.time_num,1,1))
        # Boundary Y
        U[:,:,0,:] = u_x_0_z.repeat((1,self.time_num,1)).permute(1,0,2)
        U[:,:,-1,:] = u_x_b_z.repeat((1,self.time_num,1)).permute(1,0,2)
        # Boundary Z
        U[:,:,:,0] = u_x_y_0.repeat((1,1,self.time_num)).permute(2,0,1)
        U[:,:,:,-1] = u_x_y_c.repeat((1,1,self.time_num)).permute(2,0,1)
        self.U = U.to(self.device)
    
    def tridiag_mat(self, n):
        '''Function for make tridiagonal matrix
        [[-1,1,0,0,0],
         [1,-2,1,0,0],
         [0,1,-2,1,0],
         [0,0,1,-2,1],
         [0,0,0,1,-1]]
        '''
        L = torch.diag(-2*torch.ones(n, device=self.device)) + torch.diag(torch.ones(n-1, device=self.device), 1) + torch.diag(torch.ones(n-1, device=self.device), -1)
        L[0, 0] = L[-1, -1] = -1
        return L
    
    def fit(self):
        Lx = self.tridiag_mat(self.x_num) * self.k/(self.step_x**2)
        Ly = self.tridiag_mat(self.y_num) * self.k/(self.step_y**2)
        Lz = self.tridiag_mat(self.z_num) * self.k/(self.step_z**2)
        
        Px = (torch.eye(self.x_num, device=self.device) - (self.tau/2)*Lx)
        Py = (torch.eye(self.y_num, device=self.device) - (self.tau/2)*Ly)
        Pz = (torch.eye(self.z_num, device=self.device) - (self.tau/2)*Lz)

        for t_i in tqdm(range(self.time_num-1)):
            LU = torch.einsum('ijk,il->ljk', self.U[t_i], Lx) +\
                 torch.einsum('ijk,jl->ilk', self.U[t_i], Ly) +\
                 torch.einsum('ijk,kl->ijl', self.U[t_i], Lz)

            w = torch.einsum('kl,ijk->ijl', torch.linalg.inv(Pz), LU)
            v = torch.einsum('jl,ijk->ilk', torch.linalg.inv(Py), w)
            du = torch.einsum('il,ijk->ljk', torch.linalg.inv(Px), v)

            self.U[t_i+1, 1:-1, 1:-1, 1:-1] = self.U[t_i, 1:-1, 1:-1, 1:-1] + self.tau*du[1:-1, 1:-1, 1:-1]
    
    def plot_animation_3d(self, time_step, resolution):
        '''Function create animation in 3d space with scatter plot
        params:
        :time_step (float): duration of pause between frames
        :resolution (int): number of points per axis, it should be ~10-15
        '''
        X, Y, Z = torch.meshgrid(self.grid_x, self.grid_y, self.grid_z, indexing='ij')

        points_x = self.x_num//resolution
        points_x = points_x if points_x!=0 else 1
        points_y= self.y_num//resolution
        points_y = points_y if points_y!=0 else 1
        points_z = self.z_num//resolution
        points_z = points_z if points_z!=0 else 1
        X = X[::points_x, ::points_y, ::points_z]
        Y = Y[::points_x, ::points_y, ::points_z]
        Z = Z[::points_x, ::points_y, ::points_z]
        U = self.U[:, ::points_x, ::points_y, ::points_z].cpu().numpy()

        d1, d2, d3 = np.meshgrid(
            2*(U.shape[1]//2 - np.arange(U.shape[1]))/U.shape[1],
            2*(U.shape[2]//2 - np.arange(U.shape[2]))/U.shape[2],
            2*(U.shape[3]//2 - np.arange(U.shape[3]))/U.shape[3],
            indexing='ij'
        )
        alpha = np.exp(-4*((d1)**2+(d2)**2+(d3)**2))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        points_t = self.grid_time.shape[0]//66
        points_t = points_t if points_t!=0 else 1
        for i in range(0, self.grid_time.shape[0], 1):
            ax.clear()
            sc = ax.scatter(X, Y, Z, c=U[i], cmap='hot', alpha=0.4, vmin=self.U.min(), vmax=self.U.max())
            ax.set_title(f'Time {self.grid_time[i]:.2f} / {self.grid_time[-1]:.2f}')
            ax.set_xlim(self.x_left, self.x_right)
            ax.set_ylim(self.y_left, self.y_right)
            ax.set_zlim(self.z_left, self.z_right)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.grid(True)

            if i == 0:
                cbar = fig.colorbar(sc, ax=ax, shrink=0.5, aspect=10)
                cbar.set_label('Heat')

            plt.pause(time_step)
        
        plt.show()

    def plot_animation_2d(self, time_step, slice_axis='z', slice_index=0):
        '''Function create animation of slice in 2d space
        params:
        :slice_axis (int or str): - axis where we create slice
        :slice_index (int): - index in array where we create slice
        '''
        if((slice_axis==0) or (slice_axis=='x')):
            actual_U = self.U[:,slice_index,:,:].cpu().numpy()
            extent=[self.y_left, self.y_right, self.z_left, self.z_right]
            xlabel = 'y'
            ylabel = 'z'
        elif((slice_axis==1) or (slice_axis=='y')):
            actual_U = self.U[:,:,slice_index,:].cpu().numpy()
            extent=[self.x_left, self.x_right, self.z_left, self.z_right]
            xlabel = 'x'
            ylabel = 'z'
        elif((slice_axis==2) or (slice_axis=='z')):
            actual_U = self.U[:,:,:,slice_index,].cpu().numpy()
            extent=[self.y_left, self.y_right, self.x_left, self.x_right]
            xlabel = 'y'
            ylabel = 'x'
        else:
            raise ValueError(f"There isn't slice_axis = {slice_axis}")

        plt.figure()
        for i in range(0, self.grid_time.shape[0], self.grid_time.shape[0]//66):
            plt.clf()
            plt.imshow(actual_U[i], origin="lower", extent=extent, cmap='hot', vmin=self.U.min(), vmax=self.U.max())
            plt.title(f'Time {self.grid_time[i]:.2f} / {self.grid_time[-1]:.2f}')
            plt.colorbar(label="Heat")
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.pause(time_step)
        
        plt.show()

if __name__=='__main__':
    main()