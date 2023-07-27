from random import sample
from shenfun import *
from ChannelFlow import KMM
#import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class MHDBoussinesq(KMM):
    """MHD Boussinesq channel flow solver

    Parameters
    ----------
    N : 3-tuple of ints
        The global shape in physical space (quadrature points)
    domain : 3-tuple of 2-tuples
        The size of the three domains
    Re : Reynolds number
    Rem : number
        model parameter
    Pr : Prandtl number
    dt : Timestep
    conv : Choose velocity convection method
        - 0 - Standard convection
        - 1 - Vortex type
    filename : str, optional
        Filenames are started with this name
    family : str, optional
        Chebyshev is normal, but Legendre works as well
    padding_factor : 3-tuple of numbers, optional
        For dealiasing, backward transforms to real space are
        padded with zeros in spectral space using these many points
    modplot : int, optional
        Plot some results every modplot timestep. If negative, no plotting
    modsave : int, optional
        Save results to hdf5 every modsave timestep
    moderror : int, optional
        Print diagnostics every moderror timestep
    checkpoint : int, optional
        Save required data for restart to hdf5 every checkpoint timestep
    sample_stats : int, optional
        Sample statistics every sample_stats timestep
    timestepper : str, optional
        Choose timestepper
        - 'IMEXRK222'
        - 'IMEXRK3'
        - 'IMEXRK443'

    Note
    ----
    Simulations may be killed gracefully by placing a file named 'killshenfun'
    in the folder running the solver from. The solver will then first store
    the results by checkpointing, before exiting.

    """
    def __init__(self,
                 N=(32, 32, 32),
                 domain=((-1, 1), (0, 2*np.pi), (0, 2*np.pi)),
                 Re=1000,
                 Rem=10000,
                 Ri=1,
                 Pr=0.7,
                 rho1=1,
                 rho2=1.05,
                 B0=(0, 1, 0),
                 dt=0.001,
                 conv=0,
                 filename='MHDBoussinesq',
                 family='C',
                 padding_factor=(1, 1.5, 1.5),
                 modplot=100,
                 modsave=1e8,
                 moderror=100,
                 checkpoint=1000,
                 timestepper='IMEXRK3'):
        utau = self.utau = 1
        KMM.__init__(self, N=N, domain=domain, nu=1/Re, dt=dt, conv=conv,
                     filename=filename, family=family, padding_factor=padding_factor,
                     modplot=modplot, modsave=modsave, moderror=moderror, dpdy=0,
                     checkpoint=checkpoint, timestepper=timestepper)
        self.Re = Re
        self.Rem = Rem
        self.Pr = Pr
        self.Ri = Ri
        self.rho1 = rho1
        self.rho2 = rho2
        self.B_0 = B0
        self.im1 = None
        self.im2 = None

        # New spaces and Functions used by MHD
        self.BX = FunctionSpace(N[0], family, bc=(B0[0], B0[0]), domain=self.D0.domain)
        self.BY = FunctionSpace(N[0], family, bc=(B0[1], B0[1]), domain=self.D0.domain)
        self.BZ = FunctionSpace(N[0], family, bc=(B0[2], B0[2]), domain=self.D0.domain)
        self.TBX = TensorProductSpace(comm, (self.BX, self.F1, self.F2), collapse_fourier=False, slab=True, modify_spaces_inplace=True)
        self.TBY = TensorProductSpace(comm, (self.BY, self.F1, self.F2), collapse_fourier=False, slab=True, modify_spaces_inplace=True)
        self.TBZ = TensorProductSpace(comm, (self.BZ, self.F1, self.F2), collapse_fourier=False, slab=True, modify_spaces_inplace=True)
        self.VB = VectorSpace([self.TBX, self.TBY, self.TBZ])      # B solution
        self.R0 = FunctionSpace(N[0], family, bc=(rho1, rho2), domain=self.D0.domain)
        self.TR = TensorProductSpace(comm, (self.R0, self.F1, self.F2), collapse_fourier=False, slab=True, modify_spaces_inplace=True)
        self.B_ = Function(self.VB)
        self.rho_ = Function(self.TR)
        self.NB_ = Function(self.CD)      # Convection nabla(B) dot B
        self.NBu_ = Function(self.CD)     # Convection nabla(B) dot u
        self.NuB_ = Function(self.CD)     # Convection nabla(u) dot B
        self.urho_ = Function(self.BD)    # u*rho

        # Classes for fast projections used by convection
        self.dB0dx = Project(Dx(self.B_[0], 0, 1), self.TC)
        self.dB0dy = Project(Dx(self.B_[0], 1, 1), self.TD)
        self.dB0dz = Project(Dx(self.B_[0], 2, 1), self.TD)
        self.dB1dx = Project(Dx(self.B_[1], 0, 1), self.TC)
        self.dB1dy = Project(Dx(self.B_[1], 1, 1), self.TD)
        self.dB1dz = Project(Dx(self.B_[1], 2, 1), self.TD)
        self.dB2dx = Project(Dx(self.B_[2], 0, 1), self.TC)
        self.dB2dy = Project(Dx(self.B_[2], 1, 1), self.TD)
        self.dB2dz = Project(Dx(self.B_[2], 2, 1), self.TD)

        # File for storing the results
        #self.file_w = ShenfunFile('_'.join((filename, 'B')), self.CD, backend='hdf5', mode='w', mesh='uniform')
        self.file_stats = open('_'.join((filename, 'stats'))+".h5", "w")
        self.file_rho = ShenfunFile('_'.join((filename, 'rho')), self.TR, backend='hdf5', mode='w', mesh='uniform')

        self.file_stats.write(f"{'Time':^11}{'UxUx':^11}{'UyUy':^11}{'UzUz':^11}{'SumUU':^11}{'BxBx':^11}{'ByBy':^11}{'BzBz':^11}{'SumBB:':^11}{'rho*rho':^11}{'rho':^11}{'div':^11}\n")
        # Create a checkpoint file used to restart simulations
        self.checkpoint.data['0']['B'] = [self.B_]
        self.checkpoint.data['0']['rho']=[self.rho_]

        h = TestFunction(self.TD)

        # Chebyshev matrices are not sparse, so need a tailored solver. Legendre has simply 5 nonzero diagonals and can use generic solvers.
        sol2 = chebyshev.la.Helmholtz if self.B0.family() == 'chebyshev' else la.SolverGeneric1ND

        # Modify u equation
        self.pdes['u'].N = [self.pdes['u'].N, Dx(self.NB_[0], 1, 2)+Dx(self.NB_[0], 2, 2)-Dx(Dx(self.NB_[1], 0, 1), 1, 1)-Dx(Dx(self.NB_[2], 0, 1), 2, 1), -self.Ri*(Dx(self.rho_, 1, 2)+Dx(self.rho_, 2, 2))]
        #self.pdes['u'].latex += r'+m \nu \nabla^2 (\nabla \times \vec{w})_x'

        # Modify g equation
        self.pdes['g'].N = [self.pdes['g'].N, Dx(self.NB_[2], 1, 1)-Dx(self.NB_[1], 2, 1)]
        #self.pdes['g'].latex += r'+m \nu (\nabla \times \nabla \times \vec{w})_x'

        if comm.Get_rank() == 0:
            # Modify v0 and w0 equations
            self.b1 = Function(self.D00)  # Copy from NB_[1, :, 0, 0] (cannot use view since not contiguous)
            self.b2 = Function(self.D00)  # Copy from NB_[2, :, 0, 0] (cannot use view since not contiguous)
            self.pdes1d['v0'].N = [-Expr(self.h1), Expr(self.b1)]
            self.pdes1d['w0'].N = [-Expr(self.h2), Expr(self.b2)]
            #self.pdes1d['v0'].latex += r'-m \nu \frac{\partial w_z}{\partial x}'
            #self.pdes1d['w0'].latex += r'+m \nu \frac{\partial w_y}{\partial x}'

        # MHD equations
        self.pdes['rho'] = self.PDE(TestFunction(self.TR),
                                    self.rho_,
                                    lambda f: (1/self.Re/self.Pr)*div(grad(f)),
                                    -div(self.urho_),
                                    dt=self.dt,
                                    solver=sol2,
                                    #latex=r"\frac{\partial w_x}{\partial t} +\vec{u} \cdot \nabla w_x = \kappa \nabla^2 w_x + \kappa N (\nabla \times \vec{u})_x"
                                    )

        self.pdes['B0'] = self.PDE(TestFunction(self.TBX),
                                   self.B_[0],
                                   lambda f: (1/self.Rem)*div(grad(f)),
                                   [Expr(self.NBu_[0]), -Expr(self.NuB_[0])],
                                   dt=self.dt,
                                   solver=sol2,
                                   #latex=r"\frac{\partial w_x}{\partial t} +\vec{u} \cdot \nabla w_x = \kappa \nabla^2 w_x + \kappa N (\nabla \times \vec{u})_x"
                                   )

        self.pdes['B1'] = self.PDE(TestFunction(self.TBY),
                                   self.B_[1],
                                   lambda f: (1/self.Rem)*div(grad(f)),
                                   [Expr(self.NBu_[1]), -Expr(self.NuB_[1])],
                                   dt=self.dt,
                                   solver=sol2,
                                   #latex=r"\frac{\partial w_y}{\partial t} +\vec{u} \cdot \nabla w_y = \kappa \nabla^2 w_y + \kappa N (\nabla \times \vec{u})_y"
                                   )

        self.pdes['B2'] = self.PDE(TestFunction(self.TBZ),
                                   self.B_[2],
                                   lambda f: (1/self.Rem)*div(grad(f)),
                                   [Expr(self.NBu_[2]), -Expr(self.NuB_[2])],
                                   dt=self.dt,
                                   solver=sol2,
                                   #latex=r"\frac{\partial w_z}{\partial t} +\vec{u} \cdot \nabla w_z = \kappa \nabla^2 w_z + \kappa N (\nabla \times \vec{u})_z"
                                   )

    def init_from_checkpoint(self):
        self.checkpoint.read(self.u_, 'U', step=0)
        self.checkpoint.read(self.B_, 'B', step=0)
        self.checkpoint.read(self.rho_, 'rho', step=0)
        self.g_[:] = 1j*self.K[1]*self.u_[2] - 1j*self.K[2]*self.u_[1]
        self.checkpoint.open()
        tstep = self.checkpoint.f.attrs['tstep']
        t = self.checkpoint.f.attrs['t']
        self.checkpoint.close()
        return t, tstep

    def convection(self):
        KMM.convection(self)
        BB = self.NB_
        Bu = self.NBu_
        uB = self.NuB_
        Bp = self.B_.backward(padding_factor=self.padding_factor)
        up = self.up
        dB0dxp = self.dB0dx().backward(padding_factor=self.padding_factor)
        dB0dyp = self.dB0dy().backward(padding_factor=self.padding_factor)
        dB0dzp = self.dB0dz().backward(padding_factor=self.padding_factor)
        dB1dxp = self.dB1dx().backward(padding_factor=self.padding_factor)
        dB1dyp = self.dB1dy().backward(padding_factor=self.padding_factor)
        dB1dzp = self.dB1dz().backward(padding_factor=self.padding_factor)
        dB2dxp = self.dB2dx().backward(padding_factor=self.padding_factor)
        dB2dyp = self.dB2dy().backward(padding_factor=self.padding_factor)
        dB2dzp = self.dB2dz().backward(padding_factor=self.padding_factor)
        dudxp = self.dudxp
        dudyp = self.dudyp
        dudzp = self.dudzp
        dvdxp = self.dvdxp
        dvdyp = self.dvdyp
        dvdzp = self.dvdzp
        dwdxp = self.dwdxp
        dwdyp = self.dwdyp
        dwdzp = self.dwdzp
        BB[0] = self.TDp.forward(Bp[0]*dB0dxp+Bp[1]*dB0dyp+Bp[2]*dB0dzp, BB[0])
        BB[1] = self.TDp.forward(Bp[0]*dB1dxp+Bp[1]*dB1dyp+Bp[2]*dB1dzp, BB[1])
        BB[2] = self.TDp.forward(Bp[0]*dB2dxp+Bp[1]*dB2dyp+Bp[2]*dB2dzp, BB[2])
        BB.mask_nyquist(self.mask)
        uB[0] = self.TDp.forward(up[0]*dB0dxp+up[1]*dB0dyp+up[2]*dB0dzp, uB[0])
        uB[1] = self.TDp.forward(up[0]*dB1dxp+up[1]*dB1dyp+up[2]*dB1dzp, uB[1])
        uB[2] = self.TDp.forward(up[0]*dB2dxp+up[1]*dB2dyp+up[2]*dB2dzp, uB[2])
        uB.mask_nyquist(self.mask)
        Bu[0] = self.TDp.forward(Bp[0]*dudxp+Bp[1]*dudyp+Bp[2]*dudzp, Bu[0])
        Bu[1] = self.TDp.forward(Bp[0]*dvdxp+Bp[1]*dvdyp+Bp[2]*dvdzp, Bu[1])
        Bu[2] = self.TDp.forward(Bp[0]*dwdxp+Bp[1]*dwdyp+Bp[2]*dwdzp, Bu[2])
        Bu.mask_nyquist(self.mask)
        rhop = self.rho_.backward(padding_factor=self.padding_factor)
        self.urho_ = up.function_space().forward(self.up*rhop, self.urho_)

    def tofile(self, tstep):
        #self.file_u.write(tstep, {'u': [self.u_.backward(mesh='uniform')]}, as_scalar=True)
        #self.file_w.write(tstep, {'B': [self.B_.backward(mesh='uniform')]}, as_scalar=True)
        self.file_rho.write(tstep, {'rho': [self.rho_.backward(mesh='uniform')]}, as_scalar=True)
        


    def compute_vw(self, rk):
        if comm.Get_rank() == 0:
            self.b1[:] = self.NB_[1, :, 0, 0].real
            self.b2[:] = self.NB_[2, :, 0, 0].real
        KMM.compute_vw(self, rk)

    def initialize(self, rand=0.001, from_checkpoint=False):
        if from_checkpoint:
            return self.init_from_checkpoint()
        X = self.TR.local_mesh(bcast=True)
        rhoa_ = Array(self.TR)
        
        rhoa_[:] =  (1.025+0.025*X[0])+0.001*np.cos(X[1])*np.cos(X[2])*(1-X[0])*(1+X[0]) # 3d single-mode initial perturbation
        #rhoa_[:] =  (1.025+0.025*X[0])+0.001*(np.sin(X[1])*np.sin(X[2])+np.sin(2*X[1])*np.sin(2*X[2])+np.sin(4*X[1])*np.sin(4*X[2])+np.sin(8*X[1])*np.sin(8*X[2])+np.sin(16*X[1])*np.sin(16*X[2]))*(1-X[0])*(1+X[0]) # 3d multi-mode initial perturbation
        
        self.rho_ = self.TR.forward(rhoa_, self.rho_)
        self.rho_.mask_nyquist(self.mask)
        self.B_[0, 0, 0, 0] = self.B_0[0]
        self.B_[1, 0, 0, 0] = self.B_0[1]
        self.B_[2, 0, 0, 0] = self.B_0[2]
        return 0, 0

    def print_energy_and_divergence(self, t, tstep):
        if tstep % self.moderror == 0 and self.moderror > 0:
            
            ub = self.u_.backward(self.ub,mesh="uniform")
            rhob = self.rho_.backward(mesh="uniform")
            bb = self.B_.backward(mesh="uniform")
            eu0 = inner(1, ub[0]*ub[0])
            eu1 = inner(1, ub[1]*ub[1])
            eu2 = inner(1, ub[2]*ub[2])
            eu3 = inner(1, ub[0]*ub[0]+ub[1]*ub[1]+ub[2]*ub[2])
            eb0 = inner(1, bb[0]*bb[0])
            eb1 = inner(1, bb[1]*bb[1])
            eb2 = inner(1, bb[2]*bb[2])
            eb3 = inner(1,bb[0]*bb[0] + bb[1]*bb[1] + bb[2]*bb[2])

            divu = self.divu().backward(mesh="uniform")
            e3 = np.sqrt(inner(1, divu*divu))
            d0 = inner(1, rhob*rhob)/(8*np.pi**2)
            d1 = inner(1, rhob)/(8*np.pi**2)
            if comm.Get_rank() == 0:
                print(f"{t:2.5f} {eu0:2.6e} {eu1:2.6e} {eu2:2.6e} {eu3:2.6e} {eb0:2.6e} {eb1:2.6e} {eb2:2.6e} {eb3:2.6e} {d0:2.7e} {d1:2.7e} {e3:2.6e} \n")
                self.file_stats.write(f"{t:2.5f} {eu0:2.6e} {eu1:2.6e} {eu2:2.6e} {eu3:2.6e} {eb0:2.6e} {eb1:2.6e} {eb2:2.6e} {eb3:2.6e} {d0:2.7e} {d1:2.7e} {e3:2.6e} \n")
                
    def init_plots(self):
    	"""
        rhob = self.rho_.backward()
        self.X = self.TD.local_mesh(True)
        self.im1 = 1
        if comm.Get_rank() == 0:
            plt.figure(1, figsize=(6, 3))
            self.im1 = plt.contourf(self.X[1][:, :, 0], self.X[0][:, :, 0], rhob[:, :, 0], 100)
            plt.colorbar(self.im1)
            plt.draw()
        """

    def plot(self, t, tstep):
        """
        if self.im1 is None and self.modplot > 0:
            self.init_plots()

        if tstep % self.modplot == 0 and self.modplot > 0:
            rhob = self.rho_.backward()
            if comm.Get_rank() == 0:
                X = self.X
                self.im1.axes.clear()
                self.im1.axes.contourf(X[1][:, :, 0], X[0][:, :, 0], rhob[:, :, 0], 100)
                self.im1.autoscale()
                plt.figure(1)
                plt.pause(1e-6)
        """

if __name__ == '__main__':
    from time import time
    N = (128,256,256)
    B0 = (0, 0, 0)
    d = {
        'N': N,
        'Re': 10000.,
        'Pr': 1,
        'dt': 0.001,
        'B0': B0,
        'filename': f'MHDB_3d_single_{N[0]}_{N[1]}_{N[2]}_B_{B0[0]}_{B0[1]}_{B0[2]}',
        'conv': 0,
        'modplot': 10,
        'moderror': 100,
        'modsave': 10000,
        'family': 'C',
        'checkpoint': 100,
        #'padding_factor': 1,
        'timestepper': 'IMEXRK3'
        }
    c = MHDBoussinesq(**d)
    t, tstep = c.initialize(rand=0.0, from_checkpoint=False)
    t0 = time()
    c.solve(t=t, tstep=tstep, end_time=60)
    print('Computing time %2.4f'%(time()-t0))
    if comm.Get_rank() == 0:
        #generate_xdmf('_'.join((filename, 'u'))+'.h5')
        generate_xdmf('_'.join((d['filename'], 'rho'))+'.h5')
        #generate_xdmf('_'.join((filename, 'B'))+'.h5')
