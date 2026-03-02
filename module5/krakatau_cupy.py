import cupy as cp
import numpy as np
from time import perf_counter
import sys

# Precision control
PREC = 8  # 4 for float32, 8 for float64

if PREC == 4:
    real_t = np.float32
    real_t_cp = cp.float32
elif PREC == 8:
    real_t = np.float64
    real_t_cp = cp.float64
else:
    real_t = np.float32
    real_t_cp = cp.float32

class Sim_Configuration:
    """ Configuration class for simulation parameters """
    def __init__(self, arguments):
        self.scale = 1.0           # Scaling factor for grid resolution
        self.tend = 1800.0         # Simulation duration in seconds
        self.fout = 180.0          # Output interval in seconds
        self.filename = "output.dat"  # name of the output file with history
        
        i = 1
        while i < len(arguments):
            arg = arguments[i]
            if arg == "-h":  # Write help
                print("python krakatau_cupy.py --scale <scaling factor for grid resolution> --tend <duration in seconds>",
                      "--fout <duration in seconds between each snapshot> --out <name of output file>")
                sys.exit(0)
            elif i == len(arguments) - 1:
                raise ValueError(f"The last argument ({arg}) must have a value")
            else:
                if arg == "--scale":
                    self.scale = float(arguments[i + 1])
                    if self.scale <= 0:
                        raise ValueError("scale must be a positive real number (e.g. --scale 1.0)")
                elif arg == "--tend":
                    self.tend = float(arguments[i + 1])
                    if self.tend < 0:
                        raise ValueError("tend must be a positive real number (e.g. --tend 3600)")
                elif arg == "--fout":
                    self.fout = float(arguments[i + 1])
                    if self.fout < 0:
                        raise ValueError("fout must be a positive real number (e.g. --fout 60)")
                elif arg == "--out":
                    self.filename = arguments[i + 1]
                else:
                    print("---> error: the argument type is not recognized")                
                i += 2

def rescale_grid(scale, H, lon, lat, dx, dy):
    """
    Rescale the grid resolution by a scale factor using bilinear interpolation.

    scale : factor to apply to number of grid cells. >1 finer grid, <1 coarser grid
    H : Elevation data (ny, nx)
    lon, lat : Longitude and latitude arrays
    dx, dy : Grid spacing in meters

    Returns: new_H, new_lon, new_lat, new_dx, new_dy, new_nx, new_ny
    """
    old_ny, old_nx = H.shape

    new_nx = int(old_nx * scale)    # new resolution
    new_ny = int(old_ny * scale)
    
    new_dx = dx * (old_nx / new_nx) # New grid spacing
    new_dy = dy * (old_ny / new_ny)
    
    dlon = lon[1] - lon[0]          # Original grid spacing in degrees
    dlat = lat[1] - lat[0]

    # Physical domain boundaries (cell edges, not centers)
    # Since lon/lat are cell centers, the domain extends half a cell beyond
    lon_min = lon[0] - dlon / 2
    lon_max = lon[-1] + dlon / 2
    lat_min = lat[0] - dlat / 2
    lat_max = lat[-1] + dlat / 2
    
    # New grid spacing in degrees (same physical domain, different number of cells)
    new_dlon = (lon_max - lon_min) / new_nx
    new_dlat = (lat_max - lat_min) / new_ny
    
    # New coordinate arrays (cell centers of the new grid)
    # First cell center is at domain_min + new_spacing/2
    new_lon = lon_min + 0.5*new_dlon + np.arange(new_nx) * new_dlon
    new_lat = lat_min + 0.5*new_dlat + np.arange(new_ny) * new_dlat
    
    # Convert new lon/lat to fractional indices in the old grid
    # lon[i] = lon_min + dlon/2 + i*dlon, so i = (lon - lon[0]) / dlon
    x_new = (new_lon - lon[0]) / dlon  # fractional index in old grid
    y_new = (new_lat - lat[0]) / dlat  # fractional index in old grid
    
    # Get integer and fractional parts for bilinear interpolation. In range [-1, n-1]
    x0 = np.floor(x_new).astype(int)
    y0 = np.floor(y_new).astype(int)

    # get the next indices for interpolation. In range [0, n], but clamp to n-1
    x1 = np.minimum(x0 + 1, old_nx - 1)
    y1 = np.minimum(y0 + 1, old_ny - 1)
    
    # Clamp x0 and y0 to [0, n-1]
    x0 = np.maximum(x0, 0)
    y0 = np.maximum(y0, 0)
    
    # Fractional parts (distance from lower index)
    xf = np.clip(x_new - x0, 0, 1)
    yf = np.clip(y_new - y0, 0, 1)
    
    # Bilinear interpolation
    new_H = np.empty((new_ny, new_nx), dtype=np.float64)
    for j in range(new_ny):
        # Interpolate along x for both y0[j] and y1[j] rows
        row_y0 = H[y0[j], x0] * (1 - xf) + H[y0[j], x1] * xf
        row_y1 = H[y1[j], x0] * (1 - xf) + H[y1[j], x1] * xf
        # Interpolate along y
        new_H[j, :] = row_y0 * (1 - yf[j]) + row_y1 * yf[j]
    
    return new_H, new_lon, new_lat, new_dx, new_dy, new_nx, new_ny

# This is the core of the solver where we update eta, u, and v at each time step.
def step(eta, u, v, g, f_at_u, f_at_v, H_at_u, H_at_v, land_mask, wall_u, wall_v, \
         dx, dy, dt, friction, nx, ny):
    """
    Advance solution by one time step using leapfrog in time
    1. Update η using current u, v (continuity equation)
    2. Update u, v using new η (momentum equations)
    """

    # FIXME: All arrays in this function should be CuPy arrays on the GPU
    # FIXME: This function should update eta, u, and v in-place on the GPU, without returning anything.
    # FIXME: When it works with array syntax, proceed tu use cupy elementwise kernels or raw CUDA kernels
    # FIXME: for better performance, but the array syntax should be fast enough for a first version.
    # FIXME: Look out for race conditions, since all threads run asynchronously in a kernel.

    #############################################################################    
    # Step 1: ∂η/∂t = -∂(Hu)/∂x - ∂(Hv)/∂y (continuity equation)
    dflux = (H_at_u[:, 1:] * u[:, 1:] - H_at_u[:, :-1] * u[:, :-1]) / dx + \
            (H_at_v[1:, :] * v[1:, :] - H_at_v[:-1, :] * v[:-1, :]) / dy

    # Laplacian diffusion for eta (∇²η)
    # Pad eta for boundary handling
    eta_padded = np.pad(eta, 1, mode='edge') # Pad with edge values for simple Neumann boundary conditions (zero gradient)
    laplacian_eta = (
        (eta_padded[2:, 1:-1] - 2 * eta + eta_padded[:-2, 1:-1]) / (dy * dy) +
        (eta_padded[1:-1, 2:] - 2 * eta + eta_padded[1:-1, :-2]) / (dx * dx)
    )

    # Diffusion coefficient is proportional to wave speed and grid spacing, with a small factor for stability
    diffusion = 0.01 * np.sqrt(g * H_at_u[:, :-1]) * min(dx, dy)

    eta -= dt * (dflux + diffusion * laplacian_eta)

    # Apply land mask and open boundary conditions (radiation): zero gradient in wave height
    eta[land_mask] = 0.0
    eta[0, :] = eta[1, :]
    eta[-1, :] = eta[-2, :]
    eta[:, 0] = eta[:, 1]
    eta[:, -1] = eta[:, -2]

    #############################################################################    
    # Step 2: ∂u/∂t = +fv - g·∂η/∂x - r·u
    #         ∂v/∂t = -fu - g·∂η/∂y - r·v

    # Pre calculate v at u faces and u at v faces for Coriolis terms so that u and v can be updated in-place
    v_at_u = 0.25 * (v[:-1, :-1] + v[:-1, 1:] + v[1:, :-1] + v[1:, 1:])  # (ny, nx-1) v at u faces is average of 4 surrounding v points
    u_at_v = 0.25 * (u[:-1, :-1] + u[:-1, 1:] + u[1:, :-1] + u[1:, 1:])  # (ny-1, nx)

    # Solve for ∂u/∂t = +fv - g·∂η/∂x - r·u in the interior faces in the x-direction (i=1 to nx-1, all j)
    deta_dx = (eta[:, 1:] - eta[:, :-1]) / dx                            # (ny, nx-1) deta/dx at u faces
    u[:, 1:-1] += dt * (f_at_u[:, 1:-1] * v_at_u - g * deta_dx - friction * u[:, 1:-1])

    # Apply wall boundary conditions (zero velocity at land boundaries)
    # and open boundary conditions (radiation): zero gradient in velocity in x-direction
    u[:, 1:-1] = np.where(wall_u[:, 1:-1], 0.0, u[:, 1:-1])
    u[:, 0] = u[:, 1]
    u[:, -1] = u[:, -2]

    # Solve for ∂v/∂t = -fu - g·∂η/∂y - r·v in the interior faces in the y-direction (all i, j=1 to ny-1)
    deta_dy = (eta[1:, :] - eta[:-1, :]) / dy                # (ny-1, nx)
    v[1:-1, :] += dt * (-f_at_v[1:-1, :] * u_at_v - g * deta_dy - friction * v[1:-1, :])

    # Apply wall boundary conditions and open boundary conditions
    v[1:-1, :] = np.where(wall_v[1:-1, :], 0.0, v[1:-1, :])
    v[0, :] = v[1, :]
    v[-1, :] = v[-2, :]

class ShallowWaterSolver:
    """
    Linear Shallow Water Equations Solver using staggered grid in space and in time:

    ∂η/∂t + ∂(Hu)/∂x + ∂(Hv)/∂y = 0                    (continuity)
    ∂u/∂t - fv = -g·∂η/∂x - r·u                        (x-velocity)    
    ∂v/∂t + fu = -g·∂η/∂y - r·v                        (y-velocity)

    Variables are staggered as:    
    - η (eta): Water surface elevation at cell centers (ny, nx)
    - H: Water depth at rest at cell centers (ny, nx)
    - u: x-velocity at cell faces (ny, nx+1)
    - v: y-velocity at cell faces (ny+1, nx)
    
    We use a staggered leapfrog scheme where:
    - η is updated using u, v at the current time
    - u, v are updated using η at the new time
    
    This is equivalent to the scheme used in many operational ocean models and is stable for CFL < 1.    
    """
    
    def __init__(self, elevation, lon, lat, dx, dy, cfl=0.5, friction=1e-5):
        """ Initialize the shallow water solver with bathymetry and grid information """
        self.H = np.maximum(-elevation, 0.0) # Water depth (positive below sea level, oppposite sign of elevation)
        self.lon = lon # Longitude array (1D)
        self.lat = lat # Latitude array (1D)
        self.dx = dx   # Grid spacing in meters
        self.dy = dy
        self.cfl = cfl # CFL number for stability
        self.friction = friction # Linear bottom friction coefficient (1/s)

        self.ny, self.nx = elevation.shape # Number of grid points in y and x directions
        self.g = 9.81 # Gravitational acceleration (m/s^2)
        
        self.land_mask = np.ascontiguousarray(self.H < 1., dtype=bool) # Land mask (where depth is very small)

        # Precompute H at u and v faces
        self.H_at_u = np.zeros((self.ny, self.nx + 1), dtype=real_t)
        self.H_at_u[:, 1:-1] = 0.5 * (self.H[:, :-1] +  self.H[:, 1:])
        self.H_at_u[:, 0] = self.H[:, 0]
        self.H_at_u[:, -1] = self.H[:, -1]
        
        self.H_at_v = np.zeros((self.ny + 1, self.nx), dtype=real_t)
        self.H_at_v[1:-1, :] = 0.5*(self.H[:-1, :] + self.H[1:, :])
        self.H_at_v[0, :] = self.H[0, :]
        self.H_at_v[-1, :] = self.H[-1, :]
        
        # Land masks for u and v grids
        self.land_u = self.H_at_u < 1. # Land mask at u faces
        self.land_v = self.H_at_v < 1. # Land mask at v faces        
        
        # Zero velocity where either adjacent cell is land
        self.wall_u = np.zeros((self.ny, self.nx + 1), dtype=bool)
        self.wall_u[:, 1:-1] = self.land_mask[:, :-1] | self.land_mask[:, 1:]
        
        self.wall_v = np.zeros((self.ny + 1, self.nx), dtype=bool)
        self.wall_v[1:-1, :] = self.land_mask[:-1, :] | self.land_mask[1:, :]

        # Coriolis parameter: f = 2·Ω·sin(φ)
        omega = 7.2921e-5 # Earth's rotation rate in radians/s
        self.f = (2 * omega * np.sin(np.deg2rad(lat))).astype(real_t)   # Coriolis parameter at cell centers (ny,)
        self.f_at_u = self.f[:, np.newaxis] * np.ones((1, self.nx + 1)) # Coriolis parameter at u faces (same as cell centers, since f varies only in y)
        self.f_at_v = np.zeros((self.ny + 1, self.nx), dtype=real_t)    # Coriolis parameter at v faces (average of adjacent cell centers in y direction)
        self.f_at_v[1:-1, :] = 0.5 * (self.f[:-1, np.newaxis] + self.f[1:, np.newaxis])
        self.f_at_v[0, :] = self.f[0]
        self.f_at_v[-1, :] = self.f[-1]
        
        # Initialize state variables
        self.eta = np.zeros((self.ny, self.nx), dtype=real_t)   # Surface elevation at cell centers
        self.u = np.zeros((self.ny, self.nx + 1), dtype=real_t) # x-velocity at u faces
        self.v = np.zeros((self.ny + 1, self.nx), dtype=real_t) # y-velocity at v faces
        
        # Calculate stable time step
        max_depth = np.max(self.H)
        c_max = np.sqrt(self.g * max_depth)
        self.dt = self.cfl * min(self.dx, self.dy) / c_max
        
        device = cp.cuda.Device()         # Setup and print GPU info
        print(f"Using GPU: {device.id} - {cp.cuda.runtime.getDeviceProperties(device.id)['name'].decode()}")
        print(f"Shallow Water Solver initialized:")
        print(f"  Grid size: {self.ny} x {self.nx}")
        print(f"  Grid spacing: dx={self.dx:.1f}m, dy={self.dy:.1f}m")
        print(f"  Max depth: {max_depth:.1f}m, Max wave speed: {c_max:.1f}m/s")
        print(f"  Time step: {self.dt:.2f}s (CFL={self.cfl})")
        print(f"  Bottom friction: {self.friction:.2e} 1/s")
        print(f"  Land cells: {np.sum(self.land_mask)} ({100*np.sum(self.land_mask)/self.land_mask.size:.1f}%)")

    def set_gaussian_initial_condition(self, i_center, j_center, amplitude=20.0, spread=20):
        """Set a Gaussian initial condition for the surface elevation"""
        X, Y = np.meshgrid(np.arange(self.nx), np.arange(self.ny))
        r2 = ((X - j_center) / spread)**2 + ((Y - i_center) / spread)**2
        self.eta = (amplitude * np.exp(-0.5 * r2)).astype(real_t)
        self.eta[self.land_mask] = 0.0
        print(f"Gaussian IC: center=({i_center},{j_center}), amplitude={amplitude:.1f}m, spread={spread} cells")
                
    def run(self, duration, output_interval=None):
        """Run simulation for specified duration."""
        n_steps = int(duration / self.dt)
        self.dt = duration / n_steps # Adjust dt to fit exactly into duration
        output_every = max(1, int(output_interval / self.dt)) if output_interval else max(1, n_steps // 100)
        
        times, performance, etas = [], [], []

        print(f"Running: {duration:.1f}s ({n_steps} steps), output every {output_every * self.dt:.1f}s")

        # FIXME make sure all arrays passed to step are on the GPU as CuPy arrays, 
        # and that step updates eta, u, v in-place on the GPU

        tlast = perf_counter()
        nupdated = 0
        for n in range(n_steps):
            step(
                self.eta, self.u, self.v, self.g, self.f_at_u, self.f_at_v,
                self.H_at_u, self.H_at_v, self.land_mask, self.wall_u, self.wall_v,
                self.dx, self.dy, self.dt, self.friction, self.nx, self.ny
            )
            nupdated += self.nx * self.ny # increment number of updated grid points (for performance measurement)
            if n % output_every == 0:
                tnow = perf_counter() # Wall clock time since last performance measurement
                performance.append(1e9 * (tnow - tlast) / nupdated) # nano-seconds per grid point update
                nupdated = 0
                times.append(n * self.dt)
                # FIXME make sure etas is appended with a CPU copy of eta
                # ADVANCED FIXME: Use streams and events to overlap compute and data transfer
                # ADVANCED FIXME: by having a temporary GPU array to store eta snapshots before copying
                # ADVANCED FIXME: it to CPU asynchronously while the next steps are computed on the GPU. 
                etas.append(self.eta.copy())
                if n % (10 * output_every) == 0:
                    max_eta = np.max(np.abs(self.eta))
                    print(f"  Step {n}/{n_steps}, t={n*self.dt:.1f}s, max|η|={max_eta:.2f}m")
                tlast = tnow
        
        if nupdated > 0:
            tnow = perf_counter() # Wall clock time since last performance measurement
            performance.append(1e9 * (tnow - tlast) / nupdated) # nano-seconds per grid point update
            times.append(n_steps * self.dt)
            # FIXME make sure etas is appended with a CPU copy of eta
            etas.append(self.eta.copy())
        print("Simulation complete!")
        
        return {'time': np.array(times, dtype=real_t), 'performance': np.array(performance, dtype=real_t),
                'eta': np.array(etas, dtype=real_t), 'lon': self.lon, 'lat': self.lat}

def simulate(config):
    input_file = './bathymetry_krakatau.bin'

    with open(input_file, 'rb') as f:
        nx = int.from_bytes(f.read(4), byteorder='little', signed=True)
        ny = int.from_bytes(f.read(4), byteorder='little', signed=True)
        lon = np.frombuffer(f.read(8*nx), dtype=np.float64)
        lat = np.frombuffer(f.read(8*ny), dtype=np.float64)
        elevation = np.frombuffer(f.read(8*nx*ny), dtype=np.float64).reshape((ny, nx))

    lon = lon.astype(real_t)
    lat = lat.astype(real_t)
    elevation = elevation.astype(real_t)

    # Grid spacing in degrees
    dlon = np.abs(lon[1] - lon[0])
    dlat = np.abs(lat[1] - lat[0])

    # Convert to meters (approximate, assume one degree is the same in latitude and longitude, since we are near the equator)
    dx = dlon * 111000  # meters
    dy = dlat * 111000  # meters

    # Rescale grid according to config.scale (e.g. scale=0.5 will reduce resolution by half, scale=2 will double)
    elevation, lon, lat, dx, dy, nx, ny = rescale_grid(config.scale, elevation, lon, lat, dx, dy)

    # Find Krakatau location
    krakatau_lon = 105.423
    krakatau_lat = -6.102

    j_krakatau = np.argmin(np.abs(lon - krakatau_lon))
    i_krakatau = np.argmin(np.abs(lat - krakatau_lat))

    # Create the solver instance
    solver = ShallowWaterSolver(
        elevation, lon, lat, dx, dy, 
        cfl=0.4,         # CFL number (stable for CFL < 1)
        friction=1e-5    # Bottom friction in 1/s
    )

    # Set Gaussian initial condition centered on Krakatau
    solver.set_gaussian_initial_condition(
        i_center=i_krakatau, 
        j_center=j_krakatau, 
        amplitude=40.0,  # 40 m amplitude
        spread=10        # 10 cells spread
    )

    # Run simulation
    results = solver.run(config.tend, output_interval=config.fout)

    # Average performance skipping first point, as it contains startup overhead and is not representative of steady-state performance.
    print("\nAverage performance is {:.2f} ns/grid point update".format(np.mean(results['performance'][1:])))

    # Save results to file as binary (compatible with C code)
    nframes = len(results['time'])
    print(f"\nSaving results to {config.filename} ({nframes} frames, {nx}x{ny} nx x ny grid)")
    with open(config.filename, 'wb') as f:
        f.write(np.array(nframes, dtype=np.int32).tobytes())
        f.write(np.array(nx, dtype=np.int32).tobytes())
        f.write(np.array(ny, dtype=np.int32).tobytes())
        f.write(results['lon'].astype(np.float32).tobytes())
        f.write(results['lat'].astype(np.float32).tobytes())
        f.write(np.array(results['time'], dtype=np.float32).tobytes())
        f.write(np.array(results['performance'], dtype=np.float32).tobytes())
        f.write(np.array(elevation, dtype=np.float32).tobytes())
        for i in range(nframes):
            f.write(results['eta'][i].astype(np.float32).tobytes())

if __name__ == "__main__":
    """ parse the command line and start the simulation """
    config = Sim_Configuration(sys.argv)
    simulate(config)