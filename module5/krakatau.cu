/**
 * Shallow Water Equations Solver - C++ Implementation
 * 
 * Linear Shallow Water Equations Solver using staggered grid in space and in time:
 *
 * ∂η/∂t + ∂(Hu)/∂x + ∂(Hv)/∂y = 0                    (continuity)
 * ∂u/∂t - fv = -g·∂η/∂x - r·u                        (x-velocity)    
 * ∂v/∂t + fu = -g·∂η/∂y - r·v                        (y-velocity)
 *
 * Variables are staggered as:    
 * - η (eta): Water surface elevation at cell centers (ny, nx)
 * - H: Water depth at rest at cell centers (ny, nx)
 * - u: x-velocity at cell faces (ny, nx+1)
 * - v: y-velocity at cell faces (ny+1, nx)
 * 
 * We use a staggered leapfrog scheme where:
 * - η is updated using u, v at the current time
 * - u, v are updated using η at the new time
 * 
 * This is equivalent to the scheme used in many operational ocean models and is stable for CFL < 1.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <chrono>
#include <cstring>
#include <cuda_runtime.h>

// Block size for CUDA kernels (you can experiment with different sizes for performance)
// This is how many threads will be launched per block in x and y directions.
// 256 threads per block is a common choice, but you can try different sizes to see how performance changes.
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

// Precision control
#define PREC 8  // 4 for float, 8 for double

#if PREC == 4
    using real_t = float;
#elif PREC == 8
    using real_t = double;
#else
    using real_t = float;
#endif

// 2D array class for convenient indexing
template<typename T>
class Array2D {
private:
    std::vector<T> data;
    int rows, cols;
public:
    Array2D() : rows(0), cols(0) {}
    Array2D(int ny, int nx) : rows(ny), cols(nx), data(ny * nx, T(0)) {}
    Array2D(int ny, int nx, T val) : rows(ny), cols(nx), data(ny * nx, val) {}
    
    T& operator()(int j, int i) { return data[j * cols + i]; }
    const T& operator()(int j, int i) const { return data[j * cols + i]; }
    
    int ny() const { return rows; }
    int nx() const { return cols; }
    T* ptr() { return data.data(); }
    const T* ptr() const { return data.data(); }
    
    void resize(int ny, int nx) {
        rows = ny;
        cols = nx;
        data.resize(ny * nx, T(0));
    }
    
    void fill(T val) {
        std::fill(data.begin(), data.end(), val);
    }
    
    Array2D<T> copy() const {
        Array2D<T> result(rows, cols);
        result.data = data;
        return result;
    }
};

// Configuration class for simulation parameters
class Sim_Configuration {
public:
    double scale = 1.0;              // Scaling factor for grid resolution
    double tend = 3600.0;            // Simulation duration in seconds
    double fout = 60.0;              // Output interval in seconds
    std::string filename = "output.dat";  // name of the output file with history

    Sim_Configuration(int argc, char* argv[]) {
        int i = 1;
        while (i < argc) {
            std::string arg = argv[i];
            if (arg == "-h") {  // Write help
                std::cout << "Usage: ./krakatau --scale <scaling factor for grid resolution> --tend <duration in seconds> "
                          << "--fout <duration in seconds between each snapshot> --out <name of output file>" << std::endl;
                exit(0);
            } else if (i == argc - 1) {
                throw std::runtime_error("The last argument (" + arg + ") must have a value");
            } else {
                if (arg == "--scale") {
                    scale = std::stod(argv[i + 1]);
                    if (scale <= 0) {
                        throw std::runtime_error("scale must be a positive real number (e.g. --scale 1.0)");
                    }
                } else if (arg == "--tend") {
                    tend = std::stod(argv[i + 1]);
                    if (tend < 0) {
                        throw std::runtime_error("tend must be a positive real number (e.g. --tend 3600)");
                    }
                } else if (arg == "--fout") {
                    fout = std::stod(argv[i + 1]);
                    if (fout < 0) {
                        throw std::runtime_error("fout must be a positive real number (e.g. --fout 60)");
                    }
                } else if (arg == "--out") {
                    filename = argv[i + 1];
                } else {
                    std::cout << "---> error: the argument type is not recognized" << std::endl;
                }
                i += 2;
            }
        }
    }
};

/**
 * Rescale the grid resolution by a scale factor using bilinear interpolation.
 *
 * scale : factor to apply to number of grid cells. >1 finer grid, <1 coarser grid
 * H : Elevation data (ny, nx)
 * lon, lat : Longitude and latitude arrays
 * dx, dy : Grid spacing in meters
 *
 * Returns: new_H, new_lon, new_lat, new_dx, new_dy, new_nx, new_ny (via references)
 */
void rescale_grid(double scale, 
                  const Array2D<real_t>& H, const std::vector<real_t>& lon, const std::vector<real_t>& lat,
                  double dx, double dy,
                  Array2D<real_t>& new_H, std::vector<real_t>& new_lon, std::vector<real_t>& new_lat,
                  double& new_dx, double& new_dy, int& new_nx, int& new_ny) {
    
    int old_ny = H.ny();
    int old_nx = H.nx();
    
    new_nx = static_cast<int>(old_nx * scale);    // new resolution
    new_ny = static_cast<int>(old_ny * scale);
    
    new_dx = (dx * old_nx) / new_nx; // New grid spacing
    new_dy = (dy * old_ny) / new_ny;
    
    double dlon = lon[1] - lon[0];   // Original grid spacing in degrees
    double dlat = lat[1] - lat[0];

    // Physical domain boundaries (cell edges, not centers)
    // Since lon/lat are cell centers, the domain extends half a cell beyond
    double lon_min = lon[0] - dlon / 2;
    double lon_max = lon[old_nx - 1] + dlon / 2;
    double lat_min = lat[0] - dlat / 2;
    double lat_max = lat[old_ny - 1] + dlat / 2;
    
    // New grid spacing in degrees (same physical domain, different number of cells)
    double new_dlon = (lon_max - lon_min) / new_nx;
    double new_dlat = (lat_max - lat_min) / new_ny;
    
    // New coordinate arrays (cell centers of the new grid)
    // First cell center is at domain_min + new_spacing/2
    new_lon.resize(new_nx);
    new_lat.resize(new_ny);
    for (int i = 0; i < new_nx; i++) {
        new_lon[i] = lon_min + 0.5 * new_dlon + i * new_dlon;
    }
    for (int j = 0; j < new_ny; j++) {
        new_lat[j] = lat_min + 0.5 * new_dlat + j * new_dlat;
    }
    
    // Convert new lon/lat to fractional indices in the old grid
    std::vector<double> x_new(new_nx), y_new(new_ny);
    for (int i = 0; i < new_nx; i++) {
        x_new[i] = (new_lon[i] - lon[0]) / dlon;
    }
    for (int j = 0; j < new_ny; j++) {
        y_new[j] = (new_lat[j] - lat[0]) / dlat;
    }
    
    // Bilinear interpolation
    new_H.resize(new_ny, new_nx);
    for (int j = 0; j < new_ny; j++) {
        // Get integer and fractional parts for bilinear interpolation
        int y0 = std::max(0, std::min(old_ny - 1, static_cast<int>(std::floor(y_new[j]))));
        int y1 = std::min(old_ny - 1, y0 + 1);
        double yf = std::max(0.0, std::min(1.0, y_new[j] - y0));
        
        for (int i = 0; i < new_nx; i++) {
            int x0 = std::max(0, std::min(old_nx - 1, static_cast<int>(std::floor(x_new[i]))));
            int x1 = std::min(old_nx - 1, x0 + 1);
            double xf = std::max(0.0, std::min(1.0, x_new[i] - x0));
            
            // Interpolate along x for both y0 and y1 rows
            double row_y0 = H(y0, x0) * (1 - xf) + H(y0, x1) * xf;
            double row_y1 = H(y1, x0) * (1 - xf) + H(y1, x1) * xf;
            // Interpolate along y
            new_H(j, i) = row_y0 * (1 - yf) + row_y1 * yf;
        }
    }
}

// ============================================================================
// CUDA kernels (implement the same logic as the original step routine)
// ============================================================================

/* Example
// Step 1a: ∂η/∂t = -∂(Hu)/∂x - ∂(Hv)/∂y (continuity equation)
__global__ void kernel_update_eta(real_t* eta, const real_t* u, const real_t* v,
                                   const real_t* H_at_u, const real_t* H_at_v,
                                   const char* land_mask,
                                   real_t dx, real_t dy, real_t dt, int nx, int ny) {
    ... write the code for updating eta incl land mask but excl boundaries (why ?) ...
}*/

/**
 * Advance solution by one time step using leapfrog in time:
 * 1. Update η using current u, v (continuity equation)
 * 2. Update u, v using new η (momentum equations)
 */

 /* Need to pass CUDA GPU arrays as pointers, no Array2D class available
 void step(real_t* d_eta, real_t* d_u, real_t* d_v,
          real_t g, const real_t* d_H_at_u, const real_t* d_H_at_v,
          const char* d_land_mask, const char* d_wall_u, const char* d_wall_v,
          real_t dx, real_t dy, real_t dt, int nx, int ny) {

    // declare the size of the grids and blocks that will be launched on the GPU for each kernel
    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    
    // we add BLOCK_SIZE_X/Y - 1 to ensure we cover the entire
    // domain even if it is not a perfect multiple of the block size
    dim3 grid((nx + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X,
              (ny + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
    // Do we need different block/grid sizes for different kernels?

    // Step 1: ∂η/∂t = -∂(Hu)/∂x - ∂(Hv)/∂y (continuity equation)
    kernel_update_eta<<<grid, block>>>(d_eta, d_u, d_v,
                                       d_H_at_u, d_H_at_v, d_land_mask,
                                       dx, dy, dt, nx, ny);

    ... rest of the kernels ...
*/

// FIXME Should be removed and replaced by CUDA version that launches kernels instead
void step(Array2D<real_t>& eta, Array2D<real_t>& u, Array2D<real_t>& v,
          real_t g, const Array2D<real_t>& H_at_u, const Array2D<real_t>& H_at_v,
          const Array2D<char>& land_mask, const Array2D<char>& wall_u, const Array2D<char>& wall_v,
          real_t dx, real_t dy, real_t dt, int nx, int ny) {
    
    // Step 1: ∂η/∂t = -∂(Hu)/∂x - ∂(Hv)/∂y (continuity equation)
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            real_t flux_x_right = H_at_u(j, i+1) * u(j, i+1);
            real_t flux_x_left = H_at_u(j, i) * u(j, i);
            real_t flux_y_top = H_at_v(j+1, i) * v(j+1, i);
            real_t flux_y_bottom = H_at_v(j, i) * v(j, i);
            
            real_t dflux_x = (flux_x_right - flux_x_left) / dx;
            real_t dflux_y = (flux_y_top - flux_y_bottom) / dy;
            
            eta(j, i) = eta(j, i) - dt * (dflux_x + dflux_y);
            
            // Apply land mask
            if (land_mask(j, i)) {
                eta(j, i) = 0.0;
            }
        }
    }

    // Open boundary conditions (radiation): zero gradient
    for (int i = 0; i < nx; i++) {
        eta(0, i) = eta(1, i);
        eta(ny-1, i) = eta(ny-2, i);
    }
    for (int j = 0; j < ny; j++) {
        eta(j, 0) = eta(j, 1);
        eta(j, nx-1) = eta(j, nx-2);
    }

    // Step 2: ∂u/∂t = +fv - g·∂η/∂x - r·u
    //         ∂v/∂t = -fu - g·∂η/∂y - r·v

    Array2D<real_t> u_new(ny, nx + 1);
    Array2D<real_t> v_new(ny + 1, nx);

    // Update u at interior faces (i=1 to nx-1, all j)
    for (int j = 0; j < ny; j++) {
        for (int i = 1; i < nx; i++) {
            real_t deta_dx = (eta(j, i) - eta(j, i-1)) / dx;                        // Pressure gradient
            u_new(j, i) = u(j, i) - dt * g * deta_dx;
            if (wall_u(j, i)) u_new(j, i) = 0.0; // Apply wall boundary conditions (zero velocity at land boundaries)
        }
    }

    // Update v at interior faces (all i, j=1 to ny-1)
    for (int j = 1; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            real_t deta_dy = (eta(j, i) - eta(j-1, i)) / dy;                        // Pressure gradient
            v_new(j, i) = v(j, i) - dt * g * deta_dy;            
            if (wall_v(j, i)) v_new(j, i) = 0.0; // Apply wall boundary conditions (zero velocity at land boundaries)
        }
    }

    // Open boundary: allow outflow (zero gradient)
    for (int j = 0; j < ny; j++) {
        u_new(j, 0) = u_new(j, 1);
        u_new(j, nx) = u_new(j, nx-1);
    }
    for (int i = 0; i < nx; i++) {
        v_new(0, i) = v_new(1, i);
        v_new(ny, i) = v_new(ny-1, i);
    }
    
    // Update u and v
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx + 1; i++) {
            u(j, i) = u_new(j, i);
        }
    }
    for (int j = 0; j < ny + 1; j++) {
        for (int i = 0; i < nx; i++) {
            v(j, i) = v_new(j, i);
        }
    }
}

/**
 * Shallow Water Solver class
 */
class ShallowWaterSolver {
public:
    Array2D<real_t> H;           // Water depth (positive below sea level)
    std::vector<real_t> lon;     // Longitude array (1D)
    std::vector<real_t> lat;     // Latitude array (1D)
    real_t dx, dy;               // Grid spacing in meters
    real_t cfl;                  // CFL number for stability
    int ny, nx;                  // Number of grid points in y and x directions
    real_t g = 9.81;             // Gravitational acceleration (m/s^2)
    
    Array2D<char> land_mask;     // Land mask (where depth is very small)
    Array2D<real_t> H_at_u;      // H at u faces
    Array2D<real_t> H_at_v;      // H at v faces
    Array2D<char> land_u;        // Land mask at u faces
    Array2D<char> land_v;        // Land mask at v faces
    Array2D<char> wall_u;        // Wall mask at u faces
    Array2D<char> wall_v;        // Wall mask at v faces
    
    Array2D<real_t> eta;         // Surface elevation at cell centers
    Array2D<real_t> u;           // x-velocity at u faces
    Array2D<real_t> v;           // y-velocity at v faces
    
    real_t dt;                   // Time step

    /** Initialize the shallow water solver with bathymetry and grid information */
    ShallowWaterSolver(const Array2D<real_t>& elevation, 
                       const std::vector<real_t>& lon_in, const std::vector<real_t>& lat_in,
                       real_t dx_in, real_t dy_in, real_t cfl_in = 0.5)
        : lon(lon_in), lat(lat_in), dx(dx_in), dy(dy_in), cfl(cfl_in){
        
        ny = elevation.ny();
        nx = elevation.nx();
        
        // Water depth (positive below sea level, opposite sign of elevation)
        H.resize(ny, nx);
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                H(j, i) = std::max(-elevation(j, i), 0.0);
            }
        }
        
        // Land mask (where depth is very small)
        land_mask.resize(ny, nx);
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                land_mask(j, i) = (H(j, i) < 1.0);
            }
        }

        // Precompute H at u and v faces
        H_at_u.resize(ny, nx + 1);
        for (int j = 0; j < ny; j++) {
            H_at_u(j, 0) = H(j, 0);
            for (int i = 1; i < nx; i++) {
                H_at_u(j, i) = 0.5 * (H(j, i-1) + H(j, i));
            }
            H_at_u(j, nx) = H(j, nx-1);
        }
        
        H_at_v.resize(ny + 1, nx);
        for (int i = 0; i < nx; i++) {
            H_at_v(0, i) = H(0, i);
            for (int j = 1; j < ny; j++) {
                H_at_v(j, i) = 0.5 * (H(j-1, i) + H(j, i));
            }
            H_at_v(ny, i) = H(ny-1, i);
        }
        
        // Land masks for u and v grids
        land_u.resize(ny, nx + 1);
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx + 1; i++) {
                land_u(j, i) = (H_at_u(j, i) < 1.0);
            }
        }
        
        land_v.resize(ny + 1, nx);
        for (int j = 0; j < ny + 1; j++) {
            for (int i = 0; i < nx; i++) {
                land_v(j, i) = (H_at_v(j, i) < 1.0);
            }
        }
        
        // Zero velocity where either adjacent cell is land
        wall_u.resize(ny, nx + 1);
        wall_u.fill(false);
        for (int j = 0; j < ny; j++) {
            for (int i = 1; i < nx; i++) {
                wall_u(j, i) = land_mask(j, i-1) || land_mask(j, i);
            }
        }
        
        wall_v.resize(ny + 1, nx);
        wall_v.fill(false);
        for (int j = 1; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                wall_v(j, i) = land_mask(j-1, i) || land_mask(j, i);
            }
        }
        
        // Initialize state variables
        eta.resize(ny, nx);
        eta.fill(0.0);
        u.resize(ny, nx + 1);
        u.fill(0.0);
        v.resize(ny + 1, nx);
        v.fill(0.0);
        
        // Calculate stable time step
        real_t max_depth = 0;
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                max_depth = std::max(max_depth, H(j, i));
            }
        }
        real_t c_max = std::sqrt(g * max_depth);
        dt = cfl * std::min(dx, dy) / c_max;
        
        // Count land cells
        int land_count = 0;
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                if (land_mask(j, i)) land_count++;
            }
        }
        
        std::cout << "Shallow Water Solver initialized:" << std::endl;
        std::cout << "  Grid size: " << ny << " x " << nx << std::endl;
        std::cout << "  Grid spacing: dx=" << dx << "m, dy=" << dy << "m" << std::endl;
        std::cout << "  Max depth: " << max_depth << "m, Max wave speed: " << c_max << "m/s" << std::endl;
        std::cout << "  Time step: " << dt << "s (CFL=" << cfl << ")" << std::endl;
        std::cout << "  Land cells: " << land_count << " (" << 100.0 * land_count / (ny * nx) << "%)" << std::endl;
    }

    /** Set a Gaussian initial condition for the surface elevation */
    void set_gaussian_initial_condition(int i_center, int j_center, real_t amplitude = 20.0, real_t spread = 20.0) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                real_t dx_ic = (i - j_center) / spread;
                real_t dy_ic = (j - i_center) / spread;
                real_t r2 = dx_ic * dx_ic + dy_ic * dy_ic;
                eta(j, i) = amplitude * std::exp(-0.5 * r2);
                if (land_mask(j, i)) {
                    eta(j, i) = 0.0;
                }
            }
        }
        std::cout << "Gaussian IC: center=(" << i_center << "," << j_center 
                  << "), amplitude=" << amplitude << "m, spread=" << spread << " cells" << std::endl;
    }

    /** Run simulation for specified duration */
    void run(double duration, double output_interval,
             std::vector<real_t>& times, std::vector<real_t>& performance, 
             std::vector<Array2D<real_t>>& etas) {
        
        int n_steps = static_cast<int>(duration / dt);
        int output_every = output_interval > 0 ? 
                           std::max(1, static_cast<int>(output_interval / dt)) : 
                           std::max(1, n_steps / 100);
        
        std::cout << "Running: " << duration << "s (" << n_steps << " steps), output every " 
                  << output_every * dt << "s" << std::endl;

        // FIXME --- Allocate data and copy arrays to GPU ---
        real_t *d_eta;
        cudaMalloc(&d_eta, ny *  nx * sizeof(real_t));
        // ... and more ...

        cudaMemcpy(d_eta, eta.ptr(), ny *  nx * sizeof(real_t), cudaMemcpyHostToDevice);
        // ... and more ...

        // --------------------------
        
        auto tlast = std::chrono::high_resolution_clock::now();
        long long nupdated = 0;
        
        for (int n = 0; n < n_steps; n++) {
            // FIXME Replace with call to CUDA version
            step(eta, u, v, g, H_at_u, H_at_v, land_mask, wall_u, wall_v, dx, dy, dt, nx, ny);
            nupdated += nx * ny; // increment number of updated grid points (for performance measurement)
            
            if (n % output_every == 0) {
                // We need to add a synchronise here to make sure all GPU computations are done
                cudaDeviceSynchronize();
                // FIXME copy d_eta back to CPU

                auto tnow = std::chrono::high_resolution_clock::now(); // Wall clock time since last performance measurement
                double elapsed = std::chrono::duration<double>(tnow - tlast).count();
                performance.push_back(1e9 * elapsed / nupdated); // nano-seconds per grid point update
                nupdated = 0;
                times.push_back(n * dt);
                etas.push_back(eta.copy());
                
                if (n % (10 * output_every) == 0) {
                    real_t max_eta = 0;
                    for (int j = 0; j < ny; j++) {
                        for (int i = 0; i < nx; i++) {
                            max_eta = std::max(max_eta, std::abs(eta(j, i)));
                        }
                    }
                    std::cout << "  Step " << n << "/" << n_steps << ", t=" << n * dt 
                              << "s, max|η|=" << max_eta << "m" << std::endl;
                }
                tlast = tnow;
            }
        }
        
        if (nupdated > 0) {
            // FIXME -- Same as above
            auto tnow = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(tnow - tlast).count();
            performance.push_back(1e9 * elapsed / nupdated);
            times.push_back(n_steps * dt);
            etas.push_back(eta.copy());
        }

        // FIXME --- Free GPU memory ---
        cudaFree(d_eta);
        // ... and more ...
        // -----------------------

        std::cout << "Simulation complete!" << std::endl;
    }
};

void simulate(const Sim_Configuration& config) {
    std::string input_file = "./bathymetry_krakatau.bin";
    
    std::ifstream fin(input_file, std::ios::binary);
    if (!fin) {
        throw std::runtime_error("Cannot open input file: " + input_file);
    }
    
    int32_t nx, ny;
    fin.read(reinterpret_cast<char*>(&nx), sizeof(int32_t));
    fin.read(reinterpret_cast<char*>(&ny), sizeof(int32_t));
    
    std::vector<double> lon_d(nx), lat_d(ny);
    fin.read(reinterpret_cast<char*>(lon_d.data()), nx * sizeof(double));
    fin.read(reinterpret_cast<char*>(lat_d.data()), ny * sizeof(double));
    
    std::vector<double> elevation_d(nx * ny);
    fin.read(reinterpret_cast<char*>(elevation_d.data()), nx * ny * sizeof(double));
    fin.close();
    
    // Convert to real_t
    std::vector<real_t> lon(nx), lat(ny);
    for (int i = 0; i < nx; i++) lon[i] = lon_d[i];
    for (int j = 0; j < ny; j++) lat[j] = lat_d[j];
    
    Array2D<real_t> elevation(ny, nx);
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            elevation(j, i) = elevation_d[j * nx + i];
        }
    }

    // Grid spacing in degrees
    double dlon = std::abs(lon[1] - lon[0]);
    double dlat = std::abs(lat[1] - lat[0]);

    // Convert to meters (approximate, assume one degree is the same in latitude and longitude, since we are near the equator)
    real_t dx = dlon * 111000;  // meters
    real_t dy = dlat * 111000;  // meters

    // Rescale grid according to config.scale (e.g. scale=0.5 will reduce resolution by half, scale=2 will double)
    Array2D<real_t> new_elevation;
    std::vector<real_t> new_lon, new_lat;
    real_t new_dx, new_dy;
    int new_nx, new_ny;
    rescale_grid(config.scale, elevation, lon, lat, dx, dy,
                 new_elevation, new_lon, new_lat, new_dx, new_dy, new_nx, new_ny);
    
    // Update variables
    elevation = new_elevation;
    lon = new_lon;
    lat = new_lat;
    dx = new_dx;
    dy = new_dy;
    nx = new_nx;
    ny = new_ny;

    // Find Krakatau location
    const real_t krakatau_lon = 105.423;
    const real_t krakatau_lat = -6.102;

    int j_krakatau = 0, i_krakatau = 0;
    real_t min_lon_diff = std::abs(lon[0] - krakatau_lon);
    real_t min_lat_diff = std::abs(lat[0] - krakatau_lat);
    for (int i = 1; i < nx; i++) {
        if (std::abs(lon[i] - krakatau_lon) < min_lon_diff) {
            min_lon_diff = std::abs(lon[i] - krakatau_lon);
            j_krakatau = i;
        }
    }
    for (int j = 1; j < ny; j++) {
        if (std::abs(lat[j] - krakatau_lat) < min_lat_diff) {
            min_lat_diff = std::abs(lat[j] - krakatau_lat);
            i_krakatau = j;
        }
    }

    // Create the solver instance
    ShallowWaterSolver solver(elevation, lon, lat, dx, dy, 0.4); // CFL number (stable for CFL < 1)

    // Set Gaussian initial condition centered on Krakatau
    solver.set_gaussian_initial_condition(
        i_krakatau, 
        j_krakatau, 
        40.0,   // 40 m amplitude
        10      // 10 cells spread
    );

    // Run simulation
    std::vector<real_t> times, performance;
    std::vector<Array2D<real_t>> etas;
    solver.run(config.tend, config.fout, times, performance, etas);

    // Average performance skipping first point, as it contains startup overhead
    // and is not representative of steady-state performance.
    if (performance.size() > 1) {
        double perf_sum = 0.0;
        for (size_t k = 1; k < performance.size(); k++) perf_sum += performance[k];
        std::cout << "\nAverage performance is "
                  << perf_sum / (performance.size() - 1)
                  << " ns/grid point update" << std::endl;
    }

    // Save results to file as binary (compatible with Python code)
    int nframes = times.size();
    std::cout << "\nSaving results to " << config.filename << " (" << nframes 
              << " frames, " << nx << "x" << ny << " nx x ny grid)" << std::endl;
    
    std::ofstream fout(config.filename, std::ios::binary);    
    int32_t nframes_i = nframes, nx_i = nx, ny_i = ny;
    fout.write(reinterpret_cast<char*>(&nframes_i), sizeof(int32_t));
    fout.write(reinterpret_cast<char*>(&nx_i), sizeof(int32_t));
    fout.write(reinterpret_cast<char*>(&ny_i), sizeof(int32_t));
    
    // Write lon, lat as float32
    std::vector<float> lon_f(nx), lat_f(ny);
    for (int i = 0; i < nx; i++) lon_f[i] = lon[i];
    for (int j = 0; j < ny; j++) lat_f[j] = lat[j];
    fout.write(reinterpret_cast<char*>(lon_f.data()), nx * sizeof(float));
    fout.write(reinterpret_cast<char*>(lat_f.data()), ny * sizeof(float));
    
    // Write times and performance as float32
    std::vector<float> times_f(nframes), perf_f(nframes);
    for (int i = 0; i < nframes; i++) {
        times_f[i] = times[i];
        perf_f[i] = performance[i];
    }
    fout.write(reinterpret_cast<char*>(times_f.data()), nframes * sizeof(float));
    fout.write(reinterpret_cast<char*>(perf_f.data()), nframes * sizeof(float));
    
    // Write elevation as float32
    std::vector<float> elev_f(nx * ny);
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            elev_f[j * nx + i] = elevation(j, i);
        }
    }
    fout.write(reinterpret_cast<char*>(elev_f.data()), nx * ny * sizeof(float));
    
    // Write eta frames as float32
    std::vector<float> eta_f(nx * ny);
    for (int frame = 0; frame < nframes; frame++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                eta_f[j * nx + i] = etas[frame](j, i);
            }
        }
        fout.write(reinterpret_cast<char*>(eta_f.data()), nx * ny * sizeof(float));
    }
    
    fout.close();
}

int main(int argc, char* argv[]) {
    Sim_Configuration config(argc, argv);
    simulate(config);
}