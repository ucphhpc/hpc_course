#include <vector>
#include <iostream>
#include <cmath>
#include <mpi.h>

int mpi_size; // number of processes
int mpi_rank; // rank of the process

// Exchange ghost cells between neighbouring MPI ranks in both x- and y-directions.
// For ranks that have no neighbour in a given direction, the ghost cells are 
// filled by periodic wrap-around of the interior data.
void exchange_ghost_cells(int nx, int ny, int offset_x, int offset_y, int mpi_size_x, int mpi_size_y,
                          std::vector<double> &xcoord, std::vector<double> &ycoord) {

    // FIXME Find out how are the neighbor ranks to left, rioght, top, bottom
    if (mpi_size_x > 1) {
        std::cerr << "Error: mpi_size_x > 1 not implemented yet." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        // int left = ...;
        // int right = ...;
    }

    if (mpi_size_y > 1) {
        std::cerr << "Error: mpi_size_y > 1 not implemented yet." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        // int top = ...;
        // int bottom = ...;
    }

    // --- x direction ---
    if (mpi_size_x == 1) {
        for (int j = 0; j < ny; ++j) {
            // Left ghost = last interior column
            xcoord[j * nx +      0] = xcoord[j * nx + nx - 2];
            ycoord[j * nx +      0] = ycoord[j * nx + nx - 2];
            // Right ghost = first interior column
            xcoord[j * nx + nx - 1] = xcoord[j * nx +      1];
            ycoord[j * nx + nx - 1] = ycoord[j * nx +      1];
        }
    } else {
        // FIXME Implement MPI communication to exchange ghost columns with left and right neighbours
    }

    // --- y direction ---
    if (mpi_size_y == 1) {
        for (int i = 0; i < nx; ++i) {
            // Bottom ghost = last interior row
            xcoord[      0  * nx + i] = xcoord[(ny - 2) * nx + i];
            ycoord[      0  * nx + i] = ycoord[(ny - 2) * nx + i];
            // Top ghost = first interior row
            xcoord[(ny - 1) * nx + i] = xcoord[      1  * nx + i];
            ycoord[(ny - 1) * nx + i] = ycoord[      1  * nx + i];
        }
    } else {
        // FIXME Implement MPI communication to exchange ghost rows with top and bottom neighbours
    }
}

// Verify that the ghost zones contain the correct coordinate values
// Do this in a relative sense. Check that compared to the interior values the ghost cells have the
// correct values. There are extra checks for when a boundary may wrap around the global domain.
int verify_ghost_cells(int verbose, int nx, int ny, int global_nx, int global_ny, std::vector<double> &xcoord, std::vector<double> &ycoord) {

    int errors_left = 0, errors_right = 0, errors_top = 0, errors_bottom = 0;
    int errors = 0;

    // --- Bottom and top ghost row (row 0 and ny-1) ---
    // xcoord must be identical between ghost row and adjacent interior row (same column).
    // ycoord of the ghost must be exactly one step outside the interior, periodically:
    //   bottom ghost: ycoord[0] = ycoord[1] - 1  =>  ycoord[1] - ycoord[0] == 1
    //                 or wrap:                        ycoord[1] - ycoord[0] == -(global_ny - 1)
    //   top ghost:    ycoord[ny-1] = ycoord[ny-2] + 1  =>  ycoord[ny-1] - ycoord[ny-2] == 1
    //                 or wrap:                             ycoord[ny-1] - ycoord[ny-2] == -(global_ny - 1)
    for (int i = 1; i < nx - 1; ++i) {
        if (xcoord[0 * nx + i] != xcoord[1 * nx + i]) {
            errors_bottom++;
            if (verbose >= 2) std::cout << "Error in bottom ghost row, col " << i
                << ": xcoord[0, " << i << "] = " << xcoord[0 * nx + i]
                << ", expected " << xcoord[1 * nx + i] << std::endl;
        }
        if (xcoord[(ny-1) * nx + i] != xcoord[(ny-2) * nx + i]) {
            errors_top++;
            if (verbose >= 2) std::cout << "Error in top ghost row, col " << i
                << ": xcoord[" << ny-1 << ", " << i << "] = " << xcoord[(ny-1) * nx + i]
                << ", expected " << xcoord[(ny-2) * nx + i] << std::endl;
        }
        if ((ycoord[1 * nx + i] - ycoord[0 * nx + i] != 1) &&
            (ycoord[1 * nx + i] - ycoord[0 * nx + i] != -(global_ny - 1))) {
            errors_bottom++;
            if (verbose >= 2) std::cout << "Error in bottom ghost row, col " << i
                << ": ycoord[0, " << i << "] = " << ycoord[0 * nx + i]
                << ", expected " << ycoord[1 * nx + i] - 1
                << " or " << ycoord[1 * nx + i] - 1 + global_ny << std::endl;
        }
        if ((ycoord[(ny-1) * nx + i] - ycoord[(ny-2) * nx + i] != 1) &&
            (ycoord[(ny-1) * nx + i] - ycoord[(ny-2) * nx + i] != -(global_ny - 1))) {
            errors_top++;
            if (verbose >= 2) std::cout << "Error in top ghost row, col " << i
                << ": ycoord[" << ny-1 << ", " << i << "] = " << ycoord[(ny-1) * nx + i]
                << ", expected " << ycoord[(ny-2) * nx + i] + 1
                << " or " << ycoord[(ny-2) * nx + i] + 1 - global_ny << std::endl;
        }
    }

    // --- Left and right ghost column (col 0 and nx-1) ---
    // xcoord of the ghost must be exactly one step outside the interior, periodically:
    // ycoord must be identical between ghost column and adjacent interior column (same row).
    //   left ghost:  xcoord[0] = xcoord[1] - 1  =>  xcoord[1] - xcoord[0] == 1
    //                or wrap:                        xcoord[1] - xcoord[0] == -(global_nx - 1)
    //   right ghost: xcoord[nx-1] = xcoord[nx-2] + 1  =>  xcoord[nx-1] - xcoord[nx-2] == 1
    //                or wrap:                             xcoord[nx-1] - xcoord[nx-2] == -(global_nx - 1)
    for (int j = 1; j < ny - 1; ++j) {
        if ((xcoord[j * nx + 1] - xcoord[j * nx + 0] != 1) &&
            (xcoord[j * nx + 1] - xcoord[j * nx + 0] != -(global_nx - 1))) {
            errors_left++;
            if (verbose >= 2) std::cout << "Error in left ghost col, row " << j
                << ": xcoord[" << j << ", 0] = " << xcoord[j * nx + 0]
                << ", expected " << xcoord[j * nx + 1] - 1
                << " or " << xcoord[j * nx + 1] - 1 + global_nx << std::endl;
        }
        if ((xcoord[j * nx + (nx-1)] - xcoord[j * nx + (nx-2)] != 1) &&
            (xcoord[j * nx + (nx-1)] - xcoord[j * nx + (nx-2)] != -(global_nx - 1))) {
            errors_right++;
            if (verbose >= 2) std::cout << "Error in right ghost col, row " << j
                << ": xcoord[" << j << ", " << nx-1 << "] = " << xcoord[j * nx + (nx-1)]
                << ", expected " << xcoord[j * nx + (nx-2)] + 1
                << " or " << xcoord[j * nx + (nx-2)] + 1 - global_nx << std::endl;
        }
        if (ycoord[j * nx + 0] != ycoord[j * nx + 1]) {
            errors_left++;
            if (verbose >= 2) std::cout << "Error in left ghost col, row " << j
                << ": ycoord[" << j << ", 0] = " << ycoord[j * nx + 0]
                << ", expected " << ycoord[j * nx + 1] << std::endl;
        }
        if (ycoord[j * nx + (nx-1)] != ycoord[j * nx + (nx-2)]) {
            errors_right++;
            if (verbose >= 2) std::cout << "Error in right ghost col, row " << j
                << ": ycoord[" << j << ", " << nx-1 << "] = " << ycoord[j * nx + (nx-1)]
                << ", expected " << ycoord[j * nx + (nx-2)] << std::endl;
        }
    }

    errors = errors_left + errors_right + errors_top + errors_bottom;
    if (verbose>=1) {
        std::cout << "errors_left: " << errors_left
                  << ", errors_right: " << errors_right
                  << ", errors_top: " << errors_top
                  << ", errors_bottom: " << errors_bottom
                  << std::endl;
    }
    return errors;
}

void simulate(int verbose, int global_nx, int global_ny) {
    /** Representation of a 2D domain, including ghost zones.
     *
     * The global grid has dimensions global_nx x global_ny.
     * Each rank owns a contiguous block .
     * One layer of ghost cells surrounds the local data on all sides.
     *
     * Memory layout (row-major, including ghost zones):
     *
     *   row 0        : ghost zone (bottom neighbour)
     *   row 1 .. ny-2: interior (owned) data
     *   row ny-1     : ghost zone (top neighbour)
     *
     *   col 0        : ghost zone (left neighbour)
     *   col 1 .. nx-2: interior (owned) data
     *   col nx-1     : ghost zone (right neighbour)
     *
     * If there is only one rank in a given direction, the ghost zones
     * in that direction are filled by periodic wrap-around of the interior data.
     */


    // MPI rank geometry. How many ranks in x-direction and in y-direction?
    // FIXME 2D check for divisors or something to make a reasonable rank geometry.
    // FIXME 2D For now, just do a 1D decomposition in y, so that we can focus on the MPI communication part.
    int mpi_size_x = 1;        // no decomposition in x
    int mpi_size_y = mpi_size; // all ranks split the y dimension

    // FIXME compute local domain size. Here we assume that global_n[xy] is divisible by mpi_size_[xy]
    // FIXME so that each rank gets an integer number of rows and columns.
    int interior_nx = global_nx / mpi_size_x;
    int interior_ny = global_ny / mpi_size_y;

    // FIXME This only works if mpi_ == 1
    // -1 because the first cell is a ghost cell
    const int offset_x = -1;
    const int offset_y =- 1;

    // One ghost cell on each end in both dimensions
    const int nx = interior_nx + 2;
    const int ny = interior_ny + 2;

    // x- and y-coordinate values at each grid cell (including ghost zones).
    // NB: it is up to the calculation to interpret these vectors as two dimensions.
    std::vector<double> xcoord(nx * ny), ycoord(nx * ny);

    std::cout << "rank " << mpi_rank << " owns rows ["
              << offset_y + 1 << ", " << offset_y + ny - 2
              << "] (interior_ny=" << interior_ny << ")" << std::endl;
    std::cout << "rank " << mpi_rank << " owns cols ["
              << offset_x + 1 << ", " << offset_x + nx - 2
              << "] (interior_nx=" << interior_nx << ")" << std::endl;

    // Initialise interior cells with global coordinate values
    for (int j = 1; j < ny - 1; ++j) {
        for (int i = 1; i < nx - 1; ++i) {
            xcoord[j * nx + i] = i + offset_x;
            ycoord[j * nx + i] = j + offset_y;
        }
    }

    // Exchange ghost cells between neighbouring ranks
    exchange_ghost_cells(nx, ny, offset_x, offset_y, mpi_size_x, mpi_size_y, xcoord, ycoord);

    // Verify that ghost zones contain the correct coordinate values
    int local_errors  = verify_ghost_cells(verbose, nx, ny, global_nx, global_ny, xcoord, ycoord);
    int global_errors = 0;
    MPI_Reduce(&local_errors, &global_errors, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (mpi_rank == 0) {
        if (global_errors == 0) {
            std::cout << "Ghost cell exchange: OK (all values correct)" << std::endl;
        } else {
            std::cout << "Ghost cell exchange: FAILED (" << global_errors << " errors)" << std::endl;
        }
    }
}

/** Main function that parses the command line and starts the domain decomposition exercise */
int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);                   // Initialize the MPI environment
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size); // Get the number of processes
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank); // Get the rank of the process

    // Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    // Print off a hello world message
    std::cout << "Domain Decomposition running on " << processor_name
              << ", rank " << mpi_rank << " out of " << mpi_size << std::endl;

    int global_nx = 0, global_ny = 0, verbose = 0;

    std::vector<std::string> argument({argv, argv + argc});

    for (int i = 1; i < (int)argument.size(); i += 2) {
        std::string arg = argument[i];
        if (arg == "-h") { // Write help
            std::cout << "./domain_decomposition --verbose <verbosity> --nx <global nx> --ny <global ny>\n";
            MPI_Finalize();
            return 0;
        } else if (i == (int)argument.size() - 1) {
            throw std::invalid_argument("The last argument (" + arg + ") must have a value");
        } else if (arg == "--verbose") {
            if ((verbose = std::stoi(argument[i + 1])) < 0)
                throw std::invalid_argument("verbose must be non-negative (e.g. --verbose 1)");
        } else if (arg == "--nx") {
            if ((global_nx = std::stoi(argument[i + 1])) < 1)
                throw std::invalid_argument("nx must be positive (e.g. --nx 32)");
        } else if (arg == "--ny") {
            if ((global_ny = std::stoi(argument[i + 1])) < 1)
                throw std::invalid_argument("ny must be positive (e.g. --ny 16)");
        } else {
            std::cout << "---> error: the argument type is not recognized \n";
        }
    }

    if (global_nx == 0 || global_ny == 0) {
        std::cout << "global_nx: " << global_nx << ", global_ny: " << global_ny << std::endl;
        throw std::invalid_argument("You must specify the global grid size (e.g. --nx 32 --ny 16)");
    }

    simulate(verbose, global_nx, global_ny);

    MPI_Finalize();

    return 0;
}