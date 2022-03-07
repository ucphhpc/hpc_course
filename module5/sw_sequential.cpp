#include <vector>
#include <iostream>
#include <chrono>
#include <cmath>
#include <numeric>
#include "argparse.hpp"
#include <cassert>
#include <fstream>
#include <array>

typedef float real_t;

constexpr size_t NX = 102, NY = 102; //World Size

constexpr real_t dt = 0.05;   // Size of the integration time step
constexpr real_t g = 9.80665; // Gravitational acceleration
constexpr real_t dx = 1.0;      // Integration step size in the horizontal direction
constexpr real_t dy = 1.0;      // Integration step size in the horizontal direction

/** Representation of a water world including ghost lines, which is a "1-cell padding" of rows and columns
 *  around the world. These ghost lines is a technique to implement periodic boundary conditions. */
class Water {
public:

    std::array<std::array<real_t, NX>, NY> u; // The speed in the horizontal direction.
    std::array<std::array<real_t, NX>, NY> v; // The speed in the vertical direction.
    std::array<std::array<real_t, NX>, NY> e; // The water elevation.
    Water() {
        for (size_t i = 0; i < NX*NY; i++) ((real_t*)e.data())[i] = -100000;
        for (size_t i = 1; i < NY - 1; ++i) 
        for (size_t j = 1; j < NX - 1; ++j) {
            real_t ii = 100.0 * (i - (NY - 2.0) / 2.0) / NY;
            real_t jj = 100.0 * (j - (NX - 2.0) / 2.0) / NX ;
            e[i][j] = std::exp(-0.02 * (ii * ii + jj * jj));
        }
    }
};

/** Help function to write the `data` to screen.
 *
 * @param data       The data to print.
 * @param shape      The shape of `data`.
 * @param show_stat  Whether to print additional statistics.
 */
void print(std::array<std::array<real_t, NX>, NY>& array, const Water& w, bool show_stat = true) {
    std::vector<real_t> data(array.begin()[0].end(), array.end()[0].end());
    for (uint64_t i = 0; i < NY; ++i) {
        for (uint64_t j = 0; j < NX; ++j) {
            if (!std::signbit(data[i * NX + j])) {
                std::cout << " ";
            }
            std::cout << std::scientific << data[i * NX + j] << " ";
        }
        std::cout << "\n";
    }
    if (show_stat) {
        std::cout << "[shape: (" << NY << ", " << NX << ")"
                  << ", min: " << *std::min_element(data.begin(), data.end())
                  << ", max: " << *std::max_element(data.begin(), data.end())
                  << ", avg: " << std::accumulate(data.begin(), data.end(), 0.0) / data.size()
                  << ", checksum: " << std::accumulate(data.begin(), data.end(), 0.0)
                  << "]\n";
    }
    std::cout << std::endl;
}

/* Write a history of the water heights to an ASCII file
 *
 * @param water_history  Vector of the all water worlds to write
 * @param filename       The output filename of the ASCII file
*/
void to_file(const std::vector <std::array<std::array<real_t, NX>, NY>> &water_history, const std::string &filename){
    std::ofstream file(filename);

    for (uint64_t l = 0; l < water_history.size(); ++l) 
    for (uint64_t k = 1; k < NY - 1; ++k)
    for (uint64_t m = 1; m < NX - 1; ++m) {
        file << std::scientific << water_history[l][k][m] << " ";
    }
}

/** Exchange the horizontal ghost lines i.e. copy the second data row to the very last data row and vice versa.
 *
 * @param data   The data update, which could be the water elevation `e` or the speed in the horizontal direction `u`.
 * @param shape  The shape of data including the ghost lines.
 */
void exchange_horizontal_ghost_lines(std::array<std::array<real_t, NX>, NY>& data) {
    for (uint64_t j = 0; j < NX; ++j) {
        data[0][j]      = data[NY-2][j]; 
        data[NY-1][j]   = data[1][j];
    }
}

/** Exchange the vertical ghost lines i.e. copy the second data column to the rightmost data column and vice versa.
 *
 * @param data   The data update, which could be the water elevation `e` or the speed in the vertical direction `v`.
 * @param shape  The shape of data including the ghost lines.
 */
void exchange_vertical_ghost_lines(std::array<std::array<real_t, NX>, NY>& data) {
    for (uint64_t i = 0; i < NY; ++i) {
        data[i][0] = data[i][NX-2];
        data[i][NX-1] = data[i][1];
    }
}

/** One integration step
 *
 * @param w The water world to update.
 */
void integrate(Water &w) {
    exchange_horizontal_ghost_lines(w.e);
    exchange_horizontal_ghost_lines(w.v);
    exchange_vertical_ghost_lines(w.e);
    exchange_vertical_ghost_lines(w.u);

    for (uint64_t i = 1; i < NY - 1; ++i) 
    for (uint64_t j = 1; j < NX - 1; ++j) {
        w.u[i][j] = w.u[i][j] - dt * g * (w.e[i][j+1] - w.e[i][j]) / dx;
        w.v[i][j] = w.v[i][j] - dt * g * (w.e[i + 1][j] - w.e[i][j]) / dy;
    }

    for (uint64_t i = 1; i < NY - 1; ++i) 
    for (uint64_t j = 1; j < NX - 1; ++j) {
        w.e[i][j] = w.e[i][j] - dt * (w.u[i][j] - w.u[i][j-1]) / dx -
                                dt * (w.v[i][j] - w.v[i-1][j]) / dy;
    }
}

/** Simulation of shallow water
 *
 * @param num_of_iterations  The number of time steps to simulate
 * @param size               The size of the water world excluding ghost lines
 * @param output_filename    The filename of the written water world history (HDF5 file)
 */
void simulate(uint64_t num_of_iterations, const std::string &output_filename) {
    Water water_world = Water();

    std::vector <std::array<std::array<real_t, NX>, NY>> water_history;
    double checksum = 0;

    auto begin = std::chrono::steady_clock::now();

    for (uint64_t t = 0; t < num_of_iterations; ++t) {
        integrate(water_world);
        /** If you want to check the output: **/
        if (!output_filename.empty()) {
            water_history.push_back(water_world.e);
        }
    }
    auto end = std::chrono::steady_clock::now();

    /** If you want to check the output: **/
    if (!output_filename.empty()) to_file(water_history, output_filename);

    for (size_t i = 0; i < NX*NY; i++) checksum += ((real_t*)water_world.e.data())[i];
    std::cout << "checksum: " << checksum << std::endl;
    std::cout << "elapsed time: " << (end - begin).count() / 1000000000.0 << " sec" << std::endl;
}

/** Main function that parses the command line and start the simulation */
int main(int argc, char **argv) {
    util::ArgParser args(argc, argv);
    int64_t iterations;
    if (args.cmdOptionExists("--iter")) {
        iterations = std::stoi(args.getCmdOption("--iter"));
        if (iterations < 0) {
            throw std::invalid_argument("iter most be a positive integer (e.g. --iter 100)");
        }
    } else {
        throw std::invalid_argument("You must specify the number of iterations (e.g. --iter 100)");
    }

    const std::string &output_filename = args.getCmdOption("--out");

    simulate(iterations, output_filename);
    return 0;
}
