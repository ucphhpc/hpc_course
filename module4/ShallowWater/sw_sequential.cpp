#include <vector>
#include <iostream>
#include <H5Cpp.h>
#include <chrono>
#include <cmath>
#include <numeric>
#include <argparse.hpp>
#include <omp.h>
#include <cassert>

constexpr auto dt = 0.05; // seconds per sample
constexpr auto g = 9.80665;  // gravitational acceleration
constexpr auto dx = 1;
constexpr auto dy = 1;

void print(std::vector<double> &square_matrix, bool show_stat=true) {
    if (std::sqrt(square_matrix.size()) * std::sqrt(square_matrix.size()) != square_matrix.size()) {
        throw std::invalid_argument("print() - the matrix must be a square. The vector size is " + std::to_string(square_matrix.size()));
    }
    uint64_t size = std::sqrt(square_matrix.size());
    for (uint64_t i=0; i<size; ++i) {
        for (uint64_t j=0; j<size; ++j) {
            if (!std::signbit(square_matrix[i*size + j])) {
                std::cout << " ";
            }
            std::cout << std::scientific << square_matrix[i*size + j] << " ";
        }
        std::cout << "\n";
    }
    if (show_stat){
        std::cout << "[shape: (" << size << ", " << size <<  ")"
                  << ", min: " << *std::min_element(square_matrix.begin(), square_matrix.end())
                  << ", max: " << *std::max_element(square_matrix.begin(), square_matrix.end())
                  << ", avg: " << std::accumulate(square_matrix.begin(), square_matrix.end(), 0.0) / square_matrix.size()
                  << ", checksum: " << std::accumulate(square_matrix.begin(), square_matrix.end(), 0.0)
                  << "]\n";
    }
    std::cout << std::endl;
}

std::vector<double> extract_center(const std::vector<double> &square_matrix) {
    if (std::sqrt(square_matrix.size()) * std::sqrt(square_matrix.size()) != square_matrix.size()) {
        throw std::invalid_argument("extract_center() - the matrix must be a square. The vector size is " + std::to_string(square_matrix.size()));
    }
    uint64_t size = std::sqrt(square_matrix.size());
    std::vector<double> ret;
    for(uint64_t i=1; i<size-1; ++i) {
        for(uint64_t j=1; j<size-1; ++j) {
            ret.push_back(square_matrix[i*size + j]);
        }
    }
    return ret;
}

class Water {
public:
    uint64_t size;
    std::vector<double> u;
    std::vector<double> v;
    std::vector<double> eta;
};

Water createWater(uint64_t size) {
    Water water;
    water.size = size;
    water.u = std::vector<double>(size * size, 0);
    water.v = std::vector<double>(size * size, 0);
    water.eta = std::vector<double>(size * size, -100000);

    for (uint64_t i = 1; i < water.size-1; ++i) {
        for (uint64_t j = 1; j < water.size-1; ++j) {
            uint64_t ii = i - (size-2) / 2;
            uint64_t jj = j - (size-2) / 2;
            water.eta[i * water.size + j] = std::exp(-0.02 * (ii * ii + jj * jj));
        }
    }
    return water;
}


/** One integration step
 *
 * @param water      The water to update.
 */
void integrate(Water &w) {
    const uint64_t t_ghost = 0;
    const uint64_t b_ghost = w.size - 1;
    const uint64_t t_water = 1;
    const uint64_t b_water = w.size - 2;

    for (uint64_t i = t_water; i <= b_water; ++i) {
        w.u[i * w.size + t_ghost] = w.u[i * w.size + b_water];
        w.v[i * w.size + t_ghost] = w.v[i * w.size + b_water];
        w.eta[i * w.size + t_ghost] = w.eta[i * w.size + b_water];

        w.u[i * w.size + b_ghost] = w.u[i * w.size + t_water];
        w.v[i * w.size + b_ghost] = w.v[i * w.size + t_water];
        w.eta[i * w.size + b_ghost] = w.eta[i * w.size + t_water];
    }

    for (uint64_t j = t_water; j <= b_water; ++j) {
        w.u[t_ghost * w.size + j] = w.u[b_water * w.size + j];
        w.v[t_ghost * w.size + j] = w.v[b_water * w.size + j];
        w.eta[t_ghost * w.size + j] = w.eta[b_water * w.size + j];

        w.u[b_ghost * w.size + j] = w.u[t_water * w.size + j];
        w.v[b_ghost * w.size + j] = w.v[t_water * w.size + j];
        w.eta[b_ghost * w.size + j] = w.eta[t_water * w.size + j];
    }

    for (uint64_t i = t_water; i <= b_water; ++i) {
        for (uint64_t j = t_water; j < b_water; ++j) {
            w.u[i * w.size + j] = w.u[i * w.size + j] - dt * g * (w.eta[i * w.size + j + 1] - w.eta[i * w.size + j]) / dx;
            w.v[i * w.size + j] = w.v[i * w.size + j] - dt * g * (w.eta[(i + 1) * w.size + j] - w.eta[i * w.size + j]) / dy;
        }
    }

    for (uint64_t i = t_water; i <= b_water; ++i) {
        for (uint64_t j = t_water; j < b_water; ++j) {
            w.eta[i * w.size + j] = w.eta[i * w.size + j] - dt * (w.u[i * w.size + j] - w.u[i * w.size + j - 1]) / dx - dt * (w.v[i * w.size + j] - w.v[(i - 1) * w.size + j]) / dy;
        }
    }
}


/** Write data to a hdf5 file
 *
 * @param group  The hdf5 group to write in
 * @param name   The name of the data
 * @param shape  The shape of the data
 * @param data   The data
 */
void write_hdf5(H5::Group &group, const std::string &name, const std::vector <hsize_t> &shape,
                const std::vector<double> &data) {
    H5::DataSpace dataspace(static_cast<int>(shape.size()), &shape[0]);
    H5::DataSet dataset = group.createDataSet(name.c_str(), H5::PredType::NATIVE_DOUBLE, dataspace);
    dataset.write(&data[0], H5::PredType::NATIVE_DOUBLE);
}

/** Write a history of the water heights to a HDF5 file
 *
 * @param water_history  Vector of the all water worlds to write
 * @param filename       The output filename of the HDF5 file
 */
void write_hdf5(const std::vector<std::vector<double> > &square_matrix_history, const std::string &filename) {
    H5::H5File file(filename, H5F_ACC_TRUNC);
    for (uint64_t i = 0; i < square_matrix_history.size(); ++i) {
        H5::Group group(file.createGroup("/" + std::to_string(i)));
        if (std::sqrt(square_matrix_history[i].size()) * std::sqrt(square_matrix_history[i].size()) != square_matrix_history[i].size()) {
            throw std::invalid_argument("write_hdf5() - the square_matrix matrices must be squares");
        }
        uint64_t size = std::sqrt(square_matrix_history[i].size());
        write_hdf5(group, "water", {size, size}, square_matrix_history[i]);
    }
}



/** Simulation of a flat word climate
 *
 * @param num_of_iterations  Number of time steps to simulate
 * @param size               Size of the water world including the ghost lines
 * @param output_filename    The filename of the written water world history (HDF5 file)
 */
void simulate(uint64_t num_of_iterations, uint64_t size, const std::string &output_filename) {
    Water water_world = createWater(size);
    std::vector<std::vector<double> > water_history;
    uint64_t checksum = 0;
    auto begin = std::chrono::steady_clock::now();
    for (uint64_t t = 0; t < num_of_iterations; ++t) {
        integrate(water_world);
        if (!output_filename.empty()) {
            std::vector<double> eta = extract_center(water_world.eta);
            water_history.push_back(eta);
            std::cout << t << " -- min: " << *std::min_element(eta.begin(), eta.end())
                      << ", max: " << *std::max_element(eta.begin(), eta.end())
                      << ", avg: " << std::accumulate(eta.begin(), eta.end(), 0.0) / eta.size()
                      << "\n";
            checksum += std::accumulate(eta.begin(), eta.end(), 0.0);
        }
    }
    if (!output_filename.empty()) {
        write_hdf5(water_history, output_filename);
    }
    auto end = std::chrono::steady_clock::now();
    if (!output_filename.empty()) {
        std::cout << "checksum: " << checksum << std::endl;
    }
    std::cout << "elapsed time: " << (end - begin).count() / 1000000000.0 << " sec" << std::endl;
}

/** Main function that parses the command line and start the simulation */
int main(int argc, char **argv) {
    util::ArgParser args(argc, argv);
    int64_t size, iterations;
    if (args.cmdOptionExists("--iter")) {
        iterations = std::stoi(args.getCmdOption("--iter"));
        if (iterations < 0) {
            throw std::invalid_argument("iter most be a positive integer (e.g. --iter 100)");
        }
    } else {
        throw std::invalid_argument("You must specify the number of iterations (e.g. --iter 100)");
    }
    if (args.cmdOptionExists("--size")) {
        size = std::stoi(args.getCmdOption("--size"));
        if (size < 0) {
            throw std::invalid_argument("size most be a positive integer (e.g. --size 100)");

        }
    } else {
        throw std::invalid_argument("You must specify the size of the water, which is assumed squired e.g. " \
                                    "--size 100 is a 100 by 100 water world)");
    }
    const std::string &output_filename = args.getCmdOption("--out");

    simulate(iterations, size+2, output_filename);
    return 0;
}
