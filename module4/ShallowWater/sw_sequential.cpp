#include <vector>
#include <iostream>
#include <H5Cpp.h>
#include <chrono>
#include <cmath>
#include <numeric>
#include <argparse.hpp>
#include <omp.h>


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
    water.eta = std::vector<double>(size * size);

    for (uint64_t i = 0; i < size; ++i) {
        for (uint64_t j = 0; j < size; ++j) {
            uint64_t ii = i - size / 2;
            uint64_t jj = j - size / 2;
            water.eta[i * size + j] = std::exp(-0.02 * (ii * ii + jj * jj));
        }
    }
    return water;
}

/** One integration step
 *
 * @param water      The water to update.
 */
void integrate(Water &water) {

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
void write_hdf5(const std::vector <Water> &water_history, const std::string &filename) {
    H5::H5File file(filename, H5F_ACC_TRUNC);

    for (uint64_t i = 0; i < water_history.size(); ++i) {
        H5::Group group(file.createGroup("/" + std::to_string(i)));
        write_hdf5(group, "water", {water_history[i].size, water_history[i].size}, water_history[i].eta);
    }
}

/** Simulation of a flat word climate
 *
 * @param num_of_iterations  Number of time steps to simulate
 * @param size               Size of the water water
 * @param output_filename    The filename of the written water world history (HDF5 file)
 */
void simulate(uint64_t num_of_iterations, uint64_t size, const std::string &output_filename) {
    Water water = createWater(size);
    std::vector <Water> water_history;
    uint64_t checksum = 0;
    auto begin = std::chrono::steady_clock::now();
    for (uint64_t t = 0; t < num_of_iterations; ++t) {
        integrate(water);
        if (!output_filename.empty()) {
            water_history.push_back(water);
            std::cout << t << " -- min: " << *std::min_element(water.eta.begin(), water.eta.end())
                      << ", max: " << *std::max_element(water.eta.begin(), water.eta.end())
                      << ", avg: " << std::accumulate(water.eta.begin(), water.eta.end(), 0.0) / water.eta.size()
                      << "\n";
            checksum += std::accumulate(water.eta.begin(), water.eta.end(), 0.0);
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

    simulate(iterations, size, output_filename);
    return 0;
}
