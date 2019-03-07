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

class Shape {
public:
    uint64_t rows;
    uint64_t cols;
    uint64_t total;
    Shape(uint64_t rows, uint64_t cols) : rows(rows), cols(cols), total(rows*cols) {}
};

void print(std::vector<double> &matrix, const Shape &shape, bool show_stat=true) {
    for (uint64_t i=0; i<shape.rows; ++i) {
        for (uint64_t j=0; j<shape.cols; ++j) {
            if (!std::signbit(matrix[i*shape.cols + j])) {
                std::cout << " ";
            }
            std::cout << std::scientific << matrix[i*shape.cols + j] << " ";
        }
        std::cout << "\n";
    }
    if (show_stat){
        std::cout << "[shape: (" << shape.rows << ", " << shape.cols <<  ")"
                  << ", min: " << *std::min_element(matrix.begin(), matrix.end())
                  << ", max: " << *std::max_element(matrix.begin(), matrix.end())
                  << ", avg: " << std::accumulate(matrix.begin(), matrix.end(), 0.0) / matrix.size()
                  << ", checksum: " << std::accumulate(matrix.begin(), matrix.end(), 0.0)
                  << "]\n";
    }
    std::cout << std::endl;
}

std::vector<double> extract_center(const std::vector<double> &matrix, const Shape &shape) {
    std::vector<double> ret;
    for(uint64_t i=1; i<shape.rows-1; ++i) {
        for(uint64_t j=1; j<shape.cols-1; ++j) {
            ret.push_back(matrix[i*shape.cols + j]);
        }
    }
    return ret;
}

class Water {
public:
    Shape shape;
    std::vector<double> u;
    std::vector<double> v;
    std::vector<double> e;
    Water(Shape shape) : shape(shape), u(shape.total, 0), v(shape.total, 0), e(shape.total, -100000) {}
};

Water createWater(Shape shape) {
    Water w(shape);
    for (uint64_t i = 1; i < w.shape.rows-1; ++i) {
        for (uint64_t j = 1; j < w.shape.cols-1; ++j) {
            uint64_t ii = i - (w.shape.rows-2) / 2;
            uint64_t jj = j - (w.shape.cols-2) / 2;
            w.e[i * w.shape.cols + j] = std::exp(-0.02 * (ii * ii + jj * jj));
        }
    }
    return w;
}

void exchange_horizontal_ghost_lines(std::vector<double> &data, Shape shape) {
    for (uint64_t i = 0; i < shape.cols; ++i) {
        const uint64_t top_ghost = 0 * shape.cols + i;
        const uint64_t bot_water = (shape.rows-2) * shape.cols + i;
        const uint64_t bot_ghost = (shape.rows-1) * shape.cols + i;
        const uint64_t top_water = 1 * shape.cols + i;
        data[top_ghost] = data[bot_water];
        data[bot_ghost] = data[top_water];
    }
}

void exchange_vertical_ghost_lines(std::vector<double> &data, Shape shape) {
    for (uint64_t i = 0; i < shape.cols; ++i) {
        const uint64_t left_ghost = i * shape.cols + 0;
        const uint64_t right_water = i * shape.cols + shape.cols-2;
        const uint64_t right_ghost = i * shape.cols + shape.cols-1;
        const uint64_t left_water = i * shape.cols + 1;
        data[left_ghost] = data[right_water];
        data[right_ghost] = data[left_water];
    }
}

/** One integration step
 *
 * @param water      The water to update.
 */
void integrate(Water &w) {
    exchange_horizontal_ghost_lines(w.e, w.shape);
    exchange_horizontal_ghost_lines(w.v, w.shape);

    exchange_vertical_ghost_lines(w.e, w.shape);
    exchange_vertical_ghost_lines(w.u, w.shape);

    const uint64_t stride = w.shape.cols;
    for (uint64_t i = 1; i < w.shape.rows-1; ++i) {
        for (uint64_t j = 1; j < w.shape.cols-1; ++j) {
            w.u[i * stride + j] = w.u[i * stride + j] - dt * g * (w.e[i * stride + j + 1] - w.e[i * stride + j]) / dx;
            w.v[i * stride + j] = w.v[i * stride + j] - dt * g * (w.e[(i + 1) * stride + j] - w.e[i * stride + j]) / dy;
        }
    }

    for (uint64_t i = 1; i < w.shape.rows-1; ++i) {
        for (uint64_t j = 1; j < w.shape.cols-1; ++j) {
            w.e[i * stride + j] = w.e[i * stride + j] - dt * (w.u[i * stride + j] - w.u[i * stride + j - 1]) / dx - dt * (w.v[i * stride + j] - w.v[(i - 1) * stride + j]) / dy;
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
    Water water_world = createWater(Shape(size, size));
    std::vector<std::vector<double> > water_history;
    uint64_t checksum = 0;
    auto begin = std::chrono::steady_clock::now();
    for (uint64_t t = 0; t < num_of_iterations; ++t) {
        integrate(water_world);
        if (!output_filename.empty()) {
            std::vector<double> e = extract_center(water_world.e, Shape(size, size));
            water_history.push_back(e);
            std::cout << t << " -- min: " << *std::min_element(e.begin(), e.end())
                      << ", max: " << *std::max_element(e.begin(), e.end())
                      << ", avg: " << std::accumulate(e.begin(), e.end(), 0.0) / e.size()
                      << "\n";
            checksum += std::accumulate(e.begin(), e.end(), 0.0);
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
