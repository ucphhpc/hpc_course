#include <vector>
#include <array>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <random>
#include <H5Cpp.h>
#include <chrono>
#include <argparse.hpp>
#include <bhxx/bhxx.hpp>

const double G = 6.673e-11;
const double SOLAR_MASS = 1.98892e30;

/** Class that represent a set of bodies, which can be the sun, planets, and/or asteroids */
class Bodies {
public:
    // The mass of the bodies
    bhxx::BhArray<double> mass;
    // The position of the bodies in the x, y, and z axis
    bhxx::BhArray<double> pos_x;
    bhxx::BhArray<double> pos_y;
    bhxx::BhArray<double> pos_z;
    // The velocity of the bodies in the x, y, and z axis
    bhxx::BhArray<double> vel_x;
    bhxx::BhArray<double> vel_y;
    bhxx::BhArray<double> vel_z;

    /** Return a copy of all bodies */
    Bodies copy() {
        return Bodies{mass.copy(), pos_x.copy(), pos_y.copy(), pos_z.copy(), vel_x.copy(), vel_y.copy(), vel_z.copy()};
    }

};

/** Class that represent a solar system, which consist of a sun, some planets, and many asteroids. */
class SolarSystem {
public:
    Bodies sun_and_planets;
    Bodies asteroids;

    /** Return a copy of the solar system */
    SolarSystem copy() {
       return SolarSystem{sun_and_planets.copy(), asteroids.copy()};
    }
};

/** Function that returns `x` squared for each element in `x` */
bhxx::BhArray<double> squared(const bhxx::BhArray<double> &x) {
    return x * x;
}

/** Function that returns the magnitude of velocity for each element `i` in `pos_x[i], pos_y[i], pos_z[i]` */
const bhxx::BhArray<double> circlev(const bhxx::BhArray<double> &pos_x,
                                    const bhxx::BhArray<double> &pos_y,
                                    const bhxx::BhArray<double> &pos_z) {
    return bhxx::sqrt(G * 1e6 * SOLAR_MASS / bhxx::sqrt(squared(pos_x) + squared(pos_y) + squared(pos_z)));
}

/** Return a new random solar system
 *
 * @param position_limit   The coordinate limit
 * @param num_of_planets   The number of planets
 * @param num_of_asteroids The number of asteroids
 * @param seed             The random seed the use
 * @return                 The new solar system
 */
SolarSystem random_system(double position_limit, uint64_t num_of_planets, uint64_t num_of_asteroids, uint64_t seed) {
    SolarSystem solar_system;
    // TODO: Implement
    return solar_system;
}

/** Fill the diagonal of `ary` with the value `val` */
void fill_diagonal(bhxx::BhArray<double> &ary, double val) {
    if (ary.shape().size() != 2 || ary.shape()[0] == ary.shape()[1]) {
        throw std::runtime_error("The array must be a squired matrix");
    }
    if (!ary.isContiguous()) {
        throw std::runtime_error("`ary` must be contiguous");
    }
    uint64_t view_size = ary.shape()[0];
    int64_t view_stride = view_size+1;
    // Create a view that represents the diagonal of `ary`
    bhxx::BhArray<double> diagonal_view(ary.base(), {view_size}, {view_stride}, ary.offset());
    // And assign `val` to the diagonal
    diagonal_view = val;
}

/** Update the velocity of all bodies in `a` based on the bodies in `b`
 *
 * @param a                   The bodies to update
 * @param b                   The bodies which act on `a`
 * @param diagonal_zero_fill  Set the diagonal of the force matrices to zero
 * @param dt                  The time step size
 */
void update_velocity(Bodies &a, const Bodies &b, double dt, bool diagonal_zero_fill) {
    // TODO: Implement
}

/** Integrate one time step of the solar system
 *
 * @param solar_system  The solar system to update
 * @param dt            The time step size
 */
void integrate(SolarSystem &solar_system, double dt) {
    // TODO: Implement
}

/** Write data to a hdf5 file
 *
 * @param group  The hdf5 group to write in
 * @param name   The name of the data
 * @param shape  The shape of the data
 * @param data   The data
 */
void write_hdf5(H5::Group &group, const std::string &name, const std::vector<hsize_t> &shape,
                const std::vector<double> &data) {

    H5::DataSpace dataspace(static_cast<int>(shape.size()), &shape[0]);
    H5::DataSet dataset = group.createDataSet(name.c_str(), H5::PredType::NATIVE_DOUBLE, dataspace);
    dataset.write(&data[0], H5::PredType::NATIVE_DOUBLE);
}

/** Write the solar system to a hdf5 file (use `visual.py` to visualize the hdf5 data)
 *
 * @param solar_systems  The solar system to write
 * @param filename       The filename to write to
 */
void write_hdf5(const std::vector<SolarSystem> &solar_systems, const std::string &filename) {

    H5::H5File file(filename, H5F_ACC_TRUNC);

    for (uint64_t i = 0; i < solar_systems.size(); ++i) {
        H5::Group group(file.createGroup("/" + std::to_string(i)));
        {
            std::vector<double> data;
            for (double elem: solar_systems[i].sun_and_planets.pos_x.vec()) {
                data.push_back(elem);
            }
            for (double elem: solar_systems[i].sun_and_planets.pos_y.vec()) {
                data.push_back(elem);
            }
            for (double elem: solar_systems[i].sun_and_planets.pos_z.vec()) {
                data.push_back(elem);
            }
            write_hdf5(group, "sun_and_planets_position", {3, solar_systems[i].sun_and_planets.pos_x.size()}, data);
        }
        {
            std::vector<double> data;
            for (double elem: solar_systems[i].asteroids.pos_x.vec()) {
                data.push_back(elem);
            }
            for (double elem: solar_systems[i].asteroids.pos_y.vec()) {
                data.push_back(elem);
            }
            for (double elem: solar_systems[i].asteroids.pos_z.vec()) {
                data.push_back(elem);
            }
            write_hdf5(group, "asteroids_position", {3, solar_systems[i].asteroids.pos_x.size()}, data);
        }
    }

}

/** N-body NICE simulation
 *
 * @param num_of_iterations  Number of iterations
 * @param num_of_planets     Number of planets
 * @param num_of_asteroids   Number of asteroids
 * @param seed               Random seed
 * @param filename           Filename to write each time step to. If empty, do data is written.
 */
void simulate(uint64_t num_of_iterations, uint64_t num_of_planets, uint64_t num_of_asteroids, uint64_t seed,
              const std::string &filename) {
    double dt = 1e12;
    double position_limit = 1e18;
    SolarSystem system = random_system(position_limit, num_of_planets, num_of_asteroids, seed);
    std::vector<SolarSystem> systems;
    systems.push_back(system);

    bhxx::flush();
    auto begin = std::chrono::steady_clock::now();
    for (uint64_t i = 0; i < num_of_iterations; ++i) {
        integrate(system, dt);
        if (!filename.empty()) {
            systems.push_back(system.copy());
        }
        bhxx::flush();
    }
    if (!filename.empty()) {
        write_hdf5(systems, filename);
    }
    auto end = std::chrono::steady_clock::now();
    std::cout << "elapsed time: " << (end - begin).count() / 1000000000.0 << " sec"<< std::endl;
}

/** Main function that parses the command line and start the simulation */
int main(int argc, char **argv) {
    util::ArgParser args(argc, argv);
    int64_t iterations, num_of_planets, num_of_asteroids, seed;
    if (args.cmdOptionExists("--iter")) {
        iterations = std::stoi(args.getCmdOption("--iter"));
        if (iterations < 0) {
            throw std::invalid_argument("iter most be a positive integer (e.g. --iter 100)");
        }
    } else {
        throw std::invalid_argument("You must specify the number of iterations (e.g. --iter 100)");
    }
    if (args.cmdOptionExists("--planets")) {
        num_of_planets = std::stoi(args.getCmdOption("--planets"));
        if (num_of_planets < 0) {
            throw std::invalid_argument("planets most be a positive integer (e.g. --planets 7)");
        }
    } else {
        throw std::invalid_argument("You must specify the number of planets (e.g. --planets 7)");
    }
    if (args.cmdOptionExists("--planets")) {
        num_of_planets = std::stoi(args.getCmdOption("--planets"));
        if (num_of_planets < 0) {
            throw std::invalid_argument("planets most be a positive integer (e.g. --planets 7)");
        }
    } else {
        throw std::invalid_argument("You must specify the number of planets (e.g. --planets 7)");
    }
    if (args.cmdOptionExists("--asteroids")) {
        num_of_asteroids = std::stoi(args.getCmdOption("--asteroids"));
        if (num_of_asteroids < 0) {
            throw std::invalid_argument("asteroids most be a positive integer (e.g. --asteroids 1000)");
        }
    } else {
        throw std::invalid_argument("You must specify the number of asteroids (e.g. --asteroids 1000)");
    }
    if (args.cmdOptionExists("--seed")) {
        seed = std::stoi(args.getCmdOption("--seed"));
        if (seed < 0) {
            throw std::invalid_argument("Seed most be a positive integer (e.g. --seed 42)");
        }
    } else {
        seed = std::random_device{}(); // Default seed is taken from hardware
    }
    const std::string &filename = args.getCmdOption("--out");

    simulate(static_cast<uint64_t>(iterations), static_cast<uint64_t>(num_of_planets),
             static_cast<uint64_t>(num_of_asteroids), static_cast<uint64_t>(seed), filename);
    return 0;
}
