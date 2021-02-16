#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <H5Cpp.h>
#include <chrono>
#include <argparse.hpp>

// SI units: Newtons constant of gravity, Solar Mass, distance from Earth to Sun (AU), and JPL unit of velocity, AU/day
const double G = 6.673e-11;
const double SOLAR_MASS = 1.98892e30;
const double AU = 149597870700;
const double AD = AU / (24. * 3600.);

/** Class that represent bodies, which can be a sun, a planet, or an asteroid */
class Bodies {
public:
    // The mass of the bodies
    std::vector<double> mass;
    // The position of the bodies in the x, y, and z axis
    std::vector<double> pos_x;
    std::vector<double> pos_y;
    std::vector<double> pos_z;
    // The velocity of the bodies in the x, y, and z axis
    std::vector<double> vel_x;
    std::vector<double> vel_y;
    std::vector<double> vel_z;

    void add(double &m, double &x,  double &y,  double &z, 
                        double &vx, double &vy, double &vz) {
      mass.push_back(m);
      pos_x.push_back(x);
      pos_y.push_back(y);
      pos_z.push_back(z);
      vel_x.push_back(vx);
      vel_y.push_back(vy);
      vel_z.push_back(vz);
   }
};

/** Class that represent a solar system, which consist of a sun, some planets, and many asteroids. */
class SolarSystem {
public:
    // The first Body is the sun and the rest are planets
    Bodies sun_and_planets;
    Bodies asteroids;
};

/** Function that returns the Kepler velocity -- corresponding to a body in a circular orbit */
double kepler_velocity(const double &pos_x, const double &pos_y, const double &pos_z) {
    double r = std::sqrt(pos_x * pos_x + pos_y * pos_y + pos_z * pos_z);
    return std::sqrt(G * SOLAR_MASS / r);
}

/** Return a new random solar system
 *
 * @param num_of_asteroids The number of asteroids
 * @param seed             The random seed the use
 * @return                 The new solar system
 */
SolarSystem random_system(uint64_t num_of_asteroids, uint64_t seed) {
    SolarSystem solar_system;
    // TODO: Implement
    return solar_system;
}

/** Update the velocity of `a` based on `b`
 *
 * @param a  The body to update
 * @param b  The body which act on `a`
 * @param dt The time step size
 */
//#pragma acc routine seq
void update_velocity(Bodies &a, const Bodies &b, const int &i, const int &j, const double &dt) {
    // TODO: Implement
}

/** Kick a set of bodies forward in time due to their mutual gravitational interaction
 *
 * @param a  The bodies to update
 * @param dt The time step size
 */
void kick_same(Bodies &a, const double &dt) {
    // TODO: Implement
}

/** Kick a set of bodies forward in time due to gravitational interaction with another set of bodies
 *
 * @param a  The bodies to update
 * @param b  The bodies that perturb
 * @param dt The time step size
 */
void kick_other(Bodies &a, const Bodies &b, const double &dt) {
    // TODO: Implement
}

/** Drift a set of bodies forward in time
 *
 * @param bodies The bodies to update
 * @param dt     The time step size
 */
void drift(Bodies &bodies, const double &dt) {
    // TODO: Implement
}

/** Integrate one time step of the solar system
 *
 * @param solar_system  The solar system to update
 * @param dt            The time step size
 */
void integrate(SolarSystem &solar_system, double dt) {

    // Kick is done twice --> only half dt each time
    double const hdt = 0.5*dt;
    // First kick
    // Update velocity of all bodies 
    kick_same(solar_system.sun_and_planets, hdt);
    kick_other(solar_system.asteroids,solar_system.sun_and_planets, hdt);

    // Drift: Update position of all bodies
    drift(solar_system.sun_and_planets, dt);
    drift(solar_system.asteroids, dt);

    // Second kick
    // Update velocity of all bodies 
    kick_same(solar_system.sun_and_planets, hdt);
    kick_other(solar_system.asteroids,solar_system.sun_and_planets, hdt);
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
            const std::vector<double> &data = solar_systems[i].sun_and_planets.mass;
            uint64_t n = data.size();
            write_hdf5(group, "sun_and_planets_mass", {n}, data);
        }
        {
            std::vector<double> data;
            const Bodies &bodies = solar_systems[i].sun_and_planets;
            uint64_t n = bodies.mass.size();
            for (uint64_t i = 0; i < n; ++i) {
                data.push_back(bodies.pos_x[i]);
            }
            for (uint64_t i = 0; i < n; ++i) {
                data.push_back(bodies.pos_y[i]);
            }
            for (uint64_t i = 0; i < n; ++i) {
                data.push_back(bodies.pos_z[i]);
            }
            write_hdf5(group, "sun_and_planets_position", {3, n}, data);
        }
        {
            std::vector<double> data;
            const Bodies &bodies = solar_systems[i].asteroids;
            uint64_t n = bodies.mass.size();
            for (uint64_t i = 0; i < n; ++i) {
                data.push_back(bodies.pos_x[i]);
            }
            for (uint64_t i = 0; i < n; ++i) {
                data.push_back(bodies.pos_y[i]);
            }
            for (uint64_t i = 0; i < n; ++i) {
                data.push_back(bodies.pos_z[i]);
            }
            write_hdf5(group, "asteroids_position", {3, n}, data);
        }

    }
}

/** N-body Solar System simulation
 *
 * @param num_of_iterations  Number of iterations
 * @param num_of_asteroids   Number of asteroids
 * @param seed               Random seed
 * @param filename           Filename to write each time step to. If empty, do data is written.
 */
void simulate(uint64_t num_of_iterations, uint64_t num_of_asteroids, uint64_t seed,
              uint64_t every, const std::string &filename) {
    double dt = 1e5;
    SolarSystem system = random_system(num_of_asteroids, seed);
    std::vector<SolarSystem> systems;
    systems.push_back(system);
    auto begin = std::chrono::steady_clock::now();
    for (uint64_t i = 0; i < num_of_iterations; ++i) {
        integrate(system, dt);
        if (!filename.empty() && (i % every) == 0) {
            systems.push_back(system);
        }
    }
    if (!filename.empty()) {
        write_hdf5(systems, filename);
    }
    auto end = std::chrono::steady_clock::now();
    std::cout << "elapsed time: " << (end - begin).count() / 1000000000.0 << " sec"<< std::endl;
    std::cout << "Performance: " << (end - begin).count() / 
        (num_of_iterations * (1 + 8 + num_of_asteroids)) << " nanosecs / particle update"<< std::endl;
}

/** Main function that parses the command line and start the simulation */
int main(int argc, char **argv) {
    util::ArgParser args(argc, argv);
    int64_t iterations, every, num_of_asteroids, seed;
    if (args.cmdOptionExists("--iter")) {
        iterations = std::stoi(args.getCmdOption("--iter"));
        if (iterations < 0) {
            throw std::invalid_argument("iter most be a positive integer (e.g. --iter 100)");
        }
    } else {
        throw std::invalid_argument("You must specify the number of iterations (e.g. --iter 100)");
    }
    if (args.cmdOptionExists("--every")) {
        every = std::stoi(args.getCmdOption("--every"));
        if (every < 0) {
            throw std::invalid_argument("every most be a positive integer (e.g. --every 10)");
        }
    } else {
        every = 10;
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
    if (!filename.empty()) {
        std::cout << "Writing data to " << filename << std::endl;
        std::cout << "Writing out every " << every << "th iteration to disk (--every "<< every << ")" << std::endl;
    }

    simulate(static_cast<uint64_t>(iterations), 
             static_cast<uint64_t>(num_of_asteroids),
             static_cast<uint64_t>(seed),
             static_cast<uint64_t>(every),
             filename);
    return 0;
}
