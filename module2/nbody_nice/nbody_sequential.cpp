#include <vector>
#include <array>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <random>
#include <H5Cpp.h>
#include <chrono>
#include <argparse.hpp>

const double G = 6.673e-11;
const double SOLAR_MASS = 1.98892e30;

/** Class that represent a body, which can be a sun, a planet, or an asteroid */
class Body {
public:
    // The mass of the body
    double mass;
    // The position of the body in the x, y, and z axis
    double pos_x;
    double pos_y;
    double pos_z;
    // The velocity of the body in the x, y, and z axis
    double vel_x;
    double vel_y;
    double vel_z;
};

/** Class that represent a solar system, which consist of a sun, some planets, and many asteroids. */
class SolarSystem {
public:
    // The first Body is the sun and the rest are planets
    std::vector<Body> sun_and_planets;
    std::vector<Body> asteroids;
};

/** Function that returns -1 when `x` is negative, 1 when `x` is positive, and 0 when `x` is zero. */
double sign(double x) {
    if (x < 0) {
        return -1;
    } else if (x > 0) {
        return 1;
    } else {
        return 0;
    }
}

/** Function that returns `x` squared */
double squared(double x) {
    return x * x;
}

/** Function that returns the magnitude of velocity */
double circlev(double pos_x, double pos_y, double pos_z) {
    double r = std::sqrt(squared(pos_x) + squared(pos_y) + squared(pos_z));
    return std::sqrt(G * 1e6 * SOLAR_MASS / r);
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
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> random_real(0, 1);
    Body the_sun = Body{1e6 * SOLAR_MASS, 0, 0, 0, 0, 0, 0}; 
    std::vector<Body> sun_and_planets = {the_sun}; // The first body is the sun
    for (uint64_t i = 0; i < num_of_planets; ++i) {
        double pos_x = random_real(gen);
        double pos_y = random_real(gen);
        double pos_z = random_real(gen) * .01;
        double dist = (1.0 / std::sqrt(squared(pos_x) + squared(pos_y) + squared(pos_z))) - (.8 - random_real(gen) * .1);
        pos_x *= position_limit * dist * sign(.5 - random_real(gen));
        pos_y *= position_limit * dist * sign(.5 - random_real(gen));
        pos_z *= position_limit * dist * sign(.5 - random_real(gen));

        double magv = circlev(pos_x, pos_y, pos_z);
        double abs_angle = std::atan(std::abs(pos_y / pos_x));
        double theta_v = M_PI / 2 - abs_angle;
        double vel_x = -1 * sign(pos_y) * std::cos(theta_v) * magv;
        double vel_y = sign(pos_x) * std::sin(theta_v) * magv;
        double vel_z = 0;
        double mass = random_real(gen) * SOLAR_MASS * 10 + 1e20;
        sun_and_planets.push_back(Body{mass, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z});
    }

    std::vector<Body> asteroids;
    for (uint64_t i = 0; i < num_of_asteroids; ++i) {
        double pos_x = random_real(gen);
        double pos_y = random_real(gen);
        double pos_z = random_real(gen) * .01;
        double dist = (1.0 / std::sqrt(squared(pos_x) + squared(pos_y) + squared(pos_z))) - random_real(gen) * .2;
        pos_x *= position_limit * dist * sign(.5 - random_real(gen));
        pos_y *= position_limit * dist * sign(.5 - random_real(gen));
        pos_z *= position_limit * dist * sign(.5 - random_real(gen));

        double magv = circlev(pos_x, pos_y, pos_z);
        double abs_angle = std::atan(std::abs(pos_y / pos_x));
        double theta_v = M_PI / 2 - abs_angle;
        double vel_x = -1 * sign(pos_y) * std::cos(theta_v) * magv;
        double vel_y = sign(pos_x) * std::sin(theta_v) * magv;
        double vel_z = 0;
        double mass = random_real(gen) * SOLAR_MASS * 10 + 1e14;
        asteroids.push_back(Body{mass, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z});
    }

    return SolarSystem{sun_and_planets, asteroids};
}

/** Update the velocity of `a` based on `b`
 *
 * @param a  The body to update
 * @param b  The body which act on `a`
 * @param dt The time step size
 */
void update_velocity(Body &a, const Body &b, double dt) {
    // Euclidean distance
    double r = std::sqrt(squared(b.pos_x - a.pos_x) + squared(b.pos_y - a.pos_y) +
                         squared(b.pos_z - a.pos_z));

    // Force:  F = ((G m_a m_b)/r^2)*((x_b-x_a)/r)
    double F = (G * a.mass * b.mass / squared(r)); // Force without direction

    // Update velocity of `a`
    a.vel_x += F * ((b.pos_x - a.pos_x) / r) / a.mass * dt;
    a.vel_y += F * ((b.pos_y - a.pos_y) / r) / a.mass * dt;
    a.vel_z += F * ((b.pos_z - a.pos_z) / r) / a.mass * dt;
}

/** Integrate one time step of the solar system
 *
 * @param solar_system  The solar system to update
 * @param dt            The time step size
 */
void integrate(SolarSystem &solar_system, double dt) {

    // Update velocity of the sub and planets
    for (uint64_t i = 0; i < solar_system.sun_and_planets.size(); ++i) {
        for (uint64_t j = 0; j < solar_system.sun_and_planets.size(); ++j) {
            if (i != j) {
                update_velocity(solar_system.sun_and_planets[i], solar_system.sun_and_planets[j], dt);
            }
        }
    }

    // Update the velocity of the asteroids
    for (uint64_t i = 0; i < solar_system.asteroids.size(); ++i) {
        for (uint64_t j = 0; j < solar_system.sun_and_planets.size(); ++j) {
            update_velocity(solar_system.asteroids[i], solar_system.sun_and_planets[j], dt);
        }
    }

    // Update position of all bodies
    for (Body &body: solar_system.sun_and_planets) {
        body.pos_x += body.vel_x * dt;
        body.pos_y += body.vel_y * dt;
        body.pos_z += body.vel_z * dt;
    }
    for (Body &body: solar_system.asteroids) {
        body.pos_x += body.vel_x * dt;
        body.pos_y += body.vel_y * dt;
        body.pos_z += body.vel_z * dt;
    }
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
            for (const Body &body: solar_systems[i].sun_and_planets) {
                data.push_back(body.mass);
            }
            write_hdf5(group, "sun_and_planets_mass", {solar_systems[i].sun_and_planets.size()}, data);
        }
        {
            std::vector<double> data;
            for (const Body &body: solar_systems[i].sun_and_planets) {
                data.push_back(body.pos_x);
            }
            for (const Body &body: solar_systems[i].sun_and_planets) {
                data.push_back(body.pos_y);
            }
            for (const Body &body: solar_systems[i].sun_and_planets) {
                data.push_back(body.pos_z);
            }
            write_hdf5(group, "sun_and_planets_position", {3, solar_systems[i].sun_and_planets.size()}, data);
        }
        {
            std::vector<double> data;
            for (const Body &body: solar_systems[i].asteroids) {
                data.push_back(body.pos_x);
            }
            for (const Body &body: solar_systems[i].asteroids) {
                data.push_back(body.pos_y);
            }
            for (const Body &body: solar_systems[i].asteroids) {
                data.push_back(body.pos_z);
            }
            write_hdf5(group, "asteroids_position", {3, solar_systems[i].asteroids.size()}, data);
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
    auto begin = std::chrono::steady_clock::now();
    for (uint64_t i = 0; i < num_of_iterations; ++i) {
        integrate(system, dt);
        if (!filename.empty()) {
            systems.push_back(system);
        }
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
