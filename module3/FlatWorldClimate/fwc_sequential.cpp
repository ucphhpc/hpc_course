#include <vector>
#include <iostream>
#include <random>
#include <H5Cpp.h>
#include <chrono>
#include <argparse.hpp>


using namespace std;


class World {
public:
    uint64_t latitude;
    uint64_t longitude;
    std::vector<double> data;

    World(uint64_t latitude, uint64_t longitude, double temperature) : latitude(latitude), longitude(longitude),
                                                                       data(latitude * longitude, temperature) {}
};


void radiation(World &world) {

}

/** Function that returns `x` squared */
double squared(double x) {
    return x * x;
}


void radiation(World &world, double sun_position) {
    double sun_angle = cos(sun_position);
    double sun_intensity = 865.0;
    double sun_long = (sin(sun_angle) * (world.longitude / 2)) + world.longitude / 2.;
    double sun_lat = world.latitude / 2.;
    double sun_height = 100 + cos(sun_angle) * 100.;
    double sun_height_squared = squared(sun_height);

    for (uint64_t i = 0; i < world.latitude; ++i) {
        for (uint64_t j = 0; j < world.longitude; ++j) {
            // Euclidean distance between the sun and each earth coordinate
            double distance = sqrt(squared(sun_lat - i) + squared(sun_long - j) + sun_height_squared);
            world.data[i * world.longitude + j] += (sun_intensity / distance);
        }
    }
}

void energy_emmision(World &world) {
    for (uint64_t i = 0; i < world.latitude * world.longitude; ++i) {
        world.data[i] *= 0.99;
    }
}

void diffuse(World &world) {
    for (uint64_t k = 0; k < 10; ++k) {
        for (uint64_t i = 0; i < world.latitude; ++i) {
            for (uint64_t j = 0; j < world.longitude; ++j) {

            }
        }
    }
}


void integrate(World &world, double sun_position) {
    radiation(world, sun_position);
    energy_emmision(world);
    diffuse(world);
}


void simulate(uint64_t num_of_iterations, const std::string &filename) {

    World world{196, 360, 293.15};

    const double t_div = world.longitude / 36.0;

    std::vector <World> world_history;

    auto begin = std::chrono::steady_clock::now();
    for (uint64_t t = 0; t < 100; ++t) {
        integrate(world, t / t_div);
        if (!filename.empty()) {
            world_history.push_back(world);
        }
    }
    if (!filename.empty()) {

    }
    auto end = std::chrono::steady_clock::now();
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
    const std::string &filename = args.getCmdOption("--out");

    simulate(static_cast<uint64_t>(iterations), filename);
    return 0;
}
