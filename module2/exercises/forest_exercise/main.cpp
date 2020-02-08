#include <vector>
#include <argparse.hpp>
#include <forest.hpp>

int main(int argc, char **argv) {
    util::ArgParser args(argc, argv);
    int64_t iterations;
    if (args.cmdOptionExists("--iter")) {
        iterations = std::stoi(args.getCmdOption("--iter"));
        if (iterations < 0) {
            throw std::invalid_argument("iter most be a positive integer (e.g. --iter 20)");
        }
    } else {
        throw std::invalid_argument("You must specify the number of iterations (e.g. --iter 20)");
    }

    Forest forest{};
    for (int64_t i = 0; i < iterations; i++) {
        forest.step();
        forest.save_state(i);
    }

    return 0;
}
