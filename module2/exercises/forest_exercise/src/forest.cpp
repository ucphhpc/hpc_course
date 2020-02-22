#include <cstdlib>
#include <forest.hpp>
#include <iostream>
#include <H5Cpp.h>
#include <util/hdf5.hpp>

using namespace bhxx;

Forest::Forest() {
    this->initialize();
}

// Initializes the forest as a 100x100 matrix of trees with the middle tree on fire
// Saves the initial forest to an HDF5 file
void Forest::initialize() {
    this->_trees = ones<double>({100, 100});
    this->_trees *= -1;
    // Set fire to the middle
    this->_trees[50][50] = 1;
    // set the edge to have been burned
    uint64_t view_size = this->_trees.shape()[0];
    // Setup edge views
    BhArray<double> horizontal(this->_trees.base(), {2, view_size}, {(view_size*view_size)-view_size, 1},
            this->_trees.offset());

    BhArray<double> vertical(this->_trees.base(), {view_size, 2}, {view_size, view_size-1}, this->_trees.offset());
     // Set the borders to have been burned already
     horizontal = 0;
     vertical = 0;
     this->save_state(0);
}

void Forest::step() {
    // TODO, implement the step function
    
    // For each step of the simulation, each tree on fire should have a chance to spread to their neighbouring trees 
    // who is not already on fire this step or has been previously burned

    // In turn, at the end of a step, all trees that were burning in the previous step should be set to having been burned.
    // Meaning that a tree will not be on fire for more than one simulation step.

    // Tree states
    // -1 == already burned
    // 0 == alive tree
    // 1 == tree on fire
}

// Function that saves the current state of the forest trees to an HDF5 file into the ./forest_steps directory
void Forest::save_state(uint64_t postfix_num = 0) {
    const int dir_err = system("mkdir -p forest_steps");
    if (dir_err == -1) {
        throw std::runtime_error("Failed to create the required forest_steps directory");
    }

    // Save the current forest state
    H5::H5File file("forest_steps/forest_" + std::to_string(postfix_num), H5F_ACC_TRUNC);
    H5::Group group(file.createGroup("/forest"));
    auto vec = this->_trees.vec();
    write_hdf5(group, "forest_tress", {1, vec.size()}, vec);
}
