#include <vector>
#include <H5Cpp.h>

void write_hdf5(H5::Group &group, const std::string &name, const std::vector <hsize_t> &shape,
                const std::vector<double> &data);
