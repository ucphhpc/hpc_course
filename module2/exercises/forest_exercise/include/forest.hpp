#include <bhxx/bhxx.hpp>

class Forest {
private:
   bhxx::BhArray<double> _trees;
   void initialize();

public:
   Forest();
   void render();
   void step();
   void save_state(uint64_t postfix_num);
};
