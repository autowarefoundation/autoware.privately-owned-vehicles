#include "pure_pursuit.hpp"

int main() {
    PurePursuit pp(2.5);
    while (true) {
        pp.computeSteering(0.1, 0.05, 0.0, 0.0);
    }

    return 0;
}