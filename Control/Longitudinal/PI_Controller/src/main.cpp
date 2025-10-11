#include "pi_controller.hpp"

int main()
{
    PI_Controller pi(0.1, 0.01);
    double current_speed = 5.0; // current speed in m/s
    double target_speed = 15.0; // target speed in m/s  

    int i = 5;
    while (i--){
        std::cout << pi.computeEffort(current_speed, target_speed) << std::endl;
        current_speed += 2.0; // Simulate speed increase
    }
  
    return 0;
}