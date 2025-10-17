# PI Controller C++ lib

C++ implementation of a Proportional-Integral controller for longitudinal velocity control

## Dependencies
- C++17

## Setup Instructions
1. Create build folder
   ```sh
   cd .../Control/Longitudinal/PI_Controller
   mkdir build && cd build
2. Compile
    ```sh
    cmake ..
    make
3. Test run
    ```sh
    ./PI_Controller
   ```

## Controller Details

### Parameters
   - K_p : Proportional term
   - K_i : Integral term
### Input       
- current_speed
- target_speed

### Output
- control effort