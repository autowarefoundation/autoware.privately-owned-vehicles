# Steering Controller C++ lib

C++ implementation of a stering controller using Stanley + curvature feedforward

## Dependencies
- C++17

## Setup Instructions
1. Create build folder
   ```sh
   cd .../Control/Steering/SteeringController
   mkdir build && cd build
2. Compile
    ```sh
    cmake ..
    make
3. Test run
    ```sh
    ./SteeringController
   ```

## Controller Details

### Parameters
   - wheelbase_ : fixed distance between vehicle front and rear wheel axles
   - K_cte      : how aggressive to correct cte
   - K_damp     : added for stability at low speeds, can be increased to reduce aggresive steering

### Input       
- cross-track error
- yaw_error
- curvature
- forward/longitudinal velocity

### Output
- tire steer angle

### Equation
```cpp
    double steering_angle = std::atan(K_cte * cte / (forward_velocity + K_damp)) + yaw_error - std::atan(curvature * wheelbase_);
```


