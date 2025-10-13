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
   - K_yaw      : how aggresive to correct yaw_error
   - K_cte      : how aggressive to correct cte
   - K_damp     : added for stability at low speeds, can be increased to reduce aggresive steering
   - K_ff       : factor to correct for wheel slip and non-linear relationship between steering wheel and tire steer angle (under no-slip conditions and known relationship between steering wheel and tire steer angle, this should be set to 1.0)
### Input       
- cross-track error
- yaw_error
- curvature
- forward/longitudinal velocity

### Output
- tire steer angle

### Equation
```cpp
 double steering_angle = K_yaw * yaw_error + std::atan(K_cte * cte /(forward_velocity + K_damp)) 
                        - K_ff * std::atan(curvature * wheelbase_);
```


