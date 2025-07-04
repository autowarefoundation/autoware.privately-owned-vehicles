Path Finder

Takes in EgoLane and EgoPath detection and tracks error metrics to be used by downstream controller

input: labelled EgoLane, EgoPath detection in uv pixel space from camera perspective
output: robust estimation of cte, yaw_error, curvature from predictor-corrector

How it works
Projects image pixels in camera perspective to BEV pixels
Curve fitting using quadratic polynomial
Calculate raw value for error metrics to be passed to the predictor-corrector filter (Bayes Filter is used as of now)
Redundancies:
    cte: when EgoPath is not available, offset EgoLanes cte by a factor of width
    yaw_error, curvature: product of left and right Gaussian


Missing detection cases:
EgoPath missing: offset EgoLanes cte by a factor of width, 


10 Tracked states: EgoPath (cte,yaw,curv), EgoLane left (cte,yaw,curv), EgoLane right (cte,yaw,curv), width

Product of gaussians to fuse multiple estimates of the same state

Final output states to pass to controller: cte, yaw, curv