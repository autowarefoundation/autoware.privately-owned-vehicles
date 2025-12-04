// This script takes the output of the lane fitler.

// The blue and pink perspective image polynomial fitted lines need to be sampled and the samples should be transformed into a BEV space using a Homography transform (please note - we are not creating a BEV image, only transforming discrete points from the perspective coordinates to the BEV coordinates).

// To do this, you can simply multiply the coordinates of the polyfit line samples by the Homography matrix.

// Verify this is working by visualizing the transformed points.

// Once this has been verfied, we will proceed with the drivable corridor parameter estimation and temporal tracking.