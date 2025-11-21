# INPUT : cropped-only image (2880 x 1440)

# 1. Read raw image (2880 x 1860)

# 2. Crop raw image (2880 x 1440)

# 3. Find source points and destination points. Basically I can put a single perfect frame here and manually select the points.

# 4. Define manually 4 source points by picking the best frame where car is perfectly straight (so egoleft/right make perfect trapezoid)
# Four source points will be in the order: left bottom, right bottom, left top, right top
# Must be normalized coords (0, 1)

# 5. Set destination points
# From a BEV grid of 640 x 640
# Four destination points will be in the order: left bottom (159, 639), right bottom (479, 639), left top (159, 0), right top (479, 0)
# They are NOT normalized

# OUTPUT : BEV grid with straight lane markings (not picture, just the mask itself)