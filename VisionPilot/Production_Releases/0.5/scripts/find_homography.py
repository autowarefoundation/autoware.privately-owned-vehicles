# INPUT : cropped-only image (2880 x 1440)

# Define manually 4 source points by picking the best frame where car is perfectly straight (so egoleft/right make perfect trapezoid)
# 4 source points will be in the order: left bottom, right bottom, left top, right top
# Must be normalized coords (0, 1)

# Now we have to define BEV grid size

# OUTPUT : BEV grid with straight lane markings (not picture, just the mask itself)