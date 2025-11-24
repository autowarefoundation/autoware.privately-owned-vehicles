## EgoPath
Self-driving cars most often rely on lane detection to detect the driving path, especially on highways. By detecting the left and right ego lane, the vehicle is able to determine an ideal driving path and track that path to remain centred within its lane. However, there are many circumstances when this is not possible and lanes are not visible - leading to system failure and safety risk - for example, lanes may be faded due to old road infrastructure, lanes may be covered by snow or obstructed due to road splashes/reflections during heavy rain. Another important edge case scenario is when the driving path is no longer defined by lanes, but rather by roadwork elements such as traffic cones and construction barriers, and in such scenarios, lane detection systems are unable to inform a car of where it should drive. 

EgoPath is a neural network which processes raw camera image frames and directly predicts the driving path in an end-to-end manner, allowing for safe autonomous driving in challenging road conditions where lane detection alone is insufficient.

The EgoPath network comprises a total of 3.17 Billion Floating Point Operations.
