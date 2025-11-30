Contains scripts to process open datasets and create semantic masks in a unified labelling scheme according to the DomainSeg neural task specification. To process each open dataset, the file structure and heirarchy of published open datasets was simplified manually such that each open dataset had a two top level folders, a first folder with all images and a second folder with ground truth labels.

Open semantic segmentation datasets contain various labelling methodologies and semantic classes. The scripts in create_masks parse data and create semantic colormaps in a single unified semantic format.

#### The open datasets processed for DomainSeg include:
- [ROADWork](https://www.cs.cmu.edu/~roadwork/)
- [Mapillary Vistas 2.0](https://blog.mapillary.com/update/2021/01/18/vistas-2-dataset.html)

A unified semantic class was created for both datasets which included all movable roadwork objects including traffic cones, traffic barrels, traffic drums and tubular markers. Although both ROADWork and Mapillary Vistas 2.0 were parsed for training the DomainSeg network, it was found that Mapillary Vistas 2.0 scenes were not representative of construction zones and given the small size of samples where roadwork objects were present (1.6K) - it was decided to not rely on Mapillary Vistas 2.0 during network training