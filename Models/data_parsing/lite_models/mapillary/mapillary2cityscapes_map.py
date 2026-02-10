
CITYSCAPES_TRAIN_ID_MAP = {
    "road": 0,
    "sidewalk": 1,
    "building": 2,
    "wall": 3,
    "fence": 4,
    "pole": 5,
    "traffic light": 6,
    "traffic sign": 7,
    "vegetation": 8,
    "terrain": 9,
    "sky": 10,
    "person": 11,
    "rider": 12,
    "car": 13,
    "truck": 14,
    "bus": 15,
    "train": 16,
    "motorcycle": 17,
    "bicycle": 18,
}
# Mapillary → Cityscapes class ID mapping

MAPILLARY_TO_CITYSCAPES = {

    # Animals (no Cityscapes equivalent)
    "animal--bird": 255,
    "animal--ground-animal": 255,

    # Barriers / flat / curb / etc
    "construction--barrier--curb": 1,   # sidewalk
    "construction--barrier--fence": 4,  # fence
    "construction--barrier--guard-rail": 4,
    "construction--barrier--other-barrier": 255,
    "construction--barrier--wall": 3,   # wall

    # Flat
    "construction--flat--bike-lane": 0,     # road
    "construction--flat--crosswalk-plain": 0,
    "construction--flat--curb-cut": 1,
    "construction--flat--parking": 0,
    "construction--flat--pedestrian-area": 1,
    "construction--flat--rail-track": 16,   # train
    "construction--flat--road": 0,
    "construction--flat--service-lane": 0,
    "construction--flat--sidewalk": 1,

    # Structures
    "construction--structure--bridge": 2,    # building
    "construction--structure--building": 2,
    "construction--structure--tunnel": 2,

    # Humans / riders
    "human--person": 11,
    "human--rider--bicyclist": 12,
    "human--rider--motorcyclist": 12,
    "human--rider--other-rider": 12,

    # Markings
    "marking--crosswalk-zebra": 0,
    "marking--general": 0,

    # Nature
    "nature--mountain": 9,       # terrain
    "nature--sand": 9,
    "nature--sky": 10,
    "nature--snow": 9,
    "nature--terrain": 9,
    "nature--vegetation": 8,
    "nature--water": 255,

    # Objects (ignored)
    "object--banner": 255,
    "object--bench": 255,
    "object--bike-rack": 255,
    "object--billboard": 255,
    "object--catch-basin": 255,
    "object--cctv-camera": 255,
    "object--fire-hydrant": 255,
    "object--junction-box": 255,
    "object--mailbox": 255,
    "object--manhole": 255,
    "object--phone-booth": 255,
    "object--pothole": 255,
    "object--street-light": 255,

    # Support structures
    "object--support--pole": 5,              # pole
    "object--support--traffic-sign-frame": 7,
    "object--support--utility-pole": 5,

    # Traffic elements
    "object--traffic-light": 6,
    "object--traffic-sign--back": 7,
    "object--traffic-sign--front": 7,
    "object--trash-can": 255,

    # Vehicles
    "object--vehicle--bicycle": 18,
    "object--vehicle--boat": 255,
    "object--vehicle--bus": 15,
    "object--vehicle--car": 13,
    "object--vehicle--caravan": 255,
    "object--vehicle--motorcycle": 17,
    "object--vehicle--on-rails": 16,
    "object--vehicle--other-vehicle": 255,
    "object--vehicle--trailer": 255,
    "object--vehicle--truck": 14,
    "object--vehicle--wheeled-slow": 255,

    # Void
    "void--car-mount": 255,
    "void--ego-vehicle": 255,
    "void--unlabeled": 255,
}


"""
MAPILLARY CLASSES :
animal--bird, 
animal--ground-animal, 
construction--barrier--curb, 
construction--barrier--fence, 
construction--barrier--guard-rail, 
construction--barrier--other-barrier, 
construction--barrier--wall, 
construction--flat--bike-lane, 
construction--flat--crosswalk-plain, 
construction--flat--curb-cut, 
construction--flat--parking, 
construction--flat--pedestrian-area, 
construction--flat--rail-track, 
construction--flat--road, 
construction--flat--service-lane, 
construction--flat--sidewalk, 
construction--structure--bridge, 
construction--structure--building, 
construction--structure--tunnel, 
human--person, 
human--rider--bicyclist, 
human--rider--motorcyclist, 
human--rider--other-rider, 
marking--crosswalk-zebra, 
marking--general, 
nature--mountain, 
nature--sand, 
nature--sky, 
nature--snow, 
nature--terrain, 
nature--vegetation, 
nature--water, 
object--banner, 
object--bench, 
object--bike-rack, 
object--billboard, 
object--catch-basin, 
object--cctv-camera, 
object--fire-hydrant, 
object--junction-box, 
object--mailbox, 
object--manhole, 
object--phone-booth, 
object--pothole, 
object--street-light, 
object--support--pole, 
object--support--traffic-sign-frame, 
object--support--utility-pole, 
object--traffic-light, 
object--traffic-sign--back, 
object--traffic-sign--front, 
object--trash-can, 
object--vehicle--bicycle, 
object--vehicle--boat, 
object--vehicle--bus, 
object--vehicle--car, 
object--vehicle--caravan, 
object--vehicle--motorcycle, 
object--vehicle--on-rails, 
object--vehicle--other-vehicle, 
object--vehicle--trailer, 
object--vehicle--truck, 
object--vehicle--wheeled-slow, 
void--car-mount, 
void--ego-vehicle, 
void--unlabeled, 

"""


"""
mapillary to cityscapes mapping:
animal--bird                        → ignore
animal--ground-animal              → ignore
construction--barrier--curb        → sidewalk
construction--barrier--fence       → fence
construction--barrier--guard-rail  → fence
construction--barrier--other-barrier → ignore
construction--barrier--wall        → wall
construction--flat--bike-lane      → road
construction--flat--crosswalk-plain → road
construction--flat--curb-cut       → sidewalk
construction--flat--parking        → road
construction--flat--pedestrian-area → sidewalk
construction--flat--rail-track     → train
construction--flat--road           → road
construction--flat--service-lane   → road
construction--flat--sidewalk       → sidewalk
construction--structure--bridge    → building
construction--structure--building  → building
construction--structure--tunnel    → building
human--person                      → person
human--rider--bicyclist            → rider
human--rider--motorcyclist         → rider
human--rider--other-rider          → rider
marking--crosswalk-zebra           → road
marking--general                   → road
nature--mountain                   → terrain
nature--sand                       → terrain
nature--sky                        → sky
nature--snow                       → terrain
nature--terrain                    → terrain
nature--vegetation                 → vegetation
nature--water                      → ignore
object--banner                     → ignore
object--bench                      → ignore
object--bike-rack                  → ignore
object--billboard                  → ignore
object--catch-basin                → ignore
object--cctv-camera                → ignore
object--fire-hydrant               → ignore
object--junction-box               → ignore
object--mailbox                    → ignore
object--manhole                    → ignore
object--phone-booth                → ignore
object--pothole                    → ignore
object--street-light               → ignore
object--support--pole              → pole
object--support--traffic-sign-frame → traffic sign
object--support--utility-pole      → pole
object--traffic-light              → traffic light
object--traffic-sign--back         → traffic sign
object--traffic-sign--front        → traffic sign
object--trash-can                  → ignore
object--vehicle--bicycle           → bicycle
object--vehicle--boat              → ignore
object--vehicle--bus               → bus
object--vehicle--car               → car
object--vehicle--caravan           → ignore
object--vehicle--motorcycle        → motorcycle
object--vehicle--on-rails          → train
object--vehicle--other-vehicle     → ignore
object--vehicle--trailer           → ignore
object--vehicle--truck             → truck
object--vehicle--wheeled-slow      → ignore
void--car-mount                    → ignore
void--ego-vehicle                  → ignore
void--unlabeled                    → ignore
"""