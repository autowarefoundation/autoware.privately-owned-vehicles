#!/usr/bin/env python3

import argparse
import json
import logging
import signal
import carla

import math
import time


def _setup_vehicle(world):
    logging.debug("Spawning vehicle: {}".format("vehicle.tesla.model3"))

    bp_library = world.get_blueprint_library()
    map_ = world.get_map()

    bp = bp_library.filter("vehicle.tesla.model3")[0]
    bp.set_attribute("role_name", "hero")
    bp.set_attribute("ros_name", "hero")

    print(map_.name)
    spawn_points = map_.get_spawn_points()
    for i in range(len(spawn_points)):
        waypt = map_.get_waypoint(spawn_points[i].location)
        print("Spawn Point {}: road {} lane {} section {}".format(i, waypt.road_id, waypt.lane_id, waypt.section_id))

    if 'Town01' in map_.name:
        spawn_pt = spawn_points[102]
    else:
        spawn_pt = spawn_points[5]

    return world.spawn_actor(
        bp,
        spawn_pt,
        attach_to=None)


def main(args):
    world = None
    vehicle = None
    sensors = []
    original_settings = None

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(60.0)
        print(client.get_available_maps())

        if args.map and 'Town01' not in client.get_world().get_map().name:
            logging.info("Loading Town01 map")
            client.load_world('Town01')

        world = client.get_world()

        original_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.01
        world.apply_settings(settings)

        traffic_manager = client.get_trafficmanager()
        traffic_manager.set_synchronous_mode(True)

        vehicle = _setup_vehicle(world)

        logging.info("Running...")

        while True:
            _ = world.tick()
            time.sleep(0.01)



    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')

    finally:
        # Block further KeyboardInterrupts during cleanup
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        try:
            if original_settings:
                logging.info("Restoring original settings")
                world.apply_settings(original_settings)

            if vehicle:
                if vehicle.is_alive:
                    logging.debug("Destroying vehicle: {}".format(vehicle.type_id))
                vehicle.destroy()

        finally:
            # Re-enable KeyboardInterrupt handling
            signal.signal(signal.SIGINT, signal.default_int_handler)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='CARLA ROS2 native')
    argparser.add_argument('--host', metavar='H', default='localhost',
                           help='IP of the host CARLA Simulator (default: localhost)')
    argparser.add_argument('--port', metavar='P', default=2000, type=int,
                           help='TCP port of CARLA Simulator (default: 2000)')
    argparser.add_argument('-f', '--file', default='', required=False, help='File to be executed')
    argparser.add_argument('-v', '--verbose', action='store_true', dest='debug', help='print debug information')
    argparser.add_argument('-a', '--autopilot', action='store_true', dest='autopilot',
                           help='turn on autopilot for the vehicle')
    argparser.add_argument('-m', '--map', action='store_true', dest='map', help='load Town06 map')

    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('Listening to server %s:%s', args.host, args.port)

    main(args)
