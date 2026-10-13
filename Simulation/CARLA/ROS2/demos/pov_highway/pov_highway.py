"""PoV_Highway scenario for the VisionPilot closed-loop demo (CARLA 0.9.16).

Subcommands:
  check      load PoV_Highway and print map statistics
  traffic    load PoV_Highway, spawn the ego rig and NPC traffic, run until SIGINT/SIGTERM
  kickstart  give the running ego an initial forward speed
"""

import argparse
import json
import random
import signal
import sys
import time
from pathlib import Path

import carla

MAP_NAME = "PoV_Highway"
# The map has only 2 spawn points, so positions are addressed in OpenDRIVE (road, lane, s).
ROAD_ID = 2
EGO_LANE = -2
NPC_LANES = (-1, -2)
EGO_S = 505.0
# The .xodr has no speed limits, so CARLA uses its 30 km/h default for Traffic Manager.
MAP_SPEED_LIMIT_KMH = 30.0
DEFAULT_SEED = 7
STALL_SECONDS = 4.0
# VisionPilot mis-ranges these models (lead distance ~80 m vs a true 5-19 m) and rear-ends them.
EXCLUDED_MODELS = (
    "microlino",
    "cybertruck",
    "ambulance",
    "firetruck",
    "sprinter",
    "carlacola",
    "european_hgv",
    "fusorosa",
    "t2",
    "crown",
    "isetta",
)
DEFAULT_RIG = Path(__file__).resolve().parents[2] / "config" / "carla916.json"


def connect(args):
    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    client_ver, server_ver = client.get_client_version(), client.get_server_version()
    if client_ver.split("-")[0] != server_ver.split("-")[0]:
        print(f"WARNING: carla client {client_ver} != server {server_ver}", flush=True)
    return client


def lane_transform(world_map, lane_id, s):
    wp = world_map.get_waypoint_xodr(ROAD_ID, lane_id, s)
    if wp is None:
        return None
    t = wp.transform
    return carla.Transform(t.location + carla.Location(z=0.5), t.rotation)


def find_hero(world, role_name):
    for actor in world.get_actors().filter("vehicle.*"):
        if actor.attributes.get("role_name") == role_name:
            return actor
    return None


def spawn_ego(world, rig, transform):
    lib = world.get_blueprint_library()
    bp = lib.find(rig["type"])
    bp.set_attribute("role_name", rig["id"])
    bp.set_attribute("ros_name", rig["id"])
    ego = world.spawn_actor(bp, transform)
    actors = [ego]
    for sensor in rig.get("sensors", []):
        sbp = lib.find(sensor["type"])
        sbp.set_attribute("ros_name", sensor["id"])
        sbp.set_attribute("role_name", sensor["id"])
        for key, value in sensor.get("attributes", {}).items():
            sbp.set_attribute(str(key), str(value))
        sp = sensor["spawn_point"]
        # Rig JSON is in the ROS frame; CARLA is left-handed (same conversion as config_carla.py).
        rel = carla.Transform(
            carla.Location(x=sp["x"], y=-sp["y"], z=sp["z"]),
            carla.Rotation(roll=sp["roll"], pitch=-sp["pitch"], yaw=-sp["yaw"]),
        )
        actor = world.spawn_actor(sbp, rel, attach_to=ego)
        actor.enable_for_ros()
        actors.append(actor)
    return actors


def npc_slots(ego_s, count, spacing):
    """count // len(NPC_LANES) cars per lane centred on the ego; lane i is offset by i*spacing/2."""
    per_lane = count // len(NPC_LANES)
    return [
        (lane, ego_s + 0.5 * spacing * i + (k - per_lane // 2) * spacing)
        for i, lane in enumerate(NPC_LANES)
        for k in range(per_lane)
    ]


def spawn_npcs(client, world, args):
    tm = client.get_trafficmanager(args.tm_port)
    tm.set_random_device_seed(args.seed)
    tm.set_global_distance_to_leading_vehicle(5.0)

    blueprints = [
        b
        for b in world.get_blueprint_library().filter("vehicle.*")
        if int(b.get_attribute("number_of_wheels")) == 4
        and not any(m in b.id for m in EXCLUDED_MODELS)
    ]
    world_map = world.get_map()
    npcs = []
    for lane, s in npc_slots(args.ego_s, args.cars, args.spacing):
        tf = lane_transform(world_map, lane, s)
        if tf is None:
            continue
        bp = random.choice(blueprints)
        bp.set_attribute("role_name", "npc")
        if bp.has_attribute("color"):
            bp.set_attribute("color", random.choice(bp.get_attribute("color").recommended_values))
        car = world.try_spawn_actor(bp, tf)
        if car is None:
            continue
        car.set_autopilot(True, args.tm_port)
        target = args.npc_kmh + random.uniform(-args.speed_spread, args.speed_spread)
        tm.vehicle_percentage_speed_difference(car, (1.0 - target / MAP_SPEED_LIMIT_KMH) * 100.0)
        npcs.append(car)
    return npcs


def cmd_check(args):
    client = connect(args)
    maps = client.get_available_maps()
    if not any(m.split("/")[-1] == MAP_NAME for m in maps):
        sys.exit(f"{MAP_NAME} not installed; available: {', '.join(sorted(maps))}")
    world_map = client.load_world(MAP_NAME).get_map()
    tf = lane_transform(world_map, EGO_LANE, args.ego_s)
    print(f"map={world_map.name}", flush=True)
    print(f"waypoints_2m={len(world_map.generate_waypoints(2.0))}", flush=True)
    print(f"spawn_points={len(world_map.get_spawn_points())}", flush=True)
    print(f"ego_start={tf}", flush=True)


def cmd_traffic(args):
    rig = json.loads(args.rig.read_text())
    random.seed(args.seed)
    client = connect(args)
    world = client.load_world(MAP_NAME)
    ego_tf = lane_transform(world.get_map(), EGO_LANE, args.ego_s)
    if ego_tf is None:
        sys.exit(f"no waypoint at road {ROAD_ID} lane {EGO_LANE} s={args.ego_s}")

    npcs = spawn_npcs(client, world, args)
    ego_actors = spawn_ego(world, rig, ego_tf)
    print(f"seed={args.seed} npc_kmh={args.npc_kmh}", flush=True)
    print(f"ego spawned id={ego_actors[0].id} at {ego_tf}", flush=True)
    print(f"spawned {len(npcs)} npcs", flush=True)

    stop = []
    signal.signal(signal.SIGINT, lambda *_: stop.append(1))
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    try:
        idle_since = {}
        while not stop:
            time.sleep(0.5)
            now = time.time()
            for car in npcs:
                if car.get_velocity().length() > 0.3:
                    idle_since.pop(car.id, None)
                elif now - idle_since.setdefault(car.id, now) > STALL_SECONDS:
                    # Traffic Manager occasionally leaves a car parked at spawn; re-registering frees it.
                    car.set_autopilot(False)
                    car.set_autopilot(True, args.tm_port)
                    idle_since[car.id] = now
    finally:
        client.apply_batch([carla.command.DestroyActor(a) for a in ego_actors[::-1] + npcs])
        print("actors destroyed", flush=True)


def cmd_kickstart(args):
    world = connect(args).get_world()
    hero = find_hero(world, args.role_name)
    if hero is None:
        sys.exit(f"no vehicle with role_name={args.role_name}")
    fwd = hero.get_transform().get_forward_vector()
    hero.set_target_velocity(carla.Vector3D(fwd.x * args.speed, fwd.y * args.speed, 0.0))
    print(f"{args.role_name} kick-started at {args.speed} m/s", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--timeout", type=float, default=120.0, help="client timeout, s")
    p.add_argument("--ego-s", type=float, default=EGO_S, help="ego OpenDRIVE s on road 2 lane -2")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("check")

    t = sub.add_parser("traffic")
    t.add_argument(
        "--rig", type=Path, default=DEFAULT_RIG, help="ego rig JSON (default: %(default)s)"
    )
    t.add_argument("--tm-port", type=int, default=8000)
    t.add_argument("--cars", type=int, default=30, help="split evenly over lanes -1/-2")
    t.add_argument("--spacing", type=float, default=45.0, help="NPC spacing per lane, m")
    t.add_argument("--npc-kmh", type=float, default=60.0, help="mean NPC target speed")
    t.add_argument("--speed-spread", type=float, default=10.0, help="+/- km/h per car")
    t.add_argument("--seed", type=int, default=DEFAULT_SEED, help="python random + Traffic Manager")

    k = sub.add_parser("kickstart")
    k.add_argument("--speed", type=float, default=5.0, help="m/s")
    k.add_argument("--role-name", default="hero")

    args = p.parse_args()
    {"check": cmd_check, "traffic": cmd_traffic, "kickstart": cmd_kickstart}[args.cmd](args)


if __name__ == "__main__":
    main()
