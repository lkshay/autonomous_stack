#!/usr/bin/env python3
"""spawn_basic.py  Minimal CARLA data capture script

Run this from Windows PowerShell CMD or from WSL as long as the
CARLA simulator is listening on the given hostport default 127.0.0.12000.

It will
1. Connect to the CARLA server.
2. Spawn an ego Tesla Model 3.
3. Attach a 64 beam LiDAR and an RGB camera.
4. Drive the ego in autopilot for N frames.
5. Save point clouds to ...velodyne000000.bin  KITTI float32 x y z i
    and images to ...images000000.png.

Example
      python spawn_basic.py  
             --frames 600  
             --out Cdatasetsdemo1  
             --host 127.0.0.1  --port 2000

Requirements
      pip install carla numpy opencv python

Author 2025 07 08  Virtual AV Stack bootstrap
"""
from __future__ import annotations
import argparse
import pathlib
import time
import numpy as np
import cv2
import carla


def save_lidar(out_dir: pathlib.Path):
     """Factory that returns a callback bound to *out_dir*"""

     def _callback(pointcloud: carla.LidarMeasurement):
          # raw_data already little-endian float32 x,y,z,intensity
          np.frombuffer(pointcloud.raw_data, dtype=np.float32).tofile(
                out_dir / f"{pointcloud.frame:06}.bin"
          )

     return _callback


def save_rgb(out_dir: pathlib.Path):
     def _callback(image: carla.Image):
          # Convert BGRA -> BGR numpy array
          array = np.frombuffer(image.raw_data, dtype=np.uint8)
          array = array.reshape((image.height, image.width, 4))[:, :, :3]
          cv2.imwrite(str(out_dir / f"{image.frame:06}.png"), array)

     return _callback


def main() -> None:
     parser = argparse.ArgumentParser()
     parser.add_argument("--host", default="127.0.0.1")
     parser.add_argument("--port", type=int, default=2000)
     parser.add_argument("--frames", type=int, default=600)
     parser.add_argument("--out", type=pathlib.Path, required=True)
     args = parser.parse_args()

     out_lidar = args.out / "velodyne"
     out_rgb = args.out / "images"
     out_lidar.mkdir(parents=True, exist_ok=True)
     out_rgb.mkdir(parents=True, exist_ok=True)

     client = carla.Client(args.host, args.port)
     client.set_timeout(5.0)
     world = client.get_world()

     blueprint_lib = world.get_blueprint_library()

     # Spawn ego vehicle
     ego_bp = blueprint_lib.filter("vehicle.tesla.model3")[0]
     spawn_pt = world.get_map().get_spawn_points()[0]
     ego = world.try_spawn_actor(ego_bp, spawn_pt)
     if ego is None:
          raise RuntimeError("Failed to spawn ego vehicle (spawn point occupied?)")

     # LiDAR parameters
     lidar_bp = blueprint_lib.find("sensor.lidar.ray_cast")
     lidar_bp.set_attribute("channels", "64")
     lidar_bp.set_attribute("rotation_frequency", "10")
     lidar_bp.set_attribute("points_per_second", "1300000")
     lidar_tf = carla.Transform(carla.Location(z=1.8))
     lidar = world.spawn_actor(lidar_bp, lidar_tf, attach_to=ego)

     # RGB camera parameters
     cam_bp = blueprint_lib.find("sensor.camera.rgb")
     cam_bp.set_attribute("image_size_x", "1280")
     cam_bp.set_attribute("image_size_y", "720")
     cam_bp.set_attribute("fov", "90")
     cam_tf = carla.Transform(carla.Location(x=1.2, z=1.5))
     camera = world.spawn_actor(cam_bp, cam_tf, attach_to=ego)

     # Register callbacks
     lidar.listen(save_lidar(out_lidar))
     camera.listen(save_rgb(out_rgb))

     # Let autopilot drive so scenery changes
     ego.set_autopilot(True)

     print(f"Capturing {args.frames} frames ... (Ctrl-C to abort)")
     start_frame = world.get_snapshot().frame
     target_frame = start_frame + args.frames

     try:
          while world.get_snapshot().frame < target_frame:
                time.sleep(0.05)  # lightweight wait
     finally:
          print("Stopping sensors ...")
          lidar.stop()
          camera.stop()
          ego.set_autopilot(False)
          print("Destroying actors ...")
          for actor in [lidar, camera, ego]:
                if actor.is_alive:
                     actor.destroy()
          print("Done.")


if __name__ == "__main__":
     main()
