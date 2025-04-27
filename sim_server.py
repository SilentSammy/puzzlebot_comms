import sys, os
sys.path[:0] = [os.path.abspath(os.path.join(os.path.dirname(__file__), p)) for p in ('..', '../..')]
import math
import cv2
from keybrd import is_pressed
from sim_tools import DifferentialCar, get_image, sim
import web
import threading

# Connect and get simulator objects
car_cam = sim.getObject('/LineTracer/visionSensor')
car = DifferentialCar()
sim_lock = threading.Lock()

def receive_vel(request):
    lin_vel = request.args.get('v', type=float)
    ang_vel = request.args.get('w', type=float)
    with sim_lock:
        car.linear_speed = lin_vel if lin_vel is not None else car.linear_speed
        car.angular_speed = ang_vel if ang_vel is not None else car.angular_speed
    return f"Linear Velocity: {car.linear_speed}, Angular Velocity: {car.angular_speed}", 200

def video_source():
    with sim_lock:
        frame = get_image(car_cam)
    # Check frame validity to avoid ambiguous truth testing.
    if frame is None or frame.size == 0:
        return None
    return frame

try:
    sim.startSimulation()
    web.http_endpoints = {
        "cmd_vel": receive_vel,
    }
    web.video_endpoints = {
        "car_cam": video_source,
    }
    web.start_webserver(threaded=False)
finally:
    sim.stopSimulation()