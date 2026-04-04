import sys, os
sys.path[:0] = [os.path.abspath(os.path.join(os.path.dirname(__file__), p)) for p in ('..', '../..')]
import math
import cv2
from sim_tools import DifferentialCar, get_image, sim
import io_server
import threading
import time

# Connect and get simulator objects
car_cam = sim.getObject('/Puzzlebot/visionSensor')
car = DifferentialCar()
car.left_wheel = sim.getObject('/Puzzlebot/DynamicLeftJoint')
car.right_wheel = sim.getObject('/Puzzlebot/DynamicRightJoint')
sim_lock = threading.Lock()

def receive_vel(request):
    params = dict(request.query_params)
    x = float(params['x']) if 'x' in params else None
    w = float(params['w']) if 'w' in params else None
    with sim_lock:
        if x is not None:
            car.linear_speed = x
        if w is not None:
            car.angular_speed = w
    return {"x": car.linear_speed, "w": car.angular_speed}, 200

def video_source():
    with sim_lock:
        frame = get_image(car_cam)
    if frame is None or frame.size == 0:
        return None
    return frame

if __name__ == "__main__":
    try:
        sim.startSimulation()
        
        # Start the web server and simulation
        io_server.port = 5001
        io_server.http_endpoints["cmd_vel"] = receive_vel
        io_server.stream_endpoints["car_cam"] = video_source
        io_server.start_webserver(threaded=False)
    finally:
        sim.stopSimulation()
