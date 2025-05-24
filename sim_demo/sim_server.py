import sys, os
sys.path[:0] = [os.path.abspath(os.path.join(os.path.dirname(__file__), p)) for p in ('..', '../..')]
import math
import cv2
from keybrd import is_pressed
from sim_tools import DifferentialCar, get_image, sim
import web
import threading
import time

# Connect and get simulator objects
car_cam = sim.getObject('/Puzzlebot/visionSensor')
car = DifferentialCar()
car.left_wheel = sim.getObject('/Puzzlebot/DynamicLeftJoint')
car.right_wheel = sim.getObject('/Puzzlebot/DynamicRightJoint')
sim_lock = threading.Lock()
lin_vel_safe = 0.0
ang_vel_safe = 0.0
use_safe = False

def receive_vel(request):
    global use_safe
    use_safe = False
    lin_vel = request.args.get('v', type=float)
    ang_vel = request.args.get('w', type=float)
    with sim_lock:
        car.linear_speed = lin_vel if lin_vel is not None else car.linear_speed
        car.angular_speed = ang_vel if ang_vel is not None else car.angular_speed
    return {"v": car.linear_speed, "w": car.angular_speed}, 200

def receive_vel_safe(request):
    global use_safe
    use_safe = True
    v = request.args.get('v', type=float)
    w = request.args.get('w', type=float)
    global lin_vel_safe, ang_vel_safe
    lin_vel_safe = v if v is not None else lin_vel_safe
    ang_vel_safe = w if w is not None else ang_vel_safe
    return {"v": lin_vel_safe, "w": ang_vel_safe}, 200

def video_source():
    with sim_lock:
        frame = get_image(car_cam)
    # Check frame validity to avoid ambiguous truth testing.
    if frame is None or frame.size == 0:
        return None
    return frame

if __name__ == "__main__":
    try:
        sim.startSimulation()
        # Start a timer to call car.accelerate_to and car.spin_up_to every 0.1 seconds
        def update_car_speed():
            while True:
                if use_safe:
                    with sim_lock:
                        car.accelerate_to(lin_vel_safe)
                        car.spin_up_to(ang_vel_safe)
                time.sleep(0.1)
        threading.Thread(target=update_car_speed, daemon=True).start()

        # Start the web server and simulation
        web.http_endpoints["cmd_vel"] = receive_vel
        web.http_endpoints["cmd_vel_safe"] = receive_vel_safe
        web.video_endpoints["car_cam"] = video_source
        web.start_webserver(threaded=False)
    finally:
        sim.stopSimulation()