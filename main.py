import cv2
import time
import math
from puzzlebot_client import PuzzlebotClient
import teleop as inp

# Connection
# puzzlebot = PuzzlebotClient("http://127.0.0.1:5001")
puzzlebot = PuzzlebotClient("http://192.168.137.158:5000", trim=-0.9)

def get_user_cmd(gamepad_index: int = 0, slow_thr: float = 0.2, fast_thr: float = 0.6, 
                 slow_yaw: float = math.radians(60), fast_yaw: float = math.radians(180)):
    """Get user input as velocity command dict for differential drive robot.
    
    Args:
        gamepad_index: Which gamepad/keymapping to use
        slow_thr: Linear velocity without boost (m/s)
        fast_thr: Linear velocity with full boost (m/s)
        slow_yaw: Angular velocity without boost (rad/s)
        fast_yaw: Angular velocity with full boost (rad/s)
    
    Returns:
        Command dict with 'x' (linear velocity) and 'w' (angular velocity)
    """
    
    keymappings = [
        ('w', 's', 'a', 'd', 'c'),  # Gamepad 0: fwd, back, rotate_left, rotate_right, boost
        ('i', 'k', 'j', 'l', 'm'),  # Gamepad 1
    ]

    fwd, bck, ccw, cw, boost_key = keymappings[gamepad_index] if gamepad_index < len(keymappings) else keymappings[0]
    s = str(gamepad_index)
    
    # Boost: use highest of either trigger for one-handed operation
    boost_rt = inp.get_axis(f'RT{s}')
    boost_lt = inp.get_axis(f'LT{s}')
    boost_kbd = 1.0 if inp.is_pressed(boost_key) else 0.0
    boost = max(boost_rt, boost_lt, boost_kbd)
    
    thr_scale = slow_thr + (fast_thr - slow_thr) * boost
    yaw_scale = slow_yaw + (fast_yaw - slow_yaw) * boost
    
    # Rotation: use highest magnitude from RX, LX, or keyboard
    # Negate joystick axes so left (negative) = ccw (positive w)
    w_rx = -inp.get_axis(f'RX{s}')
    w_lx = -inp.get_axis(f'LX{s}')
    w_kbd = 0.0
    if inp.is_pressed(ccw):  # 'a' = ccw = positive w
        w_kbd = 1.0
    elif inp.is_pressed(cw):  # 'd' = cw = negative w
        w_kbd = -1.0
    
    # Pick the rotation input with highest magnitude
    w = max([w_rx, w_lx, w_kbd], key=abs)
    
    return {
        'x':  inp.get_bipolar_ctrl(fwd, bck, f'LY{s}') * thr_scale,
        'w': w * yaw_scale
    }

def show_frame(img, name, scale=1):
    """Display frame in OpenCV window."""
    show_frame.first_time = show_frame.first_time if hasattr(show_frame, 'first_time') else True
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(name, cv2.WND_PROP_TOPMOST, 1)
    if show_frame.first_time:
        cv2.resizeWindow(name, int(img.shape[1]*scale), int(img.shape[0]*scale))
        show_frame.first_time = False
    cv2.imshow(name, img)
    if cv2.waitKey(1) & 0xFF == 27:
        raise KeyboardInterrupt

try:
    print("RC Control Starting...")
    print("Controls: W/S - forward/back, A/D - rotate, C/RT - boost")
    print("Q/E - decrease/increase trim")
    print("Press ESC to exit\n")
    
    while True:
        # Get camera frame
        frame = puzzlebot.get_frame()
        
        # Get user command
        cmd = get_user_cmd(gamepad_index=0)
        
        # Adjust trim on the fly
        if inp.rising_edge('q'):
            puzzlebot.trim -= 0.05
            print(f"Trim: {puzzlebot.trim:.3f}")
        if inp.rising_edge('e'):
            puzzlebot.trim += 0.05
            print(f"Trim: {puzzlebot.trim:.3f}")
        
        # Send velocity command to robot
        puzzlebot.send_vel(cmd)
        
        # Display frame if available
        if frame is not None:
            show_frame(frame, "Puzzlebot Stream")
        else:
            # Allow keyboard interrupt even without frames
            if cv2.waitKey(1) & 0xFF == 27:
                break
            time.sleep(0.01)  # Small delay to avoid busy loop
            
except KeyboardInterrupt:
    print("Exiting...")
finally:
    puzzlebot.send_vel({'x': 0, 'w': 0})
    puzzlebot._stop_stream()
    cv2.destroyAllWindows()
