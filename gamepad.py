import inputs
import importlib
import threading
import time

# ——————————————————————————————————————————————————————————————
# Global state
_pressed_buttons       = set()
_just_pressed_buttons  = set()
_just_released_buttons = set()
_toggles               = {}
_axis_states           = {}

# Map raw ev.code → friendly button name
_CODE_TO_NAME = {
    "BTN_SOUTH":     "A",
    "BTN_EAST":      "B",
    "BTN_NORTH":     "Y",
    "BTN_WEST":      "X",
    "BTN_TL":        "LB",
    "BTN_TR":        "RB",
    "BTN_TL2":       "LT",    # digital fallback
    "BTN_TR2":       "RT",    # digital fallback
    "BTN_SELECT":    "SLCT",
    "BTN_START":     "START",
    "BTN_MODE":      "GUIDE", # not on Xbox
    "BTN_THUMBL":    "L3",
    "BTN_THUMBR":    "R3",
    "BTN_DPAD_UP":   "DPAD_UP",   # often axes instead
    "BTN_DPAD_DOWN": "DPAD_DOWN",
    "BTN_DPAD_LEFT": "DPAD_LEFT",
    "BTN_DPAD_RIGHT":"DPAD_RIGHT",
}

# Map raw ev.code → friendly axis name
_ABS_TO_NAME = {
    "ABS_X":     "LX",    # left stick hor
    "ABS_Y":     "LY",    # left stick ver
    "ABS_RX":    "RX",    # right stick hor
    "ABS_RY":    "RY",    # right stick ver
    "ABS_Z":     "LT",    # left trigger
    "ABS_RZ":    "RT",    # right trigger
    "ABS_HAT0X": "DPAD_X",
    "ABS_HAT0Y": "DPAD_Y",
}

def _repr_button(code: str) -> str:
    return _CODE_TO_NAME.get(code, code)

def _repr_axis(code: str) -> str:
    return _ABS_TO_NAME.get(code)

def _gamepad_event_loop():
    global _pressed_buttons, _just_pressed_buttons, _just_released_buttons, _toggles, _axis_states
    warned = False
    while True:
        try:
            events = inputs.get_gamepad()
        except Exception:
            if not warned:
                print("Controller not found. Waiting...")
                warned = True
            importlib.reload(inputs)
            time.sleep(1)
            continue

        if warned:
            print("Controller connected.")
            warned = False

        for ev in events:
            # Handle buttons
            if ev.ev_type == "Key":
                name = _repr_button(ev.code)
                if ev.state == 1:
                    if name not in _pressed_buttons:
                        _just_pressed_buttons.add(name)
                        _toggles[name] = not _toggles.get(name, False)
                    _pressed_buttons.add(name)
                elif ev.state == 0:
                    if name in _pressed_buttons:
                        _just_released_buttons.add(name)
                    _pressed_buttons.discard(name)
            # Handle axes
            elif ev.ev_type == "Absolute":
                axis = _repr_axis(ev.code)
                if axis:
                    _axis_states[axis] = ev.state

# Start listener
_listener = threading.Thread(target=_gamepad_event_loop, daemon=True)
_listener.start()

# ——————————————————————————————————————————————————————————————
# Public API

def is_pressed(btn_name: str) -> bool:
    """Checks if a key is currently held down."""
    return btn_name in _pressed_buttons or _CODE_TO_NAME.get(btn_name) in _pressed_buttons

def is_toggled(btn_name: str) -> bool:
    """Returns the toggle state of a key. If the key is not registered, it initializes it to False."""
    if btn_name not in _toggles:
        _toggles[btn_name] = False
    return _toggles[btn_name]

def rising_edge(btn_name: str) -> bool:
    """Returns True on the first press of a button until it's released and pressed again."""
    if btn_name in _just_pressed_buttons:
        _just_pressed_buttons.discard(btn_name)
        return True
    return False

def falling_edge(btn_name: str) -> bool:
    """Returns True on the first release of a button until it's pressed and released again."""
    if btn_name in _just_released_buttons:
        _just_released_buttons.discard(btn_name)
        return True
    return False

def get_axis(axis_name: str, normalize: bool = True) -> float:
    """ Return axis state. Can be normalized to -1,+1 for sticks and 0,1 for triggers."""
    val = _axis_states.get(axis_name, 0)
    if not normalize:
        return val
    # sticks: -32768..32767 -> -1.0..1.0
    if axis_name in ("LX", "LY", "RX", "RY"):
        normalized = val / (32767.0 if val >= 0 else 32768.0)
        return round(normalized, 1)
    # triggers: 0..255 or 0..1023 -> 0.0..1.0
    if axis_name in ("LT", "RT"):
        maxv = 255 if val <= 255 else 1023
        return val / maxv
    # D-pad: already -1,0,1
    if axis_name in ("DPAD_X", "DPAD_Y"):
        return val
    return val

def get_dpad(normalize: bool = False) -> tuple:
    """ Return (x, y) for the D-pad hat. """
    return (get_axis("DPAD_X", normalize), get_axis("DPAD_Y", normalize))

# ——————————————————————————————————————————————————————————————
# Example usage
if __name__ == "__main__":
    import time

    print("Polling Xbox controller. Ctrl+C to quit.")
    while True:
        # Buttons
        if is_pressed("A"):
            print("A held")
        if rising_edge("B"):
            print("B pressed")
        if falling_edge("B"):
            print("B released")
        if is_toggled("X"):
            print("X toggled")
        
        # Left joystick
        left_stick = (get_axis("LX"), get_axis("LY"))
        if left_stick != (0, 0):
            print(f"Left stick: {left_stick}")
        
        # Right trigger
        rt = get_axis("RT")
        if rt:
            print(f"Right trigger: {rt:.2f}")
        time.sleep(0.1)
