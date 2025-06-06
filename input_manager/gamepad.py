import inputs
import importlib
import threading
import time

# ——————————————————————————————————————————————————————————————
# Global state
_pressed_buttons       = set()   # buttons currently down
_just_pressed_buttons  = set()   # buttons that went down since last query
_just_released_buttons = set()   # buttons that went up since last query
_toggles               = {}      # button_name -> bool
_axis_states           = {}      # axis_name -> raw value

# Map raw ev.code → friendly button name
_CODE_TO_NAME = {
    "BTN_SOUTH":     "A",
    "BTN_EAST":      "B",
    "BTN_NORTH":     "Y",
    "BTN_WEST":      "X",
    "BTN_TL":        "LB",
    "BTN_TR":        "RB",
    "BTN_TL2":       "LT",
    "BTN_TR2":       "RT",
    "BTN_SELECT":    "SLCT",
    "BTN_START":     "START",
    "BTN_MODE":      "GUIDE",
    "BTN_THUMBL":    "L3",
    "BTN_THUMBR":    "R3",
    "BTN_DPAD_UP":   "DPAD_UP",
    "BTN_DPAD_DOWN": "DPAD_DOWN",
    "BTN_DPAD_LEFT": "DPAD_LEFT",
    "BTN_DPAD_RIGHT":"DPAD_RIGHT",
}

# Map raw absolute codes → friendly axis names
_ABS_TO_NAME = {
    'ABS_X':     'LX',      # left stick X
    'ABS_Y':     'LY',      # left stick Y
    'ABS_RX':    'RX',      # right stick X
    'ABS_RY':    'RY',      # right stick Y
    'ABS_Z':     'LT',      # left trigger
    'ABS_RZ':    'RT',      # right trigger
    'ABS_HAT0X': 'DPAD_X',  # D-pad horizontal
    'ABS_HAT0Y': 'DPAD_Y',  # D-pad vertical
}

# Track disconnect status to throttle warnings
_warned_disconnected = False

# ——————————————————————————————————————————————————————————————
# Internal helpers
def _repr_button(code: str) -> str:
    """Return a friendly name for a raw event code."""
    return _CODE_TO_NAME.get(code, code)

# ——————————————————————————————————————————————————————————————
# Background event loop

def _gamepad_event_loop():
    global _warned_disconnected
    while True:
        try:
            events = inputs.get_gamepad()
        except Exception:
            if not _warned_disconnected:
                print("Gamepad not connected. Waiting for connection...")
                _warned_disconnected = True
            importlib.reload(inputs)
            # clear stale state
            _pressed_buttons.clear()
            _axis_states.clear()
            time.sleep(1)
            continue
        if _warned_disconnected:
            print("Gamepad connected.")
            _warned_disconnected = False

        for ev in events:
            # DIGITAL BUTTONS
            if ev.ev_type == 'Key':
                name = _repr_button(ev.code)
                if ev.state == 1:
                    if name not in _pressed_buttons:
                        _just_pressed_buttons.add(name)
                        _toggles[name] = not _toggles.get(name, False)
                    _pressed_buttons.add(name)
                else:
                    if name in _pressed_buttons:
                        _just_released_buttons.add(name)
                    _pressed_buttons.discard(name)

            # ANALOG AXES & BINARY MAPPINGS
            elif ev.ev_type == 'Absolute':
                axis = _ABS_TO_NAME.get(ev.code)
                if not axis:
                    continue
                value = ev.state
                # update raw axis state
                _axis_states[axis] = value

                # TRIGGERS as binary buttons
                if axis in ('LT', 'RT'):
                    if value > 0:
                        if axis not in _pressed_buttons:
                            _just_pressed_buttons.add(axis)
                        _pressed_buttons.add(axis)
                    else:
                        if axis in _pressed_buttons:
                            _just_released_buttons.add(axis)
                        _pressed_buttons.discard(axis)

                # D-PAD as binary buttons
                elif axis == 'DPAD_X':
                    # left
                    if value < 0:
                        if 'DPAD_LEFT' not in _pressed_buttons:
                            _just_pressed_buttons.add('DPAD_LEFT')
                        _pressed_buttons.add('DPAD_LEFT')
                    else:
                        if 'DPAD_LEFT' in _pressed_buttons:
                            _just_released_buttons.add('DPAD_LEFT')
                        _pressed_buttons.discard('DPAD_LEFT')
                    # right
                    if value > 0:
                        if 'DPAD_RIGHT' not in _pressed_buttons:
                            _just_pressed_buttons.add('DPAD_RIGHT')
                        _pressed_buttons.add('DPAD_RIGHT')
                    else:
                        if 'DPAD_RIGHT' in _pressed_buttons:
                            _just_released_buttons.add('DPAD_RIGHT')
                        _pressed_buttons.discard('DPAD_RIGHT')

                elif axis == 'DPAD_Y':
                    # up
                    if value < 0:
                        if 'DPAD_UP' not in _pressed_buttons:
                            _just_pressed_buttons.add('DPAD_UP')
                        _pressed_buttons.add('DPAD_UP')
                    else:
                        if 'DPAD_UP' in _pressed_buttons:
                            _just_released_buttons.add('DPAD_UP')
                        _pressed_buttons.discard('DPAD_UP')
                    # down
                    if value > 0:
                        if 'DPAD_DOWN' not in _pressed_buttons:
                            _just_pressed_buttons.add('DPAD_DOWN')
                        _pressed_buttons.add('DPAD_DOWN')
                    else:
                        if 'DPAD_DOWN' in _pressed_buttons:
                            _just_released_buttons.add('DPAD_DOWN')
                        _pressed_buttons.discard('DPAD_DOWN')
        # end for events
# end while

# start the listener thread as soon as this module is imported
_listener = threading.Thread(target=_gamepad_event_loop, daemon=True)
_listener.start()

# ——————————————————————————————————————————————————————————————
# Public API
def is_pressed(btn_name: str) -> bool:
    """True as long as the given button is held down."""
    return btn_name in _pressed_buttons


def is_toggled(btn_name: str) -> bool:
    """Flip-flop state for each press; returns current toggle state."""
    if btn_name not in _toggles:
        _toggles[btn_name] = False
    return _toggles[btn_name]


def rising_edge(btn_name: str) -> bool:
    """True exactly once when the button first goes down."""
    if btn_name in _just_pressed_buttons:
        _just_pressed_buttons.remove(btn_name)
        return True
    return False


def falling_edge(btn_name: str) -> bool:
    """True exactly once when the button first goes up."""
    if btn_name in _just_released_buttons:
        _just_released_buttons.remove(btn_name)
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


# ——————————————————————————————————————————————————————————————
# Example usage
if __name__ == "__main__":
    import time

    print("Polling Xbox controller. Ctrl+C to quit.")
    while True:
        if _pressed_buttons:
            print(f"Pressed buttons: {', '.join(_pressed_buttons)}")
            time.sleep(0.1)
        continue
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
