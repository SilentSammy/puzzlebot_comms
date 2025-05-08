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

# Map raw ev.code → friendly button name
_CODE_TO_NAME = {
    "BTN_SOUTH":     "A",
    "BTN_EAST":      "B",
    "BTN_NORTH":     "Y",
    "BTN_WEST":      "X",
    "BTN_TL":        "LB",
    "BTN_TR":        "RB",
    "BTN_TL2":       "LT",          # not available on xbox
    "BTN_TR2":       "RT",          # not available on xbox
    "BTN_SELECT":    "SLCT",
    "BTN_START":     "START",
    "BTN_MODE":      "GUIDE",       # not available on xbox
    "BTN_THUMBL":    "L3",
    "BTN_THUMBR":    "R3",
    "BTN_DPAD_UP":   "DPAD_UP",     # not available on xbox
    "BTN_DPAD_DOWN": "DPAD_DOWN",   # not available on xbox
    "BTN_DPAD_LEFT": "DPAD_LEFT",   # not available on xbox
    "BTN_DPAD_RIGHT":"DPAD_RIGHT",  # not available on xbox
}

def _repr_button(code: str) -> str:
    """Return a friendly name for a raw event code."""
    return _CODE_TO_NAME.get(code, code)

def _gamepad_event_loop():
    """Background thread that polls get_gamepad() and updates all sets."""
    global _pressed_buttons, _just_pressed_buttons, _just_released_buttons, _toggles
    _warned = False

    while True:
        try:
            events = inputs.get_gamepad()
        except Exception:
            if not _warned:
                print("Gamepad not connected! Retrying...")
                _warned = True
            importlib.reload(inputs)
            time.sleep(1)
            continue

        if _warned:
            print("Gamepad reconnected!")
            _warned = False

        for ev in events:
            if ev.ev_type != "Key":
                continue

            name = _repr_button(ev.code)

            # PRESS
            if ev.state == 1:
                if name not in _pressed_buttons:
                    _just_pressed_buttons.add(name)
                    _toggles[name] = not _toggles.get(name, False)
                _pressed_buttons.add(name)

            # RELEASE
            elif ev.state == 0:
                if name in _pressed_buttons:
                    _just_released_buttons.add(name)
                _pressed_buttons.discard(name)

# start the listener thread as soon as this module is imported
_listener = threading.Thread(target=_gamepad_event_loop, daemon=True)
_listener.start()

# ——————————————————————————————————————————————————————————————
# Public API

def is_pressed(btn_name: str) -> bool:
    """True while the given button is held down."""
    return btn_name in _pressed_buttons

def is_toggled(btn_name: str) -> bool:
    """Returns the current toggle state (False by default)."""
    if btn_name not in _toggles:
        _toggles[btn_name] = False
    return _toggles[btn_name]

def rising_edge(btn_name: str) -> bool:
    """True once when the button first goes down."""
    if btn_name in _just_pressed_buttons:
        _just_pressed_buttons.discard(btn_name)
        return True
    return False

def falling_edge(btn_name: str) -> bool:
    """True once when the button first goes up."""
    if btn_name in _just_released_buttons:
        _just_released_buttons.discard(btn_name)
        return True
    return False

# ——————————————————————————————————————————————————————————————
# Example usage
if __name__ == "__main__":
    print("Polling Xbox controller. Press Ctrl+C to exit.")
    try:
        while True:
            if is_pressed("A"):
                print("A is held")
            if rising_edge("B"):
                print("B was pressed")
            if falling_edge("B"):
                print("B was released")
            if is_toggled("X"):
                print("X is toggled")
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("Goodbye!")
