"""
input_manager.py

Unified input module for both keyboard (keybrd.py) and gamepad (gamepad.py) events.
Provides the same polling API: is_pressed, rising_edge, falling_edge, is_toggled.
Uppercase names (e.g. 'A', 'LB', 'DPAD_UP') route to gamepad; lowercase or mixed route to keyboard.
Each function accepts multiple names as varargs, returning the logical OR across them (except is_toggled remains single-key).
Re-exports get_axis from gamepad for axis polling.
"""

import keybrd
import gamepad
from gamepad import get_axis


def _is_gamepad(name: str) -> bool:
    """ Return True if the name refers to a gamepad button (all uppercase). """
    return isinstance(name, str) and name.isupper()


def is_pressed(*names) -> bool:
    """ Return True if any of the given keys or buttons are currently held down.
    Usage: is_pressed('a', 'A') returns True if 'a' key or 'A' button is down. """
    for name in names:
        if _is_gamepad(name):
            if gamepad.is_pressed(name):
                return True
        else:
            if keybrd.is_pressed(name):
                return True
    return False


def rising_edge(*names) -> bool:
    """ Return True once if any of the given keys or buttons went down since last check. """
    for name in names:
        if _is_gamepad(name):
            if gamepad.rising_edge(name):
                return True
        else:
            if keybrd.rising_edge(name):
                return True
    return False


def falling_edge(*names) -> bool:
    """ Return True once if any of the given keys or buttons went up since last check. """
    for name in names:
        if _is_gamepad(name):
            if gamepad.falling_edge(name):
                return True
        else:
            if keybrd.falling_edge(name):
                return True
    return False


def is_toggled(name: str) -> bool:
    """ Flip-flop state for a single key or button. Returns current toggle state. """
    if _is_gamepad(name):
        return gamepad.is_toggled(name)
    return keybrd.is_toggled(name)

if __name__ == "__main__":
    import time
    print("Testing unified input manager. Ctrl+C to quit.")
    try:
        while True:
            # keyboard example
            if is_pressed('a', 'A'):
                print("A is held down")
            if rising_edge('b', 'B'):
                print("B is pressed")
            if falling_edge('b', 'B'):
                print("B is released")

            # toggle still single-name
            if is_toggled('X'):
                print("X is toggled")
            if is_toggled('q'):
                print("Q is toggled")
            
            # trigger example
            lt = get_axis("RT")
            if lt:
                print(f"Left trigger: {lt:.2f}")
            
            # joystick example
            left_stick = (get_axis("LX"), get_axis("LY"))
            if left_stick != (0, 0):
                print(f"Left stick: {left_stick}")

            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Goodbye")
