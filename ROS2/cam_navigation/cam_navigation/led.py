import Jetson.GPIO as GPIO
import threading
import time

def run_led_sequence_nonblocking(sequence):
    # Static variables for thread and pin map
    if not hasattr(run_led_sequence_nonblocking, "active_thread"):
        run_led_sequence_nonblocking.active_thread = None
    if not hasattr(run_led_sequence_nonblocking, "pin_map"):
        run_led_sequence_nonblocking.pin_map = {}

    # Reject if sequence is already running
    if run_led_sequence_nonblocking.active_thread is not None and \
       run_led_sequence_nonblocking.active_thread.is_alive():
        print("LED sequence already in progress.")
        return None

    def run_sequence():
        try:
            GPIO.setmode(GPIO.BOARD)
            for item in sequence:
                if isinstance(item, tuple) and len(item) == 2:
                    pin, state = item
                    # Initialize pin if not already stored
                    if pin not in run_led_sequence_nonblocking.pin_map:
                        GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
                        run_led_sequence_nonblocking.pin_map[pin] = True
                    GPIO.output(pin, GPIO.HIGH if state else GPIO.LOW)
                elif isinstance(item, (int, float)):
                    time.sleep(item / 1000.0)  # assume ms to seconds
        finally:
            print("LED sequence complete.")

    # Launch the sequence thread
    t = threading.Thread(target=run_sequence)
    t.start()
    run_led_sequence_nonblocking.active_thread = t
    return t

ll_pin = 15
rl_pin = 7
loop_n = 10
sequences = {
    "left_turn":
    [
        (ll_pin, True),
        500,
        (ll_pin, False),
        500
    ],
    "right_turn":
    [
        (rl_pin, True),
        500,
        (rl_pin, False),
        500
    ],
}
sequences["left_turn"] *= loop_n
sequences["right_turn"] *= loop_n

# Optional test/demo
if __name__ == "__main__":
    # Example
    run_led_sequence_nonblocking(sequences["right_turn"])
