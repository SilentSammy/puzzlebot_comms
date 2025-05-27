import Jetson.GPIO as GPIO
import threading
import time

# Define melodies with frequencies and durations
melodies = {
    "super_mario_level_complete": [
        (659, 150),  # E5
        (784, 150),  # G5
        (1319, 150), # E6
        (1047, 150), # C6
        (1175, 150), # D6
        (1568, 300), # G6
    ],
    "zelda_secret_unlocked": [
        (392, 150),  # G4
        (523, 150),  # C5
        (659, 150),  # E5
        (784, 150),  # G5
        (1047, 300), # C6
    ],
    "windows_xp_logon": [
        (587, 200),  # D5
        (784, 200),  # G5
        (740, 200),  # F#5
        (880, 400),  # A5
    ],
    "star_wars_victory": [
        (392, 300),  # G4
        (523, 300),  # C5
        (659, 300),  # E5
        (784, 600),  # G5
    ],
    "custom_success_chime": [
        (440, 200),  # A4
        (523, 200),  # C5
        (659, 200),  # E5
        (880, 400),  # A5
        (659, 200),  # E5
        (880, 800),  # A5
    ],
    "rising_tone": [
        (440, 150),  # A4
        (523, 150),  # C5
        (587, 150),  # D5
        (659, 150),  # E5
        (784, 150),  # G5
    ],
    "falling_tone": [
        (587, 150),  # D5
        (523, 150),  # C5
        (440, 150),  # A4
        (392, 150),  # G4
        (349, 150),  # F4
    ],
    "3_highs": [
        (587, 125),
        (400, 25),
        (587, 125),
        (400, 25),
        (587, 125),
    ],
    "3_lows": [
        (440, 125),
        (587, 25),
        (440, 125),
        (587, 25),
        (440, 125),
    ],
}

# Adjust durations by speed factor
speed = 1.5  # Speed factor for melody playback
for melody in melodies.values():
    for i in range(len(melody)):
        freq, duration = melody[i]
        melody[i] = (freq, duration / speed)

def play_melody_nonblocking(melody):
    # Check if a melody is already playing.
    if hasattr(play_melody_nonblocking, "active_thread") and \
       play_melody_nonblocking.active_thread is not None and \
       play_melody_nonblocking.active_thread.is_alive():
        print("Melody is already playing.")
        return None

    buzzer_pin = 32
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(buzzer_pin, GPIO.OUT)
    pwm = GPIO.PWM(buzzer_pin, 440)  # Initialized with default freq

    def play():
        try:
            pwm.start(50)  # 50% duty cycle
            for freq, duration in melody:
                pwm.ChangeFrequency(freq)
                time.sleep(duration / 1000.0)
        finally:
            pwm.stop()
            GPIO.cleanup(buzzer_pin)

    t = threading.Thread(target=play)
    t.start()
    play_melody_nonblocking.active_thread = t
    return t

if __name__ == "__main__":
    melody = melodies["custom_success_chime"]
    play_melody_nonblocking(melody)
