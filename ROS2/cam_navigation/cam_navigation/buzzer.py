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
}

# Adjust durations by speed factor
speed = 1.5  # Speed factor for melody playback
for melody in melodies.values():
    for i in range(len(melody)):
        freq, duration = melody[i]
        melody[i] = (freq, duration / speed)

def play_melody_nonblocking(melody):
    # Static variables
    if not hasattr(play_melody_nonblocking, "active_thread"):
        play_melody_nonblocking.active_thread = None
    if not hasattr(play_melody_nonblocking, "pwm"):
        buzzer_pin = 32
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(buzzer_pin, GPIO.OUT)
        pwm = GPIO.PWM(buzzer_pin, 440)  # Default freq, will be changed
        play_melody_nonblocking.pwm = pwm
        play_melody_nonblocking.buzzer_pin = buzzer_pin
        play_melody_nonblocking.pwm_started = False

    # If melody is already playing, reject the call
    if (play_melody_nonblocking.active_thread is not None and 
        play_melody_nonblocking.active_thread.is_alive()):
        print("Melody is already playing.")
        return None

    def play():
        try:
            pwm = play_melody_nonblocking.pwm
            if not play_melody_nonblocking.pwm_started:
                pwm.start(50)  # 50% duty cycle
                play_melody_nonblocking.pwm_started = True
            for freq, duration in melody:
                pwm.ChangeFrequency(freq)
                time.sleep(duration / 1000.0)
        finally:
            pwm.stop()
            play_melody_nonblocking.pwm_started = False
            GPIO.cleanup()

    t = threading.Thread(target=play)
    t.start()
    play_melody_nonblocking.active_thread = t
    return t

if __name__ == "__main__":
    melody = melodies["super_mario_level_complete"]
    play_melody_nonblocking(melody)
