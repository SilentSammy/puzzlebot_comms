melodies = {
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

# Speed up all melodies by a factor
speed = 1.5
for key in melodies:
    melodies[key] = [(freq, int(duration / speed)) for freq, duration in melodies[key]]
