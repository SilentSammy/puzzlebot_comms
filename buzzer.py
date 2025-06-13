melodies = {
    "beep_beep": [
        (392, 300),  # G4
        (0, 100),    # pause
        (392, 300),  # G4
    ],
    "beep_beep_high": [
        (523, 200),  # C5, higher pitch, faster
        (0, 80),     # pause
        (523, 200),  # C5
    ],
    "beep_beep_low": [
        (294, 400),  # D4, lower pitch, slower
        (0, 150),    # pause
        (294, 400),  # D4
    ],
    "dixie_horn": [
        (740, 220),   # F#5 – strong opening note
        (0,   80),   # rest
        (622, 180),   # D#5
        (0,   100),   # rest
        (494, 160),   # B4
        (0,    80),   # short rest
        (494, 160),   # B4
        (0,    80),   # short rest
        (494, 160),   # B4
        (0,   120),   # rest
        (554, 180),   # C#5
        (0,   100),   # rest
        (622, 180),   # D#5
        (0,   100),   # rest
        (659, 200),   # E5 – slightly held
        (0,    80),   # short rest
        (740, 220),   # F#5 – held again
        (0,    80),  
        (740, 220),   # F#5
        (0,    80),
        (740, 220),   # F#5 – longest final cluster
        (0,   180),   # longer closing rest
        (622, 200),   # D#5 – gentle finish
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
    "STOP": [(523, 60), (0, 200), (523, 60), (0, 200), (392, 80), (0, 300)],
    "YIELD": [ (659, 40), (0, 120), (784, 40), (0, 120), (659, 40), (0, 120), (784, 60), (0, 200) ],
    "ROAD_WORK": [
        (349, 30), (0, 100), (392, 30), (0, 100), (440, 30), (0, 100),
        (392, 30), (0, 100), (349, 30), (0, 100), (392, 30), (0, 100),
        (440, 30), (0, 100), (392, 30), (0, 200)
    ]

}

# Speed up all melodies by a factor
speed = 1.5
for key in melodies:
    melodies[key] = [(freq, int(duration / speed)) for freq, duration in melodies[key]]
