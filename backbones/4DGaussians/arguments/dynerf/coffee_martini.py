_base_ = './default.py'

# Desynchronized setting: cam00 is the test view with all 300 frames; every training
# camera observes a 200-frame window [s, s+199] of the common timeline. Its time offset
# on the global [-1, 1] axis is (s - 50) * 2 / 299 (seed 777, scripts/data/build_unsync_scene.py).
ModelParams = dict(
    eval_index=0,
    gt_delta_time=[
        0.00000000000,  # cam00 (test view)
       -0.30100334448,  # cam01: [5, 204]
       +0.13377926421,  # cam02: [70, 269]
       +0.15384615385,  # cam04: [73, 272]
       -0.09364548495,  # cam05: [36, 235]
       -0.04682274247,  # cam06: [43, 242]
       +0.19397993311,  # cam07: [79, 278]
       +0.33444816054,  # cam08: [100, 299]
       -0.10702341137,  # cam09: [34, 233]
       -0.33444816054,  # cam10: [0, 199]
       +0.30769230769,  # cam11: [96, 295]
       -0.02006688963,  # cam12: [47, 246]
       +0.29431438127,  # cam13: [49, 248] (misaligned +45 frames)
       +0.12040133779,  # cam14: [68, 267]
       +0.23411371237,  # cam16: [85, 284]
       +0.04682274247,  # cam18: [57, 256]
       -0.14046822742,  # cam19: [29, 228]
       +0.17391304348,  # cam20: [76, 275]
    ],
)
