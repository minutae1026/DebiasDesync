_base_ = './default.py'

# Desynchronized setting: cam00 is the test view with all 300 frames; every training
# camera observes a 200-frame window [s, s+199] of the common timeline. Its time offset
# on the global [-1, 1] axis is (s - 50) * 2 / 299 (seed 777, scripts/data/build_unsync_scene.py).
ModelParams = dict(
    eval_index=0,
    gt_delta_time=[
        0.00000000000,  # cam00 (test view)
       -0.09364548495,  # cam01: [36, 235]
       +0.23411371237,  # cam02: [85, 284]
       +0.25418060201,  # cam03: [88, 287]
       -0.23411371237,  # cam04: [15, 214]
       -0.00668896321,  # cam05: [49, 248]
       +0.33444816054,  # cam06: [100, 299]
       -0.10702341137,  # cam07: [34, 233]
       -0.33444816054,  # cam08: [0, 199]
       -0.02006688963,  # cam09: [47, 246]
       +0.13377926421,  # cam10: [70, 269]
       -0.24080267559,  # cam11: [14, 213]
       -0.14046822742,  # cam12: [29, 228]
       +0.12040133779,  # cam13: [68, 267]
       +0.15384615385,  # cam14: [73, 272]
       +0.19397993311,  # cam15: [79, 278]
       -0.30100334448,  # cam16: [5, 204]
       +0.04682274247,  # cam17: [57, 256]
       +0.30769230769,  # cam18: [96, 295]
       +0.17391304348,  # cam19: [76, 275]
       -0.04682274247,  # cam20: [43, 242]
    ],
)
