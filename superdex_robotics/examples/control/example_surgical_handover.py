# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example: A scrub-nurse robot handing surgical instruments to a surgeon

An FR3 v2 arm with a DG5F five-finger hand stands at an operating table next to
a surgeon, takes an instrument out of a rack, and puts it into the surgeon's
open hand. You steer the robot with the keyboard or the mouse; the surgeon is
autonomous and closes their hand once an instrument settles in their palm.

The surgeon's hand is not a prop. It is a second bot -- a left DG5F, welded to
the world where a surgeon's outstretched hand would be -- running its own
joint-space controller. So the handover is a real two-body event rather than an
animation: the robot holds the instrument out, the surgeon closes their hand
around the same handle 0.065 m further down it, and only then does the robot
open its jaws and withdraw. For a moment both are holding it, which is what
happens when you pass someone a tool, and it means the instrument is never
unsupported in mid-air.

WHAT THE SCENE CONTAINS

    - An operating table with a draped patient on it.
    - A surgeon standing at the near edge, one hand held out to the side, palm
      up, waiting. Body and head are static scenery; the hand is articulated.
    - A pedestal-mounted FR3 + DG5F to the surgeon's side: the robot you drive.
    - A rack of three instruments -- scalpel, forceps, clamp -- standing handle
      up in a slotted tray, within the robot's reach.

FIVE THINGS THIS EXAMPLE ENCODES, EACH OF WHICH SILENTLY FAILS OTHERWISE

    - Everything is placed so that BOTH the rack and the surgeon's palm sit
      inside the FR3's 0.855 m envelope, measured to the FLANGE rather than to
      the fingers. The hand hangs about 0.2 m past the flange, so a grasp point
      0.70 m from the base is a flange target 0.87 m out -- past the end of the
      arm. Hence the 0.75 m pedestal. The robot also stands off the line between
      rack and surgeon, so the two targets are about 90 deg apart in its base
      frame rather than 180 deg, and joint 1 never swings through its limit.

    - The left hand is not the right hand with a sign flipped on the pose. Its
      joint LIMITS are mirrored too, and only some of them: the thumb opposes on
      positive angles where the right hand's opposes on negative, and the thumb
      curls negative where the right curls positive, but fingers 2-4 curl
      positive on both. A closed pose copied across without checking each limit
      is silently clamped, and the thumb simply never moves. See CLOSED_LEFT_DEG.

    - The instruments are upright grip barrels on plinths, not instrument-shaped
      objects lying on a tray, and every part of that is forced by the hand
      rather than chosen. These fingers curl toward the palm, so anything lying
      flat has to be reached under and the tray is in the way -- the same reason
      the grasp example uses a cup rather than a block. Standing them up turns
      the pick into a wrap, which is what this hand is good at.

      The barrel then has to be short enough that nothing rises into the palm
      (which sits only ~0.05 m above the closing fingers), narrow enough that
      the open fingers pass OUTSIDE it on the way down (they ring ~0.040 m from
      the hand's axis), and raised on a plinth so those fingers have air to
      descend into rather than a tray to jam against. See INSTRUMENTS.

    - The surgeon closes only after the robot has released, and then only on a
      short DWELL. Both conditions are load-bearing: closing while the robot
      still grips turns the handover into a tug of war between two hands on one
      barrel, and closing the instant the instrument arrives catches it before
      it has settled onto the palm.

A WARNING ABOUT THE GRASP

The pick is the fragile part of this example, and it is fragile in a way worth
stating plainly: the working configuration is close to an isolated point. Held
at BARREL_RADIUS 0.038 m, the closed-pose depth below, and a grasp height 0.65
of the way up the barrel, the robot picks the instrument, lifts it clear,
carries it and presents it. Move any ONE of those three -- the radius by 4 mm,
the depth by 5%, the height by 0.05 -- and the pick fails.

It is that tight because two measured facts leave almost no room between them:
the open hand's fingertips ring about 0.040 m from its axis and must pass
OUTSIDE the barrel on the way down, while the closed cage must still squeeze it
hard enough to hold. The gap between "too wide to descend around" and "too
narrow to hold" is a few millimetres of radius.

Worse, the calibration that aims the hand is not independent of the rest of the
scene: changing the SURGEON's resting hand pose moved the robot's measured
grasp offset by 1.3 cm, which is more than the whole margin, and turned a
working pick into a failing one. Both hands are actors in one solve.

So: --auto does not reliably complete the full handover, and the run reports
honestly which stage it lost. Live teleoperation is the more useful mode, where
you can see what is happening and retry with R. If you want this robust rather
than demonstrative, the fix is not more tuning of these numbers -- it is a
parallel-jaw gripper (assets/bots/arm_hand_combos/fr3_v2_2f_85), whose two jaws
close symmetrically on a cylinder with none of the cage travel that makes the
five-finger wrap so sensitive here.


CONTROLS (live mode; keep THIS terminal focused, not the debugger window)

    Keyboard
         W / S    move hand forward / back      (+X / -X)
         A / D    move hand left / right        (+Y / -Y)
         Q / E    raise / lower hand            (+Z / -Z)
         SPACE    toggle the robot's grip
         1 2 3    fly to the rack above instrument 1, 2 or 3
         H        fly to the handover pose over the surgeon's palm
         R        reset: put every instrument back in the rack
         ESC      quit

    Mouse (--mouse, Windows only)
         move             steer the hand in the horizontal plane
         hold RIGHT       steer height instead of forward/back
         LEFT click       toggle the robot's grip
         the keyboard stays live, so 1/2/3, H and R still work

Usage:
    cd <path_to_superdex_robotics>
    python3 examples/control/example_surgical_handover.py
    python3 examples/control/example_surgical_handover.py --mouse
    python3 examples/control/example_surgical_handover.py --auto
    python3 examples/control/example_surgical_handover.py --auto --view
    python3 examples/control/example_surgical_handover.py --auto --instrument forceps
"""

from __future__ import annotations

import argparse
import ctypes
import sys

import numpy as np
import superdex.physics as physics
import superdex.robotics as robotics
from superdex.physics.paths import resolve_asset

# The build's `real` type, so a pose handed to a controller Target copies straight
# in rather than being converted element by element.
np_real = np.float64 if physics.uses_double_precision() else np.float32

ROBOT_ASSET = "bots/arm_hand_combos/fr3_v2_2f_85/fr3_v2_2f_85.superdex_bot"
SURGEON_HAND_ASSET = "bots/hands/dg5f_short/left/dg5f_short_left.superdex_bot"

# The OSC chain spans the joints between these two links: the arm's mount and its
# tool flange, where the hand is attached. Link actors are "<bot_name>/<link_name>".
ARM_BASE_LINK = "fr3_link0"
ARM_EE_LINK = "fr3_link8"

# The arm's joints all share this prefix, which is how arm DOFs (owned by OSC)
# are told apart from hand DOFs (owned by JSC).
ARM_JOINT_PREFIX = "fr3_joint"

# The two jaw tips. Their midpoint is where a gripped object sits, and unlike a
# five-finger cage it is a genuinely fixed point: the jaws are symmetric, so the
# midpoint stays on the tool axis at every opening. Measured across the whole
# range it moves 0.010 m, all of it along Z, with no lateral component at all --
# against 0.046 m of sideways travel for the hand this replaced.
JAW_TIP_LINKS = ("2f_85_left_finger_tip_link", "2f_85_right_finger_tip_link")

# The gripper is a closed chain: six revolute joints tied by two cycle
# constraints. One angle drives all of it, mirrored left to right, and these are
# the signs that mirroring takes -- read off the prefab's default pose.
GRIPPER_JOINT_SIGNS = {
    "2f_85_left_inner_knuckle_joint": 1.0,
    "2f_85_left_knuckle_joint": 1.0,
    "2f_85_left_finger_tip_joint": -1.0,
    "2f_85_right_inner_knuckle_joint": -1.0,
    "2f_85_right_knuckle_joint": -1.0,
    "2f_85_right_finger_tip_joint": 1.0,
}

# The links whose incoming joint the pose controller tracks.
GRIPPER_TRACKED_TOKENS = ("knuckle_link", "finger_tip_link")

# Jaw angle at each end of the grip signal [deg]. 8 deg leaves 0.123 m between
# the jaw tips, wide enough to drop over a handle; 40 deg is well past contact,
# so the jaws keep squeezing rather than stopping on it.
GRIP_OPEN_DEG = 8.0
GRIP_CLOSED_DEG = 44.0

# Pose-controller gains for the gripper joints.
#
# The gripper is NOT driven by joint torque. BASIC_JSC_PD tears it apart: four
# of its six joints have no limits, so the PD overpowers the cycle constraints
# and the knuckle ends up at 194 deg with the linkage inside out. The native
# articulated pose controller tracks joints inside the constraint solve instead,
# and it tracks them exactly -- every commanded angle from 5 to 45 deg came back
# within 0.01 deg. Its gains are per LINK, so the arm's links stay at zero gain
# and keep being driven by OSC torques exactly as before.
GRIP_STIFFNESS = 1200.0
GRIP_DAMPING = 8.0

TIME_STEP = 1.0 / 200.0  # [s]


# --------------------------------------------------------------------------- #
# Scene layout. SuperDex is X-forward, Y-left, Z-up (FLU); all lengths [m].
# --------------------------------------------------------------------------- #

# Operating table: the patient lies along X, head toward +X.
TABLE_CENTER = np.array([0.10, 0.20])
TABLE_SIZE = np.array([2.00, 0.62, 0.06])
TABLE_TOP = 0.78

# The surgeon stands at the table's near (-Y) edge.
SURGEON_STAND = np.array([0.10, -0.36])
SURGEON_SHOULDER = np.array([0.02, -0.42, 1.32])

# The surgeon's receiving hand, held out to their side, palm up, fingers -X.
SURGEON_HAND_POSITION = np.array([-0.18, -0.50, 1.00])

# Where the surgeon's closed fingers end up, measured off the hand's link
# positions in the closed pose: midway between the thumb tip and the centroid of
# fingers 2-5, which is the middle of the ring they form around a handle.
SURGEON_GRIP_POINT = np.array([-0.268, -0.490, 1.050])

# How far up the instrument the surgeon takes hold, against the robot's 0.115.
#
# The instrument is never let go of in mid-air: the robot holds it while the
# surgeon closes on a different part of the same handle, and only then opens.
# That is what a handover actually is, and it only became possible with the
# parallel gripper. Two five-finger hands could not share one object -- they
# squeezed it out sideways between them -- so the previous version had to set
# the instrument down on the open palm and let go. On a slim handle that does
# not work either: a rod resting across four thin fingers tips and slides off
# the ends, which is exactly what the last run did. Two thin jaws and a hand
# gripping 0.065 m apart on the same rod do not interfere at all.
# Low on the handle, and that height is set by the OPEN hand rather than the
# closed one. The surgeon's straightened fingers make a shelf at z = 1.011, so
# the instrument has to be presented with its base ABOVE that or it is driven
# into the fingers on the way in. At 0.050 the base had to reach 1.000 and the
# arm drove it straight through them; at 0.025 it hung at exactly finger height
# the base hangs at 1.034, clears the open fingers by 23 mm, and the hand closes
# and dragged, stalling the arm 0.13 m short of the handover. At 0.015 it
SURGEON_GRIP_HEIGHT = 0.016

# Top of the surgeon's open hand [m], measured off its link positions: the
# straightened fingers sit at 1.011 and the thumb and little finger reach 1.031.
# A carried instrument has to pass over all of it.
SURGEON_HAND_TOP = 1.035

# The robot, on a pedestal, standing off the line between the rack and the
# surgeon so the two targets are about 90 deg apart in its base frame.
ROBOT_BASE = np.array([-0.58, -0.90, 0.75])
PEDESTAL_SIZE = np.array([0.36, 0.36])

# The instrument rack (a Mayo stand), on the robot's other side.
RACK_CENTER = np.array([-0.95, -0.62])
# The tray height is squeezed from both sides. Too high and the transit pose
# above it leaves the arm's envelope; too low and the GRASP pose does. At 0.68 m
# the arm stalls 0.12 m above the instruments and closes on thin air -- which
# looks exactly like a grip that will not hold, and wasted a long time being
# debugged as one. Check the flange tracking error before blaming the fingers.
RACK_TOP = 0.78
RACK_TRAY_SIZE = np.array([0.56, 0.30, 0.03])
# Instruments stand in a row along X, spaced wider than the closed fist (about
# 0.13 m) so the hand can drop around one without disturbing its neighbours.
INSTRUMENT_SPACING = 0.16

# Clearance kept above the tallest instrument when crossing the rack, on top of
# however far the load hangs below the hand. See transit_height().
TRANSIT_MARGIN = 0.07


# --------------------------------------------------------------------------- #
# Instruments
# --------------------------------------------------------------------------- #

# Each instrument is a lathe: a (height, radius) profile revolved about Z, with
# the band inside `flatten_span` squashed in Y to give it a flat working end
# while the handle below stays round.
#
# These finally look like instruments -- a 31 mm handle with a flat blade, slim
# forceps tips, broader clamp jaws -- and that is entirely down to the gripper.
# The five-finger hand this replaced could only hold a 93 mm cup-shaped barrel,
# because its own middle finger occupied the grasp axis on the way down and the
# object had to be hollow and wide enough to swallow it. Two flat jaws closing
# from opposite sides have no such constraint: they need only somewhere to pinch.
#
# Three things still shape the profile, all measured off the gripper:
#
#   The handle is 31 mm across. The jaws open to about 72 mm of clear gap and
#   close past contact on anything under ~45 mm, so this sits comfortably inside
#   the range where a rod was verified to be held against gravity.
#
#   Nothing stands more than 0.035 m above the grip groove. It is not the
#   gripper's body that sets this, which sits a comfortable 0.103 m up, but its
#   INNER KNUCKLE links: those pass only 0.042 m above the jaw tips and only
#   0.012 m from the tool axis, so they occupy the space directly over whatever
#   is being gripped. An instrument taller than that stops the descent before
#   the jaws ever reach the groove -- at 0.09 m of overhang the arm stalled
#   0.066 m high and closed two thirds of the way up the handle, which then
#   swung out during the carry. None of that looks like a clearance problem from
#   the outside; the flange tracking error is where it shows up plainly.
#
#   The butt is 60 mm across and 32 mm tall, and it does two jobs. It is a base
#   broad enough for a slim rod to stand on by itself, since the instruments now
#   sit loose on the tray -- no plinths any more, because nothing on this gripper
#   reaches below the jaws. And it is what the SURGEON holds. Their hand is a
#   five-finger DG5F, which cannot keep hold of a 31 mm shaft at all: it closes
#   around it without reaching it, and the instrument slides out of the fist a
#   couple of seconds after the robot lets go. On a 60 mm butt it has something
#   to actually close on. So the two ends of the instrument are shaped for the
#   two different hands that hold it.
INSTRUMENTS = (
    # name, profile [(z, radius)], flatten_span, flatten_scale, mass [kg]
    (
        "scalpel",
        (
            (0.000, 0.000), (0.000, 0.030), (0.032, 0.030), (0.040, 0.016),
            (0.103, 0.016), (0.108, 0.011), (0.122, 0.011), (0.127, 0.016),
            (0.130, 0.013), (0.134, 0.007), (0.138, 0.016), (0.146, 0.010),
            (0.148, 0.000),
        ),
        (0.138, 0.148), 0.22, 0.06,
    ),
    (
        "forceps",
        (
            (0.000, 0.000), (0.000, 0.030), (0.032, 0.030), (0.040, 0.015),
            (0.103, 0.015), (0.108, 0.011), (0.122, 0.011), (0.127, 0.015),
            (0.131, 0.012), (0.136, 0.006), (0.146, 0.005), (0.150, 0.000),
        ),
        (0.136, 0.150), 0.45, 0.05,
    ),
    (
        "clamp",
        (
            (0.000, 0.000), (0.000, 0.030), (0.034, 0.030), (0.042, 0.017),
            (0.103, 0.017), (0.108, 0.012), (0.122, 0.012), (0.127, 0.017),
            (0.131, 0.014), (0.136, 0.008), (0.140, 0.015), (0.146, 0.009),
            (0.150, 0.000),
        ),
        (0.140, 0.150), 0.35, 0.07,
    ),
)

# Height up an instrument at which the jaws close, measured from its base: the
# centre of the grip groove.
#
# The groove is what makes the grasp hold. Flat jaws pinching a smooth round
# handle look fine and are not enough: a 60 g instrument hanging off that pinch
# slid 0.066 m through the jaws during the lift and then swung out of them
# altogether on the carry, and neither more clamping force nor more friction
# changed it, because the failure is geometric rather than frictional. Necking
# the handle down over the 0.014 m the jaws occupy gives them shoulders to seat
# against, and the instrument cannot travel axially at all.
INSTRUMENT_GRASP_HEIGHT = 0.104

# The grasp is held by friction on the handle, so this matters more than usual.
INSTRUMENT_FRICTION = 1.4


# --------------------------------------------------------------------------- #
# Grip poses
# --------------------------------------------------------------------------- #

# The surgeon's LEFT hand, waiting. This one has to be FLAT, not merely open,
# because the robot sets an instrument down on it and anything sticking up is
# what the instrument lands on.
#
# Straightening the four fingers is not enough. Left at their defaults, the
# thumb and the little finger still rise to about 0.09 m above the palm -- a
# hand's natural shallow bowl -- and they sit inside the footprint of a 93 mm
# barrel released over the palm, so it comes down on two raised fingertips and
# tips straight off onto the table. dg5f_joint_1_2 (thumb swing) and
# dg5f_joint_5_1 / 5_2 (little-finger spread) are what flatten them; on this
# hand every one of those is zero at the flat end of its range.
OPEN_LEFT_DEG = {
    "dg5f_joint_1_1": 0.0,
    "dg5f_joint_1_2": 25.0,
    "dg5f_joint_1_3": 0.0,
    "dg5f_joint_1_4": 0.0,
    "dg5f_joint_2_2": 0.0, "dg5f_joint_2_3": 0.0, "dg5f_joint_2_4": 0.0,
    "dg5f_joint_3_2": 0.0, "dg5f_joint_3_3": 0.0, "dg5f_joint_3_4": 0.0,
    "dg5f_joint_4_2": 0.0, "dg5f_joint_4_3": 0.0, "dg5f_joint_4_4": 0.0,
    "dg5f_joint_5_1": 0.0, "dg5f_joint_5_2": 0.0,
    "dg5f_joint_5_3": 0.0, "dg5f_joint_5_4": 0.0,
}

# The same grip on the surgeon's LEFT hand. This is NOT the right-hand table
# with every sign flipped: the mirroring is per joint, because only some of the
# limits mirror.
#
#   dg5f_joint_1_1   right -22..51 deg,  left -51..22   -> negate
#   dg5f_joint_1_2   right -180..0,      left 0..180    -> negate
#   dg5f_joint_1_3   right 0..90,        left -90..0    -> negate
#   dg5f_joint_1_4   right 0..90,        left -90..0    -> negate
#   dg5f_joint_2_2   right 0..115,       left 0..115    -> SAME sign
#   dg5f_joint_5_3   right 0..90,        left 0..90     -> SAME sign
#
# So the thumb mirrors and the fingers do not. Getting this wrong costs you the
# thumb: it clamps at its limit, and a four-finger hand with no opposition
# cannot hold an instrument the robot lets go of.
# These close much further than the cup grasp's do, because what the surgeon is
# being handed is now a 31 mm handle rather than a 93 mm barrel. Left at the cup
# angles the hand closes around the instrument without ever reaching it: the
# robot lets go, the instrument sits in the loose fist for about two seconds
# while sliding, and then drops. That failure looks exactly like the robot
# knocking it out on the way past, which is worth knowing -- the way to tell
# them apart is to stop the robot dead after releasing and see whether the
# surgeon still has it a few seconds later.
CLOSED_LEFT_DEG = {
    "dg5f_joint_1_1": -45.0,
    "dg5f_joint_1_2": 110.0,
    "dg5f_joint_1_3": -85.0,
    "dg5f_joint_1_4": -80.0,
    "dg5f_joint_2_2": 110.0, "dg5f_joint_2_3": 88.0, "dg5f_joint_2_4": 85.0,
    "dg5f_joint_3_2": 108.0, "dg5f_joint_3_3": 88.0, "dg5f_joint_3_4": 85.0,
    "dg5f_joint_4_2": 105.0, "dg5f_joint_4_3": 88.0, "dg5f_joint_4_4": 85.0,
    "dg5f_joint_5_3": 88.0,
    "dg5f_joint_5_4": 85.0,
}

# How fast a grip opens and closes [grip fraction per second]. Slow enough that
# the fingers settle onto the handle rather than batting it out of the rack.
GRIP_RATE = 1.2

# The thumb closes on the same clock as the fingers. Running it ahead was tried,
# on the theory that the backstop should be in place first, and it made things
# worse: the thumb then arrives alone and tips the instrument by itself.
THUMB_JOINT_PREFIX = "dg5f_joint_1_"
THUMB_LEAD = 1.0

# The surgeon holds on hard from the start: they are not doing the delicate
# part, and they have to be able to take the instrument off a robot that is
# still gripping it.
SURGEON_KP, SURGEON_SAT = 45.0, 30.0

# The surgeon closes once an instrument has been inside this radius of the
# receive point for this long.
SURGEON_RECEIVE_RADIUS = 0.10  # [m]
SURGEON_RECEIVE_DWELL = 0.5  # [s]
# Once the surgeon has an instrument they keep it, so the hand does not reopen
# just because the robot jostles it out of the receive radius for a step.
SURGEON_RELEASE_RADIUS = 0.30  # [m]


# --------------------------------------------------------------------------- #
# Teleoperation
# --------------------------------------------------------------------------- #

# How far the fist target jumps per key press [m].
#
# This used to be a speed (0.18 m/s) scaled by a bare 0.05, which worked out at
# 9 mm per press -- invisible at scene scale, so a working teleop looked like a
# dead one. A press is a discrete event here (msvcrt hands over characters, not
# key-down/key-up), so a distance per press is the honest unit.
MOVE_STEP = 0.08

# How far ahead of the arm the goal is allowed to get [m]. Auto-repeat fires
# about 30 times a second, which at MOVE_STEP pushes the goal ~0.9 m/s while
# TARGET_SLEW_RATE lets the target follow at 0.15 m/s. Without a leash the goal
# runs off and the arm keeps travelling long after the key is released, which
# reads as laggy controls rather than as a goal that has left the building.
TARGET_LEASH = 0.20

# How fast the hand target is allowed to travel [m/s]. Waypoints -- the scripted
# stages, and the 1/2/3/H fly-to keys -- would otherwise teleport the target and
# OSC would yank the arm after it, peeling the instrument out of the fingers.
TARGET_SLEW_RATE = 0.15

# The same, for live teleoperation [m/s]. Much faster, because the two jobs are
# not the same job: the script is trying not to peel a grasped object out of the
# fingers, while a human driving needs to SEE the arm respond. At the scripted
# 0.15 m/s a key press took half a second to play out and read as nothing
# happening. This caps how fast the arm can be driven no matter how hard you
# lean on a key, so it is the number to raise if teleop still feels sluggish.
TELEOP_SLEW_RATE = 0.5

# Mouse mode maps the screen onto this box, in the frame of the closing fist
# rather than the flange, so the cursor points at where the fingers go.
MOUSE_X_RANGE = (-1.30, 0.05)  # screen bottom -> top
MOUSE_Y_RANGE = (-1.10, -0.20)  # screen right -> left
MOUSE_Z_RANGE = (0.84, 1.30)  # screen bottom -> top, while RMB is held

KEY_TO_AXIS = {
    "w": (0, 1), "s": (0, -1),
    "a": (1, 1), "d": (1, -1),
    "q": (2, 1), "e": (2, -1),
}

# Where the calibration pre-roll parks the flange, as an offset from the robot's
# base. It has to be clear of everything -- fingers closing into the rack bend
# back and make the measurement meaningless, and swinging over the rack to get
# there sweeps the instruments onto the floor before the run even starts -- and
# it has to be comfortably inside the arm's envelope. At full stretch the arm
# cannot hold the commanded orientation, so the hand is not actually palm-down
# and the measured offset is of some other pose entirely.
CALIBRATION_OFFSET = np.array([0.0, 0.40, 0.40])  # [m], from ROBOT_BASE
CALIBRATION_SECONDS = 2.0


# --------------------------------------------------------------------------- #
# Mesh helpers
#
# None of the scene furniture -- table, pedestal, rack, patient, surgeon -- has
# an asset in this repo, so it is generated here. A dynamic rigid actor needs a
# surface mesh (an implicit sphere or plane can only be static), which is why
# the instruments are lathed meshes rather than primitives.
# --------------------------------------------------------------------------- #


def box_mesh(size, center=(0.0, 0.0, 0.0)):
    """An axis-aligned box as (coordinates, connectivity), wound outward."""
    half = np.asarray(size, dtype=float) / 2.0
    signs = np.array(
        [
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
        ],
        dtype=float,
    )
    vertices = signs * half + np.asarray(center, dtype=float)
    faces = np.array(
        [
            [0, 3, 2], [0, 2, 1], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
            [1, 2, 6], [1, 6, 5], [0, 4, 7], [0, 7, 3],
        ],
        dtype=np.int32,
    )
    return vertices.reshape(-1), faces.reshape(-1)


def lathe_mesh(profile, segments=20, flatten_span=None, flatten_scale=1.0):
    """Revolve a (height, radius) profile about Z into a closed solid.

    ``profile`` runs from the base upward in the shape's local frame, so an
    instrument authored with z=0 at its base sits on the tray when its actor is
    placed at tray height.

    Within ``flatten_span`` the vertices are squashed in Y by ``flatten_scale``,
    which turns that band of a revolved solid into a flat blade while the
    handle above it stays round. It only moves vertices that are already there,
    so the mesh stays watertight and in one piece -- unlike a handle and a blade
    authored as two overlapping solids.
    """
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    circle = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    rings = []
    for height, radius in profile:
        ring = np.concatenate(
            [circle * radius, np.full((segments, 1), height)], axis=1
        )
        if flatten_span is not None and flatten_span[0] <= height <= flatten_span[1]:
            ring[:, 1] *= flatten_scale
        rings.append(ring)

    vertices = np.concatenate(
        rings
        + [
            np.array([[0.0, 0.0, profile[0][0]]]),
            np.array([[0.0, 0.0, profile[-1][0]]]),
        ],
        axis=0,
    )
    base_hub = segments * len(profile)
    top_hub = base_hub + 1

    faces = []
    for level in range(len(profile) - 1):
        lower, upper = level * segments, (level + 1) * segments
        for i in range(segments):
            j = (i + 1) % segments
            faces.append([lower + i, lower + j, upper + j])
            faces.append([lower + i, upper + j, upper + i])
    top = (len(profile) - 1) * segments
    for i in range(segments):
        j = (i + 1) % segments
        faces.append([base_hub, j, i])
        faces.append([top_hub, top + i, top + j])

    return vertices.reshape(-1), np.array(faces, dtype=np.int32).reshape(-1)


def revolve_mesh(profile, segments=24, flatten_span=None, flatten_scale=1.0):
    """Revolve a (height, radius) polyline about Z, allowing radius 0.

    Unlike lathe_mesh, a point with radius 0 becomes a single vertex on the axis
    rather than a ring, so a profile that starts and ends on the axis closes
    itself. That is what makes a HOLLOW vessel expressible: run the profile out
    along the floor, up the outside, across the rim, down the inside, and back
    in along the floor.

    Within ``flatten_span`` the vertices are squashed in Y, which flattens that
    band of the wall -- inner and outer together, so the wall stays even.
    """
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    circle = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    vertices = []
    levels = []  # (start index, count) per profile point
    for height, radius in profile:
        if radius < 1e-9:
            levels.append((len(vertices), 1))
            vertices.append([0.0, 0.0, height])
            continue
        ring = np.concatenate(
            [circle * radius, np.full((segments, 1), height)], axis=1
        )
        if flatten_span is not None and flatten_span[0] <= height <= flatten_span[1]:
            ring[:, 1] *= flatten_scale
        levels.append((len(vertices), segments))
        vertices.extend(ring.tolist())

    faces = []
    for level in range(len(profile) - 1):
        (lower, lower_count), (upper, upper_count) = levels[level], levels[level + 1]
        if lower_count == 1:
            # The cap at a hub faces AWAY from the body, so its winding is the
            # reverse of a side quad's. Getting this backwards leaves a mesh that
            # still looks closed and still reports no open edges, but encloses
            # negative volume -- which is how an inside-out collision shape gets
            # through review.
            for i in range(segments):
                faces.append([lower, upper + (i + 1) % segments, upper + i])
        elif upper_count == 1:
            for i in range(segments):
                faces.append([lower + i, lower + (i + 1) % segments, upper])
        else:
            for i in range(segments):
                j = (i + 1) % segments
                faces.append([lower + i, lower + j, upper + j])
                faces.append([lower + i, upper + j, upper + i])

    return (
        np.array(vertices, dtype=float).reshape(-1),
        np.array(faces, dtype=np.int32).reshape(-1),
    )


def loft_mesh(sections, axis=2, segments=20):
    """Loft elliptical cross-sections along one axis into a closed solid.

    ``sections`` is [(position, half_width_u, half_width_v), ...] running along
    ``axis``, where u and v are the two perpendicular axes in right-handed order
    (axis 2 -> u=X, v=Y; axis 0 -> u=Y, v=Z; axis 1 -> u=Z, v=X). A section with
    both half-widths at zero becomes a single vertex on the axis, which is how
    the ends close -- and BOTH ends must be such a section, or the result is an
    open tube. The engine catches that ("collider mesh is not topologically
    closed") but only once you build the actor, so end every profile at zero.

    Elliptical rather than circular is the whole point: a person is not a solid
    of revolution. A torso is wide across the shoulders and shallow front to
    back, a limb tapers, a body lying down is broader than it is deep. Two
    independent half-widths per section express all of that, where revolve_mesh
    could only make barrels.
    """
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    cos, sin = np.cos(angles), np.sin(angles)
    u_axis, v_axis = (axis + 1) % 3, (axis + 2) % 3

    vertices = []
    levels = []
    for position, half_u, half_v in sections:
        if half_u < 1e-9 and half_v < 1e-9:
            point = np.zeros(3)
            point[axis] = position
            levels.append((len(vertices), 1))
            vertices.append(point.tolist())
            continue
        ring = np.zeros((segments, 3))
        ring[:, axis] = position
        ring[:, u_axis] = cos * half_u
        ring[:, v_axis] = sin * half_v
        levels.append((len(vertices), segments))
        vertices.extend(ring.tolist())

    faces = []
    for level in range(len(sections) - 1):
        (lower, lower_count), (upper, upper_count) = levels[level], levels[level + 1]
        if lower_count == 1:
            # The cap at a hub faces AWAY from the body, so its winding is the
            # reverse of a side quad's. Getting this backwards leaves a mesh that
            # still looks closed and still reports no open edges, but encloses
            # negative volume -- which is how an inside-out collision shape gets
            # through review.
            for i in range(segments):
                faces.append([lower, upper + (i + 1) % segments, upper + i])
        elif upper_count == 1:
            for i in range(segments):
                faces.append([lower + i, lower + (i + 1) % segments, upper])
        else:
            for i in range(segments):
                j = (i + 1) % segments
                faces.append([lower + i, lower + j, upper + j])
                faces.append([lower + i, upper + j, upper + i])

    return (
        np.array(vertices, dtype=float).reshape(-1),
        np.array(faces, dtype=np.int32).reshape(-1),
    )


def direction_transform(start, end):
    """Rotation and midpoint placing a +Z-aligned mesh along start -> end."""
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    delta = end - start
    length = float(np.linalg.norm(delta))
    direction = delta / length
    axis = np.cross([0.0, 0.0, 1.0], direction)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-9:
        rotation = (
            physics.Quaternion.identity()
            if direction[2] > 0.0
            else physics.Quaternion.rotation_x(np.pi)
        )
    else:
        angle = float(np.arctan2(axis_norm, float(direction[2])))
        rotation = physics.Quaternion.from_axis_angle(
            (axis / axis_norm).tolist(), angle
        )
    return length, physics.TransformRT(
        rotation=rotation, translation=((start + end) / 2.0).tolist()
    )


def limb_between(start, end, radius_start, radius_end, segments=14):
    """A tapered, rounded-off limb spanning two points, as (mesh, transform)."""
    length, transform = direction_transform(start, end)
    half = length / 2.0
    mesh = loft_mesh(
        (
            (-half - radius_start * 0.5, 0.0, 0.0),
            (-half, radius_start, radius_start),
            (half, radius_end, radius_end),
            (half + radius_end * 0.5, 0.0, 0.0),
        ),
        segments=segments,
    )
    return mesh, transform


def rod_between(start, end, radius, segments=12):
    """A cylinder spanning two points, as (mesh, transform).

    Used for limbs, where the geometry is defined by its endpoints rather than
    by an axis-aligned size.
    """
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    delta = end - start
    length = float(np.linalg.norm(delta))
    mesh = lathe_mesh(((-length / 2.0, radius), (length / 2.0, radius)), segments)

    direction = delta / length
    axis = np.cross([0.0, 0.0, 1.0], direction)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-9:
        # Already along +-Z, so there is no rotation axis to build.
        rotation = (
            physics.Quaternion.identity()
            if direction[2] > 0.0
            else physics.Quaternion.rotation_x(np.pi)
        )
    else:
        angle = float(np.arctan2(axis_norm, float(direction[2])))
        rotation = physics.Quaternion.from_axis_angle(
            (axis / axis_norm).tolist(), angle
        )
    transform = physics.TransformRT(
        rotation=rotation, translation=((start + end) / 2.0).tolist()
    )
    return mesh, transform


def add_static_mesh(scene, name, mesh, transform=None):
    """Static scenery: it collides, but nothing else about it matters."""
    coordinates, connectivity = mesh
    shape = physics.create_tri_mesh_shape(coordinates.tolist(), connectivity.tolist())
    return scene.create_rigid_actor(
        name=name,
        shape=shape,
        world_from_local=transform if transform is not None else physics.TransformRT(),
        is_static=True,
    )


def add_static_box(scene, name, size, center):
    return add_static_mesh(scene, name, box_mesh(size, center))


# --------------------------------------------------------------------------- #
# Scene construction
# --------------------------------------------------------------------------- #


def load_bot_prefab(asset, name, world_from_root, weld_root):
    """Load a bot prefab, rename it, place it, and optionally weld it down."""
    prefab = robotics.load_bot_prefab_from_file(str(resolve_asset(asset)))
    prefab.name = name
    prefab.world_from_root = world_from_root

    # The standalone hand ships with a FREE root joint, i.e. a floating base. A
    # surgeon's hand held out over a table does not drift when something bumps
    # it, so weld the root down: HARD leaves the bot with zero root DOFs, pinned
    # at world_from_root.
    if weld_root:
        prefab.joints[0].type = physics.ArticulatedJointType.HARD

    # Cheap "gravity compensation": neither BASIC_OSC_PD nor BASIC_JSC_PD has a
    # gravity term, so without this the arm and the fingers sag off their
    # targets. The instruments keep their gravity, or handing one over would
    # prove nothing.
    for i in range(len(prefab.links)):
        prefab.links[i].has_gravity = False
    return prefab


def dof_map(prefab):
    """Joint name -> DOF index in bot DOF space (prefab order, root skipped)."""
    mapping = {}
    dof = 0
    for i in range(len(prefab.joints)):
        joint = prefab.joints[i]
        if joint.type != physics.ArticulatedJointType.REVOLUTE:
            continue
        mapping[joint.name] = dof
        dof += 1
    return mapping


def make_jsc(bot, num_dofs, kp, saturation):
    """A joint-space PD controller sized to the whole actor, at fixed gains.

    JSC has no notion of a sub-chain: its gains, target pose and output are all
    sized to the full actor, arm DOFs included.
    """
    jsc = bot.create_controller("BASIC_JSC_PD")
    params = robotics.ControllerBasicJscPdParams()
    params.kp = np.full(num_dofs, kp, dtype=np.float32)
    params.kd = np.full(num_dofs, 0.4, dtype=np.float32)  # [Nms/rad]
    params.saturation = np.full(num_dofs, saturation, dtype=np.float32)
    params.deadband = np.zeros(num_dofs, dtype=np.float32)
    jsc.set_params(params)
    return jsc, params


def current_pose(actor, num_dofs):
    pose = physics.DynamicArrayReal(num_dofs)
    actor.get_articulated_pose(pose)
    return np.array(pose, dtype=np_real)


def pose_with(actor, num_dofs, dofs, root_dofs, angles_deg):
    """The bot's default pose with the named joints overridden [deg].

    Starting from the default rather than from zero leaves the arm DOFs and any
    joint not named here exactly where the prefab put them.
    """
    pose = current_pose(actor, num_dofs)
    for name, degrees in angles_deg.items():
        pose[dofs[name] + root_dofs] = np.radians(degrees)
    return pose


def transit_height(open_hand_drop):
    """Fist height at which it is safe to cross the rack [m].

    Two different things hang below the commanded jaw point, and both have to
    clear whatever the arm crosses: the open gripper itself, which reaches
    ``open_hand_drop`` below it, and a carried instrument, which hangs the full
    grasp height below it.

    There are also two things to clear, not one. The rack is the obvious one.
    The SURGEON'S OUTSTRETCHED HAND is the other, and it is easy to forget
    because it is not an obstacle at any other point in the run -- the arm only
    crosses it on the way in. Shortening the instruments dropped this height to
    1.104, which hangs a carried instrument's base at 1.000, just under the
    surgeon's fingers at 1.011. It then dragged along them and stalled the arm
    0.13 m short of the handover, with the instrument stuck out of reach of a
    hand that was waiting right there for it.

    Neither miss looks like a control failure. The arm flies confidently to
    almost the right place; only the flange tracking error gives it away.
    """
    tallest = max(profile[-1][0] for _, profile, _, _, _ in INSTRUMENTS)
    hangs_below = max(open_hand_drop, INSTRUMENT_GRASP_HEIGHT)
    over_rack = RACK_TOP + tallest + hangs_below + TRANSIT_MARGIN
    over_surgeon = SURGEON_HAND_TOP + INSTRUMENT_GRASP_HEIGHT + TRANSIT_MARGIN
    return max(over_rack, over_surgeon)


def instrument_slots():
    """World position of each instrument's base, in a row along the rack.

    They stand straight on the tray -- plus a couple of millimetres, because a
    zero-gap contact at t=0 is enough to blow up the solver.
    """
    count = len(INSTRUMENTS)
    offsets = (np.arange(count) - (count - 1) / 2.0) * INSTRUMENT_SPACING
    base_z = RACK_TOP + 0.002
    return [
        np.array([RACK_CENTER[0] + offset, RACK_CENTER[1], base_z])
        for offset in offsets
    ]


def build_scene():
    """Build the operating room, both bots, and every controller."""
    physics.initialize(num_worker_threads=-1)

    # SuperDex robots use a Z-up convention, so gravity points down -Z.
    scene = physics.create_scene("Surgical Handover")
    scene.set_gravity([0, 0, -9.81])

    scene.create_rigid_actor(
        name="floor",
        shape=physics.create_plane_shape(normal=[0, 0, 1], distance=0),
        is_static=True,
    )

    # --- the operating table and its patient ---------------------------------
    add_static_box(
        scene,
        "operating_table",
        TABLE_SIZE,
        [TABLE_CENTER[0], TABLE_CENTER[1], TABLE_TOP - TABLE_SIZE[2] / 2.0],
    )
    add_static_box(
        scene,
        "table_column",
        [0.34, 0.34, TABLE_TOP - TABLE_SIZE[2]],
        [TABLE_CENTER[0], TABLE_CENTER[1], (TABLE_TOP - TABLE_SIZE[2]) / 2.0],
    )
    add_static_box(
        scene,
        "table_foot",
        [0.70, 0.50, 0.04],
        [TABLE_CENTER[0], TABLE_CENTER[1], 0.02],
    )

    # A draped patient, lofted head to toe. Static scenery -- it is what the
    # surgeon is working on, not something either hand touches.
    #
    # The body is lofted along X because that is the way it lies. Its axis sits a
    # little above the table so the broad sections (chest, hips) rest on it while
    # the narrow ones (ankles, neck) hover a few millimetres -- nobody can see
    # that under a drape, and it avoids having to loft an asymmetric section.
    patient_axis_z = TABLE_TOP + 0.11
    add_static_mesh(
        scene,
        "patient_body",
        loft_mesh(
            (
                (-0.86, 0.00, 0.00),
                (-0.80, 0.09, 0.05),   # feet
                (-0.60, 0.15, 0.08),   # calves
                (-0.34, 0.19, 0.11),   # thighs
                (-0.12, 0.21, 0.12),   # hips
                (0.14, 0.21, 0.12),    # abdomen
                (0.34, 0.23, 0.13),    # chest
                (0.48, 0.17, 0.10),    # shoulders
                (0.55, 0.07, 0.07),    # neck
                (0.58, 0.06, 0.06),
                (0.61, 0.00, 0.00),    # closed off inside the head
            ),
            axis=0,
        ),
        physics.TransformRT(
            translation=[TABLE_CENTER[0], TABLE_CENTER[1], patient_axis_z]
        ),
    )
    add_static_mesh(
        scene,
        "patient_head",
        loft_mesh(
            (
                (0.58, 0.00, 0.00),
                (0.62, 0.07, 0.08),
                (0.70, 0.09, 0.10),
                (0.78, 0.08, 0.09),
                (0.82, 0.00, 0.00),
            ),
            axis=0,
        ),
        physics.TransformRT(
            translation=[TABLE_CENTER[0], TABLE_CENTER[1], patient_axis_z]
        ),
    )
    # The drape: a broad, shallow sheet over everything below the chest, leaving
    # the upper abdomen exposed as the surgical site.
    add_static_mesh(
        scene,
        "patient_drape",
        loft_mesh(
            (
                (-0.90, 0.00, 0.00),
                (-0.86, 0.26, 0.03),
                (-0.30, 0.29, 0.04),
                (0.10, 0.29, 0.04),
                (0.16, 0.24, 0.02),
                (0.18, 0.00, 0.00),
            ),
            axis=0,
        ),
        physics.TransformRT(
            translation=[TABLE_CENTER[0], TABLE_CENTER[1], patient_axis_z + 0.07]
        ),
    )

    # --- the surgeon ---------------------------------------------------------
    # Body, head and arms are scenery; the receiving hand further down is a real
    # bot with its own controller.
    surgeon_x, surgeon_y = float(SURGEON_STAND[0]), float(SURGEON_STAND[1])

    def place_surgeon(name, mesh, offset=(0.0, 0.0, 0.0)):
        add_static_mesh(
            scene,
            name,
            mesh,
            physics.TransformRT(
                translation=[surgeon_x + offset[0], surgeon_y + offset[1], offset[2]]
            ),
        )

    # Gown and torso in one loft: hem flared at the bottom, narrowing at the
    # waist, widest across the shoulders. Wide in X and shallow in Y because the
    # surgeon stands facing the table (+Y), so their shoulders span X.
    place_surgeon(
        "surgeon_gown",
        loft_mesh(
            (
                (0.70, 0.00, 0.00),
                (0.72, 0.25, 0.17),   # hem
                (0.95, 0.21, 0.15),
                (1.10, 0.19, 0.14),   # waist
                (1.24, 0.23, 0.15),
                (1.34, 0.25, 0.14),   # shoulders
                (1.40, 0.16, 0.11),
                (1.43, 0.06, 0.06),   # neck
                (1.46, 0.05, 0.05),
                (1.49, 0.00, 0.00),   # closed off inside the head
            )
        ),
    )
    place_surgeon(
        "surgeon_head",
        loft_mesh(
            (
                (1.44, 0.00, 0.00),
                (1.48, 0.07, 0.08),
                (1.55, 0.09, 0.10),
                (1.63, 0.08, 0.09),
                (1.68, 0.00, 0.00),
            )
        ),
        (0.0, 0.02, 0.0),
    )
    # Legs under the gown hem, so the figure stands rather than floats.
    for side, sign in (("left", 1.0), ("right", -1.0)):
        mesh, transform = limb_between(
            [surgeon_x + sign * 0.09, surgeon_y, 0.78],
            [surgeon_x + sign * 0.10, surgeon_y, 0.02],
            0.075,
            0.055,
        )
        add_static_mesh(scene, f"surgeon_leg_{side}", mesh, transform)

    # The arm the receiving hand hangs off, in two tapered segments so it has an
    # elbow. The hand's own mount link is at its +X end, so the arm arrives from
    # +X. Kept slim, because the robot works in the space just past the wrist.
    surgeon_elbow = np.array([-0.02, -0.52, 1.12])
    for name, (a, b, r0, r1) in {
        "surgeon_upper_arm": (SURGEON_SHOULDER, surgeon_elbow, 0.058, 0.048),
        "surgeon_forearm": (
            surgeon_elbow,
            SURGEON_HAND_POSITION + np.array([0.02, 0.0, 0.0]),
            0.048,
            0.040,
        ),
    }.items():
        mesh, transform = limb_between(a, b, r0, r1)
        add_static_mesh(scene, name, mesh, transform)

    # The other arm, reaching over the patient, so the surgeon is working rather
    # than standing to attention. It stays on the +X side, well clear of the
    # robot's approach.
    other_shoulder = np.array([surgeon_x + 0.16, surgeon_y - 0.02, 1.32])
    other_elbow = np.array([surgeon_x + 0.24, surgeon_y + 0.14, 1.14])
    other_hand = np.array([surgeon_x + 0.20, surgeon_y + 0.34, 0.96])
    for name, (a, b, r0, r1) in {
        "surgeon_upper_arm_far": (other_shoulder, other_elbow, 0.058, 0.048),
        "surgeon_forearm_far": (other_elbow, other_hand, 0.048, 0.040),
    }.items():
        mesh, transform = limb_between(a, b, r0, r1)
        add_static_mesh(scene, name, mesh, transform)

    # --- the robot's pedestal and the instrument rack -------------------------
    add_static_box(
        scene,
        "robot_pedestal",
        [PEDESTAL_SIZE[0], PEDESTAL_SIZE[1], ROBOT_BASE[2]],
        [ROBOT_BASE[0], ROBOT_BASE[1], ROBOT_BASE[2] / 2.0],
    )

    add_static_box(
        scene,
        "rack_tray",
        RACK_TRAY_SIZE,
        [RACK_CENTER[0], RACK_CENTER[1], RACK_TOP - RACK_TRAY_SIZE[2] / 2.0],
    )
    add_static_box(
        scene,
        "rack_post",
        [0.10, 0.10, RACK_TOP - RACK_TRAY_SIZE[2]],
        [RACK_CENTER[0], RACK_CENTER[1], (RACK_TOP - RACK_TRAY_SIZE[2]) / 2.0],
    )
    add_static_box(
        scene, "rack_foot", [0.40, 0.34, 0.03], [RACK_CENTER[0], RACK_CENTER[1], 0.015]
    )

    # --- the instruments -----------------------------------------------------
    contact = physics.ContactParams()
    contact.coulomb_friction_coefficient = INSTRUMENT_FRICTION

    instruments = []
    for (name, profile, flatten_span, flatten_scale, mass), base in zip(
        INSTRUMENTS, instrument_slots()
    ):
        coordinates, connectivity = revolve_mesh(
            profile, flatten_span=flatten_span, flatten_scale=flatten_scale
        )
        actor = scene.create_rigid_actor(
            name=name,
            shape=physics.create_tri_mesh_shape(
                coordinates.tolist(), connectivity.tolist()
            ),
            world_from_local=physics.TransformRT(translation=base.tolist()),
            contact=contact,
            mass=mass,
            has_gravity=True,
        )
        # Where the centre of mass sits above the base. The handover aims the
        # instrument by its base, but "has the surgeon got it?" can only be asked
        # of the centre of mass, so the two have to be related by a measured
        # number rather than by eye.
        com_local = (
            np.asarray(actor.get_center_of_mass_transform().translation, dtype=float)
            - base
        )
        instruments.append(
            {"name": name, "actor": actor, "home": base, "com_local": com_local}
        )

    # --- the robot -----------------------------------------------------------
    # Turned to face the table (+Y), which puts the rack and the surgeon's hand
    # roughly 45 deg either side of straight ahead.
    robot_prefab = load_bot_prefab(
        ROBOT_ASSET,
        "scrub_robot",
        physics.TransformRT(
            rotation=physics.Quaternion.rotation_z(np.pi / 2.0),
            translation=ROBOT_BASE.tolist(),
        ),
        weld_root=False,  # the FR3 prefab already welds its base to the world
    )
    robot_dofs = dof_map(robot_prefab)
    robot_link_names = [
        robot_prefab.links[i].name for i in range(len(robot_prefab.links))
    ]

    context = robotics.create_context()
    robot = robotics.create_bot(scene, robot_prefab, context)
    robot_actor = robot.get_articulated_actor()
    robot_num_dofs = robot_actor.get_num_dofs()

    # Everything from here indexes the actor, where the root's DOFs come first,
    # so shift across that gap. This arm is welded to the world and contributes
    # none, but reading the count off the actor keeps a floating base correct.
    robot_root_dofs = robot_actor.get_articulated_shape_info().dof_info[0].get_size()
    arm_dof_indices = (
        np.array(
            sorted(
                dof
                for name, dof in robot_dofs.items()
                if name.startswith(ARM_JOINT_PREFIX)
            ),
            dtype=np.int32,
        )
        + robot_root_dofs
    )

    osc = robot.create_controller("BASIC_OSC_PD")
    osc.initialize(
        f"{robot.get_name()}/{ARM_BASE_LINK}", f"{robot.get_name()}/{ARM_EE_LINK}"
    )
    osc_params = osc.get_params()
    osc_params.kp_p = 1400.0
    osc_params.kd_p = 90.0
    osc_params.kp_r = 60.0
    osc_params.kd_r = 5.0
    osc_params.max_translation_error = 0.05  # [m]
    osc_params.max_rotation_error = 0.4  # [rad]
    osc_params.b_apply_max_osc_torque_normalization = True
    osc.set_params(osc_params)

    # The gripper runs on the actor's own pose controller, not on a JSC. Gains
    # are per LINK, and only the gripper's links get any, so the arm stays free
    # for OSC to drive by external torque exactly as before.
    tracked_links = [
        i
        for i, name in enumerate(robot_link_names)
        if name.startswith("2f_85") and name.endswith(GRIPPER_TRACKED_TOKENS)
    ]
    pose_params = physics.PoseControllerParams(len(robot_link_names))
    for link_index in tracked_links:
        pose_params.joint_tracking[link_index] = physics.PoseTrackingParams(
            stiffness=GRIP_STIFFNESS, damping=GRIP_DAMPING
        )
    robot_actor.add_articulated_pose_controller(pose_params)

    gripper_dofs = {
        name: robot_dofs[name] + robot_root_dofs for name in GRIPPER_JOINT_SIGNS
    }

    # --- the surgeon's hand --------------------------------------------------
    surgeon_prefab = load_bot_prefab(
        SURGEON_HAND_ASSET,
        "surgeon_hand",
        physics.TransformRT(
            # Palm up, fingers along -X: this turn puts the hand's local +X (its
            # palm normal) onto world +Z and its local +Z (along the fingers)
            # onto world -X, so closing curls the tips up into a cup.
            rotation=physics.Quaternion.rotation_y(-np.pi / 2.0),
            translation=SURGEON_HAND_POSITION.tolist(),
        ),
        weld_root=True,
    )
    surgeon_dofs = dof_map(surgeon_prefab)
    surgeon = robotics.create_bot(scene, surgeon_prefab, context)
    surgeon_actor = surgeon.get_articulated_actor()
    surgeon_num_dofs = surgeon_actor.get_num_dofs()
    surgeon_root_dofs = (
        surgeon_actor.get_articulated_shape_info().dof_info[0].get_size()
    )
    surgeon_jsc, _ = make_jsc(surgeon, surgeon_num_dofs, SURGEON_KP, SURGEON_SAT)

    return {
        "scene": scene,
        "context": context,
        "robot": robot,
        "robot_actor": robot_actor,
        "robot_link_names": robot_link_names,
        "robot_num_dofs": robot_num_dofs,
        "arm_dof_indices": arm_dof_indices,
        "osc": osc,
        "gripper_dofs": gripper_dofs,
        "robot_base_pose": current_pose(robot_actor, robot_num_dofs),
        "surgeon": surgeon,
        "surgeon_actor": surgeon_actor,
        "surgeon_num_dofs": surgeon_num_dofs,
        "surgeon_jsc": surgeon_jsc,
        "surgeon_open_pose": pose_with(
            surgeon_actor, surgeon_num_dofs, surgeon_dofs, surgeon_root_dofs,
            OPEN_LEFT_DEG,
        ),
        "surgeon_closed": {
            surgeon_dofs[name] + surgeon_root_dofs: np.radians(degrees)
            for name, degrees in CLOSED_LEFT_DEG.items()
        },
        "surgeon_thumb_dofs": frozenset(
            surgeon_dofs[name] + surgeon_root_dofs
            for name in CLOSED_LEFT_DEG
            if name.startswith(THUMB_JOINT_PREFIX)
        ),
        "instruments": instruments,
    }


def blend_pose(open_pose, closed_targets, grip, thumb_dofs=frozenset()):
    """Blend an open hand pose toward its closed one. ``grip`` runs 0 -> 1.

    The thumb runs ahead of the fingers on its own clock, finishing by
    ``THUMB_LEAD``; see the note there for why.
    """
    thumb_progress = min(1.0, grip / THUMB_LEAD)
    pose = np.array(open_pose, dtype=np_real)
    for dof, closed in closed_targets.items():
        progress = thumb_progress if dof in thumb_dofs else grip
        pose[dof] = open_pose[dof] + progress * (closed - open_pose[dof])
    return pose


def slew(current, goal, rate, dt):
    """Step ``current`` toward ``goal`` at no more than ``rate`` [m/s]."""
    delta = np.asarray(goal, dtype=float) - np.asarray(current, dtype=float)
    distance = float(np.linalg.norm(delta))
    limit = rate * dt
    if distance <= limit or distance == 0.0:
        return np.asarray(goal, dtype=float).copy()
    return np.asarray(current, dtype=float) + delta * (limit / distance)


def link_positions(actor, link_names):
    """World translations of every link, keyed by name."""
    transforms = physics.DynamicArrayTransformRT(len(link_names))
    actor.get_articulated_link_transforms(transforms)
    return {
        name: np.asarray(transforms[i].translation, dtype=float)
        for i, name in enumerate(link_names)
    }


# --------------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------------- #


class KeyboardTeleop:
    """Non-blocking key polling. Windows uses msvcrt, POSIX raw-mode stdin."""

    def __init__(self):
        self._posix = False
        if sys.platform != "win32":
            import termios
            import tty

            self._termios = termios
            self._fd = sys.stdin.fileno()
            self._saved = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
            self._posix = True

    def poll(self):
        """Return every key pressed since the last call, lowercased."""
        keys = []
        if sys.platform == "win32":
            import msvcrt

            while msvcrt.kbhit():
                char = msvcrt.getch()
                # Arrow and function keys arrive as a two-byte prefixed sequence.
                if char in (b"\x00", b"\xe0"):
                    msvcrt.getch()
                    continue
                keys.append(char.decode("latin-1").lower())
        else:
            import select

            while select.select([sys.stdin], [], [], 0)[0]:
                keys.append(sys.stdin.read(1).lower())
        return keys

    def close(self):
        if self._posix:
            self._termios.tcsetattr(self._fd, self._termios.TCSADRAIN, self._saved)


class MouseTeleop:
    """Absolute screen-to-workspace mouse steering (Windows only).

    Reads the cursor and the buttons through the Win32 API rather than through a
    window, so it keeps working while the debugger has focus -- the same reason
    the keyboard path polls the console directly.

    The mapping is absolute: the screen rectangle IS the workspace rectangle. A
    relative mapping would need the cursor recentred every tick to stop it
    running off the edge of the control range, and that fights the user for
    their pointer for as long as the example is open.
    """

    LEFT_BUTTON = 0x01
    RIGHT_BUTTON = 0x02

    def __init__(self):
        if sys.platform != "win32":
            raise RuntimeError("--mouse is only implemented on Windows")
        import ctypes.wintypes

        self._user32 = ctypes.windll.user32
        self._point_type = ctypes.wintypes.POINT
        self._width = float(max(1, self._user32.GetSystemMetrics(0)))
        self._height = float(max(1, self._user32.GetSystemMetrics(1)))
        self._left_was_down = False

    def _button(self, code):
        # The high bit of GetAsyncKeyState is "currently down".
        return bool(self._user32.GetAsyncKeyState(code) & 0x8000)

    def poll(self):
        """Return (fraction_x, fraction_y, right_held, left_clicked)."""
        point = self._point_type()
        self._user32.GetCursorPos(ctypes.byref(point))
        fraction_x = min(max(point.x / self._width, 0.0), 1.0)
        fraction_y = min(max(point.y / self._height, 0.0), 1.0)

        left_down = self._button(self.LEFT_BUTTON)
        left_clicked = left_down and not self._left_was_down
        self._left_was_down = left_down

        return fraction_x, fraction_y, self._button(self.RIGHT_BUTTON), left_clicked


def lerp(span, fraction):
    return span[0] + (span[1] - span[0]) * fraction


# --------------------------------------------------------------------------- #
# The scripted handover
# --------------------------------------------------------------------------- #

# (label, seconds, waypoint, grip goal). A waypoint of None means "hold the
# previous one", so that stage only changes the grip. The waypoints themselves
# are resolved at runtime against the chosen instrument and the surgeon's palm.
HANDOVER_SCRIPT = (
    ("cross to the rack", 4.0, "rack_transit", 0.0),
    ("descend around handle", 3.5, "rack_grasp", 0.0),
    ("close on the handle", 2.5, None, 1.0),
    ("lift clear of the rack", 3.5, "rack_transit", 1.0),
    ("carry to the surgeon", 6.0, "handover_transit", 1.0),
    ("present to the surgeon", 4.0, "handover", 1.0),
    ("wait for the surgeon to take it", 2.5, None, 1.0),
    # Open IN PLACE before moving. Rising while the jaws are still parting drags
    # the instrument back out of the surgeon's hand -- and unlike the five-finger
    # hand, nothing on this gripper reaches below the jaws, so there is no reason
    # to retreat and open at the same time.
    ("open the jaws", 3.0, None, 0.0),
    ("lift away", 2.5, "handover_clear", 0.0),
    ("withdraw", 4.0, "retreat", 0.0),
)


def script_waypoints(instrument_home, transit_z):
    """Fist positions the script aims at, for the chosen instrument."""
    grasp = instrument_home + np.array([0.0, 0.0, INSTRUMENT_GRASP_HEIGHT])
    handover = SURGEON_GRIP_POINT + np.array(
        [0.0, 0.0, INSTRUMENT_GRASP_HEIGHT - SURGEON_GRIP_HEIGHT]
    )
    return {
        "rack_transit": np.array([grasp[0], grasp[1], transit_z]),
        "rack_grasp": grasp,
        "handover_transit": np.array([handover[0], handover[1], transit_z]),
        "handover": handover,
        # Withdraw SIDEWAYS, not upward. The instrument's top sits inside the
        # gripper's throat, where the inner knuckles pass 0.012 m from the tool
        # axis, so lifting straight up hooks them under the blade and pulls it
        # back out of the surgeon's hand. The jaws separate along world X, which
        # leaves their open sides facing +-Y, and the robot is on the -Y side --
        # so backing out that way slides the instrument straight out of the gap.
        "handover_clear": handover + np.array([0.0, -0.22, 0.04]),
        "retreat": handover + np.array([0.10, -0.30, 0.26]),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Teleoperate a scrub-nurse robot handing instruments to a surgeon."
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="run the scripted handover instead of live teleoperation",
    )
    parser.add_argument(
        "--view",
        action="store_true",
        help="attach the debugger in --auto mode (live mode always attaches)",
    )
    parser.add_argument(
        "--mouse",
        action="store_true",
        help="steer with the mouse as well as the keyboard (Windows only)",
    )
    parser.add_argument(
        "--instrument",
        default=INSTRUMENTS[0][0],
        choices=[name for name, *_ in INSTRUMENTS],
        help="which instrument the surgeon asks for",
    )
    args = parser.parse_args()

    sim = build_scene()
    scene = sim["scene"]
    robot_actor, surgeon_actor = sim["robot_actor"], sim["surgeon_actor"]
    osc, surgeon_jsc = sim["osc"], sim["surgeon_jsc"]
    robot_num_dofs, surgeon_num_dofs = sim["robot_num_dofs"], sim["surgeon_num_dofs"]
    arm_dof_indices = sim["arm_dof_indices"]
    link_names = sim["robot_link_names"]
    instruments = sim["instruments"]

    requested = next(item for item in instruments if item["name"] == args.instrument)

    robot_dof_indices = np.arange(robot_num_dofs, dtype=np.int32)
    surgeon_dof_indices = np.arange(surgeon_num_dofs, dtype=np.int32)

    # The FR3 base is a fixed weld, so world_from_root is constant and can be
    # captured once to convert world-frame targets into the root frame OSC wants.
    world_from_root = osc.get_current_observations_from_mochi().world_from_root

    # Palm down: a half turn about world X flips the hand's local +Z to world -Z,
    # so the fingers hang over whatever the target is above.
    ee_down = physics.Quaternion.rotation_x(np.pi)

    # Mutated by the loop and read by drive(), which runs both hands per tick.
    state = {"surgeon_grip": 0.0}

    def instrument_centre(item):
        return np.asarray(
            item["actor"].get_center_of_mass_transform().translation, dtype=float
        )

    def drive(flange_target, grip):
        """One control tick: OSC on the arm, JSC on both hands, then step."""
        world_from_target_ee = physics.TransformRT()
        world_from_target_ee.translation = np.asarray(
            flange_target, dtype=float
        ).tolist()
        world_from_target_ee.rotation = ee_down
        # OSC targets are expressed in the actor root frame.
        target_root_from_ee = world_from_root.inverse() * world_from_target_ee

        # np.array (not np.asarray) because the spans the controllers return are
        # read-only views onto their internal buffers.
        arm_tau = np.array(
            osc.compute_output(
                osc.get_current_observations_from_mochi(),
                robotics.ControllerBasicOscPdTarget(
                    root_from_target_ee=target_root_from_ee
                ),
            ),
            dtype=np.float32,
        )
        # The arm is the only thing driven by torque now. The gripper's own
        # pose controller holds the jaws, so there is no second torque vector to
        # add in and nothing to zero out.
        robot_actor.set_external_forces_on_dofs(
            dof_indices=robot_dof_indices, force_values=arm_tau
        )
        set_jaws(grip)

        surgeon_obsv = surgeon_jsc.get_current_observations_from_mochi()
        surgeon_obsv.dt = TIME_STEP
        surgeon_tau = np.array(
            surgeon_jsc.compute_output(
                surgeon_obsv,
                robotics.ControllerBasicJscPdTarget(
                    target_pose=blend_pose(
                        sim["surgeon_open_pose"],
                        sim["surgeon_closed"],
                        state["surgeon_grip"],
                        sim["surgeon_thumb_dofs"],
                    )
                ),
            ),
            dtype=np.float32,
        )
        surgeon_actor.set_external_forces_on_dofs(
            dof_indices=surgeon_dof_indices, force_values=surgeon_tau
        )

        scene.step(TIME_STEP)

    def set_jaws(grip):
        """Command the jaw opening. ``grip`` runs 0 (open) -> 1 (closed)."""
        angle = GRIP_OPEN_DEG + grip * (GRIP_CLOSED_DEG - GRIP_OPEN_DEG)
        pose = np.array(sim["robot_base_pose"], dtype=np_real)
        for name, sign in GRIPPER_JOINT_SIGNS.items():
            pose[sim["gripper_dofs"][name]] = np.radians(angle * sign)
        target = physics.DynamicArrayReal(robot_num_dofs)
        for i in range(robot_num_dofs):
            target[i] = float(pose[i])
        robot_actor.set_articulated_target_pose(target)

    def jaw_centre():
        """Midpoint of the two jaw tips: where a gripped object sits."""
        positions = link_positions(robot_actor, link_names)
        return np.mean([positions[name] for name in JAW_TIP_LINKS], axis=0)

    def calibrate_grasp_frame():
        """Measure fingertip centroid -> flange with the hand held in a fist.

        OSC aims the flange, but the fingers are what has to end up around the
        barrel, and the hand hangs roughly a fifth of a metre below the flange.
        Measuring it here means the example never hard-codes it, and a different
        hand still gets aimed correctly.

        The measurement is taken CLOSED, not open, and that is the whole trick:
        these fingers curl toward the palm, so the fist forms several centimetres
        behind where the open fingers hang. Aim with the open-hand offset and the
        barrel ends up outside the closing fingers, which sweep it off its holder
        instead of gripping it.

        The open hand's reach below the flange is measured in the same pre-roll,
        because it is the other half of the same problem: the command frame is a
        closed fist and the open hand is nowhere near it. transit_height() needs
        it to know how low the fingers dangle while crossing the rack.
        """
        park = ROBOT_BASE + CALIBRATION_OFFSET
        for _ in range(int(CALIBRATION_SECONDS / TIME_STEP)):
            drive(park, 0.0)

        positions = link_positions(robot_actor, link_names)
        open_lowest = min(
            value[2] for name, value in positions.items() if name.startswith("2f_85")
        )
        open_below_flange = positions[ARM_EE_LINK][2] - open_lowest

        grip = 0.0
        while grip < 1.0:
            grip = min(1.0, grip + GRIP_RATE * TIME_STEP)
            drive(park, grip)
        for _ in range(int(CALIBRATION_SECONDS / TIME_STEP)):
            drive(park, 1.0)

        offset = (
            link_positions(robot_actor, link_names)[ARM_EE_LINK] - jaw_centre()
        )

        # Reopen before returning, so the caller starts from a known open hand.
        while grip > 0.0:
            grip = max(0.0, grip - GRIP_RATE * TIME_STEP)
            drive(park, grip)
        for _ in range(int(CALIBRATION_SECONDS / TIME_STEP)):
            drive(park, 0.0)

        print(f"  calibrated fist->flange offset: {np.round(offset, 4)} m")
        return offset, open_below_flange - float(offset[2])

    def calibrate_above(slot_xy, height, rough_offset):
        """Re-measure the grasp frame directly over the slot it will pick from.

        The pre-roll measures the hand in free air at a park pose chosen for
        being clear of everything, which is nowhere near the rack. That turns
        out to matter: the arm reaches the rack in a different configuration,
        and OSC carries a different residual orientation error there, so the
        hand is not quite in the same attitude. The offset is a WORLD-frame
        vector, so a couple of degrees of attitude difference moves it by
        centimetres -- and the whole grasp margin is about one centimetre.

        Measuring again here, hovering over the slot at transit height with the
        same XY and orientation the grasp will use, removes that transfer error:
        the only difference from the grasp pose is height. The first offset is
        still needed to get here at all, hence rough_offset.
        """
        park = np.array([slot_xy[0], slot_xy[1], height]) + rough_offset
        for _ in range(int(CALIBRATION_SECONDS / TIME_STEP)):
            drive(park, 0.0)

        grip = 0.0
        while grip < 1.0:
            grip = min(1.0, grip + GRIP_RATE * TIME_STEP)
            drive(park, grip)
        for _ in range(int(CALIBRATION_SECONDS / TIME_STEP)):
            drive(park, 1.0)

        offset = (
            link_positions(robot_actor, link_names)[ARM_EE_LINK] - jaw_centre()
        )

        while grip > 0.0:
            grip = max(0.0, grip - GRIP_RATE * TIME_STEP)
            drive(park, grip)
        for _ in range(int(CALIBRATION_SECONDS / TIME_STEP)):
            drive(park, 0.0)
        return offset

    def reset_instruments():
        for item in instruments:
            item["actor"].set_root_transform(
                physics.TransformRT(translation=item["home"].tolist())
            )
            item["actor"].set_velocity(
                linear_vel=[0.0, 0.0, 0.0], angular_vel=[0.0, 0.0, 0.0]
            )

    interactive = not args.auto
    keyboard = None
    mouse = None
    if interactive:
        keyboard = KeyboardTeleop()
        if args.mouse:
            mouse = MouseTeleop()
        print(f"\nThe surgeon asks for the {requested['name']}.\n")
        print("  W/S forward-back   A/D left-right   Q/E up-down")
        print("  SPACE toggle grip  1/2/3 fly to a rack slot   H fly to handover")
        print("  R reset the rack   ESC quit")
        if mouse is not None:
            print("\n  Mouse: move to steer, hold RIGHT for height, LEFT click to grip.")
        print("\nKeep THIS terminal focused while steering.")

    # Declare the coordinate convention before attach(), which starts the server,
    # so the debugger renders the scene the right way up: SuperDex is X-forward,
    # Y-left, Z-up (FLU).
    physics.get_debug_server().set_coordinate_space(
        physics.CoordinateSpace(axes=physics.CoordinateSpaceAxes.FLU)
    )

    attached = False
    if interactive or args.view:
        print("Waiting for the debugger to connect...")
        attached = physics.debugger.attach()
        if not attached:
            print("Could not attach the debugger; is another instance already running?")
            if keyboard is not None:
                keyboard.close()
            physics.shutdown()
            return

    fist_to_flange, open_hand_drop = calibrate_grasp_frame()
    transit_z = transit_height(open_hand_drop)
    print(f"  open hand hangs {open_hand_drop:.3f} m below the fist")
    print(f"  transit height:  {transit_z:.3f} m")

    def receive_point(item):
        """Where an instrument's centre of mass sits once it is in the surgeon's hand."""
        return SURGEON_GRIP_POINT + np.array(
            [0.0, 0.0, float(item["com_local"][2]) - SURGEON_GRIP_HEIGHT]
        )

    # Re-measure over the slot, where the arm is in its grasping configuration.
    refined = calibrate_above(requested["home"][:2], transit_z, fist_to_flange)
    print(f"  re-measured over the slot:      {np.round(refined, 4)} m")
    print(f"  transfer error it corrects:     {np.round(refined - fist_to_flange, 4)} m")
    fist_to_flange = refined

    waypoints = script_waypoints(requested["home"], transit_z)
    # Everything below steers the FIST, and the flange target is derived from it
    # at the end of the tick. That is what makes the mouse mapping and the fly-to
    # keys mean what they say: the cursor points at where the fingers go, not at
    # where the wrist goes.
    fist_target = waypoints["rack_transit"].copy()
    fist_goal = fist_target.copy()

    grip = 0.0
    grip_goal = 0.0

    slot_keys = {str(index + 1): item for index, item in enumerate(instruments)}

    stage = 0
    stage_started = scene.get_total_simulation_time()
    surgeon_grip_goal = 0.0
    surgeon_dwell = 0.0
    surgeon_holds = False
    handover_time = None
    next_report = 0.0
    running = True

    try:
        while running:
            # Live mode and --view run until the debugger goes away; headless
            # --auto runs until the script finishes.
            if attached and not physics.debugger.is_attached():
                break

            now = scene.get_total_simulation_time()

            if args.auto:
                label, duration, waypoint, grip_goal = HANDOVER_SCRIPT[stage]
                if waypoint is not None:
                    fist_goal = waypoints[waypoint]
                if now - stage_started > duration:
                    print(
                        f"  [{now:5.1f}s] {label:24s} "
                        f"instrument_z={instrument_centre(requested)[2]:.3f} m  "
                        f"robot={grip:.2f}  surgeon={state['surgeon_grip']:.2f}"
                    )
                    stage += 1
                    stage_started = now
                    if stage >= len(HANDOVER_SCRIPT):
                        running = False
                        continue
            else:
                for key in keyboard.poll():
                    if key == "\x1b":  # ESC
                        running = False
                    elif key == " ":
                        grip_goal = 0.0 if grip_goal > 0.5 else 1.0
                    elif key == "r":
                        reset_instruments()
                    elif key == "h":
                        fist_goal = np.array(
                            [
                                waypoints["handover"][0],
                                waypoints["handover"][1],
                                transit_z,
                            ]
                        )
                    elif key in slot_keys:
                        # To the transit height above the slot, not down onto it:
                        # the descent is the part you want to be driving yourself.
                        home = slot_keys[key]["home"]
                        fist_goal = np.array([home[0], home[1], transit_z])
                    elif key in KEY_TO_AXIS:
                        axis, sign = KEY_TO_AXIS[key]
                        fist_goal = fist_goal.copy()
                        fist_goal[axis] += sign * MOVE_STEP

                lead = fist_goal - fist_target
                distance = float(np.linalg.norm(lead))
                if distance > TARGET_LEASH:
                    fist_goal = fist_target + lead * (TARGET_LEASH / distance)

                if mouse is not None:
                    fraction_x, fraction_y, right_held, left_clicked = mouse.poll()
                    fist_goal = fist_goal.copy()
                    fist_goal[1] = lerp(MOUSE_Y_RANGE, 1.0 - fraction_x)
                    if right_held:
                        fist_goal[2] = lerp(MOUSE_Z_RANGE, 1.0 - fraction_y)
                    else:
                        fist_goal[0] = lerp(MOUSE_X_RANGE, 1.0 - fraction_y)
                    if left_clicked:
                        grip_goal = 0.0 if grip_goal > 0.5 else 1.0

            # Slewing the target rather than jumping to it keeps OSC's
            # acceleration low enough that friction holds the instrument. Without
            # it, a fly-to key or a script waypoint peels it out of the fingers.
            fist_target = slew(
                fist_target,
                fist_goal,
                TARGET_SLEW_RATE if args.auto else TELEOP_SLEW_RATE,
                TIME_STEP,
            )

            # Ramp the grip toward its goal so the fingers close smoothly.
            grip += float(
                np.clip(grip_goal - grip, -GRIP_RATE * TIME_STEP, GRIP_RATE * TIME_STEP)
            )

            # --- the surgeon decides for themselves ---------------------------
            # Whichever instrument is nearest the palm, so the surgeon takes what
            # is actually offered rather than only the one they asked for.
            nearest = min(
                instruments,
                key=lambda item: float(
                    np.linalg.norm(instrument_centre(item) - receive_point(item))
                ),
            )
            reach = float(
                np.linalg.norm(instrument_centre(nearest) - receive_point(nearest))
            )
            if surgeon_holds:
                # Keep hold unless the instrument has clearly gone.
                if reach > SURGEON_RELEASE_RADIUS:
                    surgeon_holds = False
                    surgeon_grip_goal = 0.0
                    handover_time = None
            elif reach < SURGEON_RECEIVE_RADIUS:
                # No longer conditional on the robot having released: the point
                # of the handover is that both are on the instrument at once,
                # briefly, so it is never unsupported.
                surgeon_dwell += TIME_STEP
                if surgeon_dwell >= SURGEON_RECEIVE_DWELL:
                    surgeon_holds = True
                    surgeon_grip_goal = 1.0
                    handover_time = now
                    if not args.auto:
                        print(f"\n  the surgeon takes the {nearest['name']}.")
            else:
                surgeon_dwell = 0.0

            state["surgeon_grip"] += float(
                np.clip(
                    surgeon_grip_goal - state["surgeon_grip"],
                    -GRIP_RATE * TIME_STEP,
                    GRIP_RATE * TIME_STEP,
                )
            )

            drive(fist_target + fist_to_flange, grip)

            if not args.auto and now >= next_report:
                next_report = now + 0.25
                sys.stdout.write(
                    f"\r  fist {np.round(fist_target, 2)}  grip {grip:4.2f}  "
                    f"surgeon has it: {'yes' if surgeon_holds else 'no '}   "
                )
                sys.stdout.flush()
    finally:
        if keyboard is not None:
            keyboard.close()

    if args.auto:
        centre = instrument_centre(requested)
        offset = float(np.linalg.norm(centre - receive_point(requested)))
        print(f"\nInstrument           : {requested['name']}")
        print(f"Final position       : {np.round(centre, 3)} m")
        print(f"Distance from palm   : {offset:.3f} m")
        print(f"Surgeon's grip       : {state['surgeon_grip']:.2f}")
        if handover_time is not None:
            print(f"Handed over at       : {handover_time:.1f} s")
        # The robot has withdrawn by now, so if the surgeon is not holding the
        # instrument then it has fallen -- to the floor, or back onto the rack.
        if not surgeon_holds:
            print("HANDOVER FAILED: the surgeon is not holding the instrument.")
        elif offset > SURGEON_RECEIVE_RADIUS * 1.6:
            print("HANDOVER FAILED: the instrument slipped out of the surgeon's hand.")
        else:
            print("HANDOVER OK: the surgeon is holding the instrument.")

    robotics.destroy_bot(scene, sim["surgeon"])
    robotics.destroy_bot(scene, sim["robot"])
    physics.shutdown()
    print("Simulation complete.")


if __name__ == "__main__":
    main()
