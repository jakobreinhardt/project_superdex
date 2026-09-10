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

"""Example: Steering a hand onto a cup and grasping it

An FR3 v2 arm with a DG5F five-finger hand picks a paper cup up off the ground.
The arm is driven by an operational-space PD controller (OSC), so what you steer
is the hand pose in world space rather than joint angles; the fingers are driven
by a joint-space PD controller (JSC) that interpolates between an open and a
closed pose.

The grasp is a wrap, not a pinch: the hand closes on the cup's walls and holds it
by friction. That is far more forgiving of a few millimetres of positioning error
than a fingertip pinch, and it is why the object is a 93 mm cup rather than a
small block -- these fingers curl toward the palm, so gripping something short
and flat would mean reaching under it, and the ground is in the way.

OSC aims the arm's tool flange (fr3_link8), but the fingers are what has to end
up around the cup, and the hand hangs roughly 0.2 m below the flange. Rather than
hard-code that, a calibration pre-roll measures it: the hand is closed into a
fist in free space and the fingertip centroid is compared with the flange. It is
measured CLOSED on purpose -- the fist forms about 9 cm behind where the open
fingers hang, so aiming with the open-hand offset leaves the cup outside the
closing fingers and they sweep it away. Swap in a different hand or object and
the example still aims correctly.

Two modes:

  --auto    Run a fixed pick-and-place script (approach, descend, close, lift,
            carry, release) and report the cup height, so the grasp can be
            checked without a human at the keyboard. Headless unless --view.

  (default) Live keyboard teleoperation. The hand target follows your keys while
            the simulation runs, which is as close to steering the robot as this
            repo gets today: the debugger is a viewer and has no drag
            manipulator, and SuperDex Teleop (VR) is not released yet.

Keys (live mode; keep this terminal focused, not the debugger window):
     W / S    move hand forward / back   (+X / -X)
     A / D    move hand left / right     (+Y / -Y)
     Q / E    raise / lower hand         (+Z / -Z)
     SPACE    toggle grip (close / open)
     R        put the cup back
     ESC      quit

Each press moves the hand MOVE_STEP; hold a key to glide. A readout of the hand
target, grip and cup height prints while you steer, so you can tell a key that
did nothing from a key that never arrived.

Usage:
    cd <path_to_superdex_robotics>
    python3 examples/control/example_teleop_grasp.py
    python3 examples/control/example_teleop_grasp.py --auto
    python3 examples/control/example_teleop_grasp.py --auto --view
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import superdex.physics as physics
import superdex.robotics as robotics
from superdex.physics.paths import resolve_asset

# The build's `real` type, so a pose handed to a controller Target copies straight
# in rather than being converted element by element.
np_real = np.float64 if physics.uses_double_precision() else np.float32

BOT_ASSET = "bots/arm_hand_combos/fr3_dg5f_short/right/fr3_dg5f_short_right.superdex_bot"
OBJECT_ASSET = "prefabs/paper_cups/collision/paper_cup.mochi.h5"

# The OSC chain spans the joints between these two links: the arm's mount and its
# tool flange, where the hand is attached. Link actors are "<bot_name>/<link_name>".
ARM_BASE_LINK = "fr3_link0"
ARM_EE_LINK = "fr3_link8"

# The arm's joints all share this prefix, which is how arm DOFs (owned by OSC)
# are told apart from hand DOFs (owned by JSC).
ARM_JOINT_PREFIX = "fr3_joint"

# DG5F finger 1 is the thumb; 2..5 are index through little.
FINGERS = (2, 3, 4, 5)

# The links whose centroid defines where the finger cage actually is.
FINGERTIP_LINKS = tuple(f"dg5f_link_{finger}_tip" for finger in FINGERS)

# Closed-pose angle per joint [deg]. The grip signal blends the default (open)
# pose toward these, so only the joints named here move.
#
# These are not uniform because the DG5F's joints are not: the ranges differ per
# finger, and two of them are signed the other way. Check the prefab's
# min_limit/max_limit before changing a value -- a target outside a joint's range
# is silently clamped by the limit, and the finger simply never arrives.
#
#   dg5f_joint_1_2  spans -180..0 deg, so the thumb opposes on NEGATIVE angles.
#   dg5f_joint_5_2  is a spread (0..35 deg), not a curl, so the little finger
#                   curls at _3/_4 while fingers 2-4 curl at _2/_3.
CLOSED_ANGLES_DEG = {
    # Thumb: swing it across the palm, then curl, so it opposes the fingers.
    # Without this the hand just rakes the cup along the ground.
    "dg5f_joint_1_1": 45.0,
    "dg5f_joint_1_2": -80.0,
    "dg5f_joint_1_3": 55.0,
    "dg5f_joint_1_4": 45.0,
    # Index, middle, ring: curl at the two knuckles. Well past the cup's wall,
    # so they keep squeezing after contact instead of stopping exactly on it.
    "dg5f_joint_2_2": 85.0,
    "dg5f_joint_2_3": 70.0,
    "dg5f_joint_3_2": 85.0,
    "dg5f_joint_3_3": 70.0,
    "dg5f_joint_4_2": 85.0,
    "dg5f_joint_4_3": 70.0,
    # Little finger: curls one segment further out than the others.
    "dg5f_joint_5_3": 70.0,
    "dg5f_joint_5_4": 60.0,
}

# How fast the grip opens and closes [grip fraction per second]. Slow enough that
# the fingers settle onto the cup rather than knocking it over.
GRIP_RATE = 1.2

# Finger stiffness, soft while closing and firm once the grasp has formed.
#
# One setting cannot do both jobs. Closing at the firm gains makes the fingers
# arrive hard enough to knock the cup over before they ever surround it; holding
# at the soft gains does not generate enough friction, and the cup slides down
# through the fingers over a second or two -- it lifts, then quietly slips out
# during the carry. So the hand closes soft and is tightened afterwards, which is
# also what a real grasp does: approach compliant, then squeeze.
GRIP_KP_SOFT, GRIP_SAT_SOFT = 8.0, 5.0  # [Nm/rad], torque clamp
# Verified by sweep: at 22/14 the cup lifts but slips out during the sideways
# carry; 35/25 holds it all the way through.
GRIP_KP_FIRM, GRIP_SAT_FIRM = 35.0, 25.0
# Firmness ramps in over this long once the fingers are fully closed [s].
FIRM_RAMP_SECONDS = 0.6

# The object to pick up: a 93 mm wide, 113 mm tall paper cup, mesh origin at the
# centre of its base. Its size is what makes this work -- a hand-sized object with
# tall walls can be grasped by closing on its SIDES. The 25 mm Box-and-Blocks
# block cannot: these fingers curl toward the palm, so to hold something that
# short they would have to reach under it, and the ground is in the way.
OBJECT_START = np.array([0.50, 0.0, 0.0])  # [m] base centre, resting on the ground
OBJECT_MASS = 0.03  # [kg], about right for a full-ish paper cup
# The 0.5 default lets the cup slip through the fingers; the grasp is held by
# friction on the walls, so this matters more than usual.
OBJECT_FRICTION = 1.1

# Where the fist should close, as a fraction of the object's height. Above the
# centre of mass, so the cup pendulums back upright instead of tipping out of the
# hand. Verified across 0.55-0.75; below about 0.5 the fingers catch the base and
# tip the cup over.
CAGE_HEIGHT_FRACTION = 0.65

# Flange height used for the calibration pre-roll. Well clear of the ground, so
# the open hand can settle without driving its fingers into the floor -- which
# would bend them back and make the measured offset meaningless.
CALIBRATION_FLANGE_HEIGHT = 0.45  # [m]
CALIBRATION_SECONDS = 2.5

TIME_STEP = 1.0 / 200.0  # [s]

# How far the hand target jumps per key press [m].
#
# This used to be a speed (0.18 m/s) scaled by a bare 0.05, which worked out at
# 9 mm per press. That is invisible at scene scale, and since live mode printed
# nothing, a working teleop was indistinguishable from a dead one -- tapping a
# key really did appear to do nothing at all. A press is a discrete event here
# (msvcrt hands over characters, not key-down/key-up), so a distance per press
# is the honest unit; holding a key lets auto-repeat turn it into a glide of
# roughly 0.9 m/s.
MOVE_STEP = 0.08

# How far ahead of the arm the target is allowed to get [m]. Auto-repeat can
# push the target far faster than OSC can follow it, and the arm then keeps
# chasing long after the key is released, which feels like the controls are
# lagging rather than the target having run off. A short leash keeps the motion
# answering the keys instead of their history.
TARGET_LEASH = 0.20

# How fast the scripted hand target is allowed to travel [m/s]. The stages are
# waypoints, so without this the target teleports between them and OSC yanks the
# arm after it -- which peels the cup out of the fingers mid-lift. Slewing the
# target keeps the acceleration low enough that friction holds.
TARGET_SLEW_RATE = 0.15


def build_scene():
    """Load the bot, add a ground plane and the cup, and wire up both controllers."""
    physics.initialize(num_worker_threads=-1)

    # SuperDex robots use a Z-up convention, so gravity points down -Z.
    scene = physics.create_scene("Teleop Grasp")
    scene.set_gravity([0, 0, -9.81])

    bot_prefab = robotics.load_bot_prefab_from_file(str(resolve_asset(BOT_ASSET)))

    # Cheap "gravity compensation": neither BASIC_OSC_PD nor BASIC_JSC_PD has a
    # gravity term, so without this the arm and fingers sag off their targets.
    # The cup keeps its gravity, or "picking it up" would prove nothing.
    for i in range(len(bot_prefab.links)):
        bot_prefab.links[i].has_gravity = False

    # Split the DOFs into the two disjoint sets the controllers own, numbering
    # them in bot DOF space: prefab joint order, skipping the root joint.
    arm_dofs: list[int] = []
    joint_name_to_dof: dict[str, int] = {}
    dof = 0
    for i in range(len(bot_prefab.joints)):
        joint = bot_prefab.joints[i]
        if joint.type != physics.ArticulatedJointType.REVOLUTE:
            continue
        joint_name_to_dof[joint.name] = dof
        if joint.name.startswith(ARM_JOINT_PREFIX):
            arm_dofs.append(dof)
        dof += 1

    robotics_context = robotics.create_context()
    bot = robotics.create_bot(scene, bot_prefab, robotics_context)
    bot_actor = bot.get_articulated_actor()

    scene.create_rigid_actor(
        name="ground",
        shape=physics.create_plane_shape(normal=[0, 0, 1], distance=0),
        is_static=True,
    )

    # An implicit sphere or plane cannot be a dynamic body -- a dynamic rigid
    # actor needs a surface mesh -- so the object is loaded from a mesh asset.
    object_shape = physics.load_shape_from_file(
        file_path=str(resolve_asset(OBJECT_ASSET))
    )
    # Read the object's extent off its mesh rather than hard-coding it, so the
    # grasp height still lands correctly if you swap the asset.
    aabb = physics.get_shape_aabb(object_shape)
    object_height = float(
        np.asarray(aabb.max, dtype=float)[2] - np.asarray(aabb.min, dtype=float)[2]
    )

    object_contact = physics.ContactParams()
    object_contact.coulomb_friction_coefficient = OBJECT_FRICTION
    grasp_object = scene.create_rigid_actor(
        name="cup",
        shape=object_shape,
        world_from_local=physics.TransformRT(translation=OBJECT_START.tolist()),
        contact=object_contact,
        mass=OBJECT_MASS,
        has_gravity=True,
    )

    num_dofs = bot_actor.get_num_dofs()

    # Everything from here indexes the actor, where the root's DOFs come first,
    # so shift across that gap. This arm is welded to the world and contributes
    # none, but reading the count off the actor keeps a floating base correct.
    num_root_dofs = bot_actor.get_articulated_shape_info().dof_info[0].get_size()
    arm_dof_indices = np.array(arm_dofs, dtype=np.int32) + num_root_dofs

    # Map each closed-pose joint to its actor DOF index and target angle [rad].
    closed_targets = {
        joint_name_to_dof[name] + num_root_dofs: np.radians(degrees)
        for name, degrees in CLOSED_ANGLES_DEG.items()
    }

    # --- OSC on the arm --------------------------------------------------------
    osc = bot.create_controller("BASIC_OSC_PD")
    bot_name = bot.get_name()
    osc.initialize(f"{bot_name}/{ARM_BASE_LINK}", f"{bot_name}/{ARM_EE_LINK}")
    osc_params = osc.get_params()
    osc_params.kp_p = 1400.0
    osc_params.kd_p = 90.0
    osc_params.kp_r = 60.0
    osc_params.kd_r = 5.0
    osc_params.max_translation_error = 0.05  # [m]
    osc_params.max_rotation_error = 0.4  # [rad]
    osc_params.b_apply_max_osc_torque_normalization = True
    osc.set_params(osc_params)

    # --- JSC on the hand -------------------------------------------------------
    # JSC has no notion of a sub-chain: its gains, target pose and output are all
    # sized to the full actor, arm DOFs included.
    jsc = bot.create_controller("BASIC_JSC_PD")
    jsc_params = robotics.ControllerBasicJscPdParams()
    jsc_params.kd = np.full(num_dofs, 0.4, dtype=np.float32)  # [Nms/rad]
    jsc_params.deadband = np.zeros(num_dofs, dtype=np.float32)
    set_grip_firmness(jsc, jsc_params, num_dofs, 0.0)

    default_pose = physics.DynamicArrayReal(num_dofs)
    bot_actor.get_articulated_pose(default_pose)
    open_pose = np.array(default_pose, dtype=np_real)

    link_names = [bot_prefab.links[i].name for i in range(len(bot_prefab.links))]

    return {
        "scene": scene,
        "bot": bot,
        "actor": bot_actor,
        "object": grasp_object,
        "object_height": object_height,
        "osc": osc,
        "jsc": jsc,
        "arm_dof_indices": arm_dof_indices,
        "closed_targets": closed_targets,
        "jsc_params": jsc_params,
        "open_pose": open_pose,
        "link_names": link_names,
    }


def finger_target(open_pose, closed_targets, grip):
    """Blend the open hand pose toward the closed one. ``grip`` runs 0 -> 1."""
    pose = np.array(open_pose, dtype=np_real)
    for dof, closed in closed_targets.items():
        pose[dof] = open_pose[dof] + grip * (closed - open_pose[dof])
    return pose


def set_grip_firmness(jsc, jsc_params, num_dofs, firmness):
    """Blend the finger gains from soft (0) to firm (1) and push them to the JSC."""
    kp = GRIP_KP_SOFT + firmness * (GRIP_KP_FIRM - GRIP_KP_SOFT)
    saturation = GRIP_SAT_SOFT + firmness * (GRIP_SAT_FIRM - GRIP_SAT_SOFT)
    jsc_params.kp = np.full(num_dofs, kp, dtype=np.float32)
    jsc_params.saturation = np.full(num_dofs, saturation, dtype=np.float32)
    jsc.set_params(jsc_params)


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


# Scripted stages: (label, seconds, cage_offset, grip_goal). The offset is where
# the CLOSING FIST should end up relative to the cup's base, not where the
# flange goes -- the flange correction is measured at runtime and added on top.
PICK_SCRIPT = [
    ("settle above cup", 2.0, np.array([0.0, 0.0, 0.30]), 0.0),
    ("descend around cup", 2.5, None, 0.0),
    ("close fingers", 2.5, None, 1.0),
    ("lift", 2.5, np.array([0.0, 0.0, 0.25]), 1.0),
    ("carry sideways", 2.5, np.array([0.0, -0.22, 0.25]), 1.0),
    ("release", 2.0, np.array([0.0, -0.22, 0.12]), 0.0),
]

KEY_TO_AXIS = {
    "w": (0, 1), "s": (0, -1),
    "a": (1, 1), "d": (1, -1),
    "q": (2, 1), "e": (2, -1),
}


def main():
    parser = argparse.ArgumentParser(
        description="Steer a five-finger hand onto a cup and grasp it."
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="run the scripted pick sequence instead of keyboard teleop",
    )
    parser.add_argument(
        "--view",
        action="store_true",
        help="attach the debugger in --auto mode (live mode always attaches)",
    )
    args = parser.parse_args()

    sim = build_scene()
    scene, actor = sim["scene"], sim["actor"]
    grasp_object, object_height = sim["object"], sim["object_height"]
    cage_height = CAGE_HEIGHT_FRACTION * object_height
    osc, jsc = sim["osc"], sim["jsc"]
    open_pose, closed_targets = sim["open_pose"], sim["closed_targets"]
    jsc_params = sim["jsc_params"]
    arm_dof_indices, link_names = sim["arm_dof_indices"], sim["link_names"]

    num_dofs = actor.get_num_dofs()
    all_dof_indices = np.arange(num_dofs, dtype=np.int32)

    # The FR3 base is a fixed weld, so world_from_root is constant and can be
    # captured once to convert world-frame targets into the root frame OSC wants.
    world_from_root = osc.get_current_observations_from_mochi().world_from_root

    # Palm down: a half turn about world X flips the hand's local +Z to world -Z.
    ee_down = physics.Quaternion.rotation_x(np.pi)


    def object_centre():
        """The object's centre of mass, which is what 'did it lift' really means."""
        return np.asarray(
            grasp_object.get_center_of_mass_transform().translation, dtype=float
        )

    def fingertip_centroid():
        positions = link_positions(actor, link_names)
        return np.mean([positions[name] for name in FINGERTIP_LINKS], axis=0)

    def drive(hand_target, grip):
        """One control tick: OSC on the arm, JSC on the fingers, then step."""
        world_from_target_ee = physics.TransformRT()
        world_from_target_ee.translation = np.asarray(hand_target, dtype=float).tolist()
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
        jsc_obsv = jsc.get_current_observations_from_mochi()
        jsc_obsv.dt = TIME_STEP
        hand_tau = np.array(
            jsc.compute_output(
                jsc_obsv,
                robotics.ControllerBasicJscPdTarget(
                    target_pose=finger_target(open_pose, closed_targets, grip)
                ),
            ),
            dtype=np.float32,
        )

        # Both torque vectors span the whole actor. OSC already zeros everything
        # outside its arm chain, but JSC does not, so zero its arm entries here or
        # the two would fight over those DOFs.
        hand_tau[arm_dof_indices] = 0.0
        actor.set_external_forces_on_dofs(
            dof_indices=all_dof_indices, force_values=arm_tau + hand_tau
        )
        scene.step(TIME_STEP)

    def calibrate_cage_offset():
        """Measure fingertip centroid -> flange with the hand held in a fist.

        OSC aims the flange, but the fingers are what has to end up around the
        object, and the hand hangs roughly a fifth of a metre below the flange.
        Measuring that here means the script never hard-codes it, and a different
        hand still gets aimed correctly.

        The measurement is taken CLOSED, not open, and that is the whole trick.
        These fingers curl toward the palm, so the fist forms about 9 cm behind
        where the open fingers hang. Aiming with the open-hand offset puts the
        object outside the closing fingers and they sweep it away instead of
        gripping it; aiming with the closed offset puts it exactly where they
        converge. The hand is reopened afterwards so the approach is unobstructed.

        Done high above the ground so the fingers can move freely -- pressed into
        the floor they would bend back and the measurement would be meaningless.
        """
        park = object_centre() * np.array([1.0, 1.0, 0.0]) + np.array(
            [0.0, 0.0, CALIBRATION_FLANGE_HEIGHT]
        )
        for _ in range(int(CALIBRATION_SECONDS / TIME_STEP)):
            drive(park, 0.0)

        grip = 0.0
        while grip < 1.0:
            grip = min(1.0, grip + GRIP_RATE * TIME_STEP)
            drive(park, grip)
        for _ in range(int(CALIBRATION_SECONDS / TIME_STEP)):
            drive(park, 1.0)

        offset = link_positions(actor, link_names)[ARM_EE_LINK] - fingertip_centroid()

        # Reopen before returning, so the caller starts from a known open hand.
        while grip > 0.0:
            grip = max(0.0, grip - GRIP_RATE * TIME_STEP)
            drive(park, grip)
        for _ in range(int(CALIBRATION_SECONDS / TIME_STEP)):
            drive(park, 0.0)

        print(f"  calibrated fist->flange offset: {np.round(offset, 4)} m")
        return offset

    interactive = not args.auto
    teleop = None
    if interactive:
        teleop = KeyboardTeleop()
        print("W/S forward-back   A/D left-right   Q/E up-down")
        print("SPACE toggle grip  R reset cup      ESC quit")
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
            if teleop is not None:
                teleop.close()
            physics.shutdown()
            return

    # Measure the hand offset before anything aims at the cup.
    cage_to_flange = calibrate_cage_offset()

    # Where the object sits before anyone touches it. The script aims at this
    # fixed point rather than at the live object pose: once grasped it moves with
    # the hand, and chasing it would make the target run away from the arm.
    grasp_anchor = object_centre()
    resting_height = grasp_anchor[2]

    # The cage offsets are measured from the object's BASE, not its centre of
    # mass, so the grasp height reads as a fraction of its height.
    grasp_anchor = grasp_anchor * np.array([1.0, 1.0, 0.0]) + np.array(
        [0.0, 0.0, OBJECT_START[2]]
    )

    # Park the hand above the cup, fist aligned with it.
    hand_target = object_centre() + np.array([0.0, 0.0, 0.18]) + cage_to_flange
    grip = 0.0
    grip_goal = 0.0
    # 0 while closing, ramping to 1 once the fingers are shut. See the comment on
    # GRIP_KP_SOFT: soft to close without knocking the cup over, firm to hold it.
    firmness = 0.0

    stage = 0
    stage_started = scene.get_total_simulation_time()
    next_report = 0.0
    lifted_peak = 0.0
    # Height at the end of the carry, before the deliberate release. Peak height
    # alone would pass a grasp that lifts the cup and then drops it mid-carry.
    height_after_carry = None
    running = True

    try:
        while running:
            # Live mode and --view run until the debugger goes away; headless
            # --auto runs until the script finishes.
            if attached and not physics.debugger.is_attached():
                break

            t = scene.get_total_simulation_time()

            if args.auto:
                label, duration, cage_offset, grip_goal = PICK_SCRIPT[stage]
                # Aim the fist at the cup's resting spot, not at the cup
                # itself: once it is grasped it moves with the hand, and chasing
                # it would make the target run away from the arm.
                # A None offset means "at the grasp height", which depends on the
                # object's measured size rather than a fixed constant.
                if cage_offset is None:
                    cage_offset = np.array([0.0, 0.0, cage_height])
                hand_target = slew(
                    hand_target,
                    grasp_anchor + cage_offset + cage_to_flange,
                    TARGET_SLEW_RATE,
                    TIME_STEP,
                )
                if t - stage_started > duration:
                    print(
                        f"  [{t:5.1f}s] {label:22s} obj_z={object_centre()[2]:.3f} m  "
                        f"grip={grip:.2f}  firm={firmness:.2f}"
                    )
                    if label == "carry sideways":
                        height_after_carry = object_centre()[2]
                    stage += 1
                    stage_started = t
                    if stage >= len(PICK_SCRIPT):
                        running = False
                        continue
                # Only count height from the lift onward, so the cup's resting
                # height and any nudge during the approach do not register.
                if stage >= 3:
                    lifted_peak = max(lifted_peak, object_centre()[2])
            else:
                for key in teleop.poll():
                    if key == "\x1b":  # ESC
                        running = False
                    elif key == " ":
                        grip_goal = 0.0 if grip_goal > 0.5 else 1.0
                    elif key == "r":
                        grasp_object.set_root_transform(
                            physics.TransformRT(translation=OBJECT_START.tolist())
                        )
                    elif key in KEY_TO_AXIS:
                        axis, sign = KEY_TO_AXIS[key]
                        hand_target[axis] += sign * MOVE_STEP

                flange = link_positions(actor, link_names)[ARM_EE_LINK]
                lead = hand_target - flange
                distance = float(np.linalg.norm(lead))
                if distance > TARGET_LEASH:
                    hand_target = flange + lead * (TARGET_LEASH / distance)

            # Ramp the grip toward its goal so the fingers close smoothly.
            grip += float(
                np.clip(grip_goal - grip, -GRIP_RATE * TIME_STEP, GRIP_RATE * TIME_STEP)
            )

            # Tighten only once the fingers are shut, and go slack again as soon
            # as the grip is released, so the next approach is compliant.
            firm_goal = 1.0 if (grip_goal >= 1.0 and grip >= 1.0) else 0.0
            step = TIME_STEP / FIRM_RAMP_SECONDS
            new_firmness = float(np.clip(firm_goal - firmness, -step, step)) + firmness
            if new_firmness != firmness:
                firmness = new_firmness
                set_grip_firmness(jsc, jsc_params, num_dofs, firmness)

            drive(hand_target, grip)

            # A live readout, so a key that did nothing is distinguishable from
            # a key that never arrived. Live mode printed nothing at all before,
            # which is half of why small movements read as broken input.
            if not args.auto and t >= next_report:
                next_report = t + 0.25
                sys.stdout.write(
                    f"\r  hand {np.round(hand_target, 3)}  grip {grip:4.2f}  "
                    f"cup_z {object_centre()[2]:.3f}   "
                )
                sys.stdout.flush()
    finally:
        if teleop is not None:
            teleop.close()

    if args.auto:
        carried = height_after_carry if height_after_carry is not None else 0.0
        print(f"\nPeak height during the lift : {lifted_peak:.3f} m")
        print(f"Height at end of the carry  : {carried:.3f} m")
        print(f"(resting height was {resting_height:.3f} m)")
        # Peak height alone would pass a grasp that lifts the cup and then drops
        # it halfway through the carry, which is exactly what the softer grip did.
        if lifted_peak <= resting_height + 0.05:
            print("GRASP FAILED: the cup never got clear of the ground.")
        elif carried <= resting_height + 0.05:
            print("GRASP FAILED: the cup was lifted but slipped out during the carry.")
        else:
            print("GRASP OK: the cup was lifted, carried, and set back down.")

    robotics.destroy_bot(scene, sim["bot"])
    physics.shutdown()
    print("Simulation complete.")


if __name__ == "__main__":
    main()
