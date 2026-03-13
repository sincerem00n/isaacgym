# Copyright (c) 2025, RAI Hanumanoid Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# IsaacGym (IsaacGymEnvs) implementation of the Hanumanoid A3 locomotion task.
#
# Observations and rewards are ported directly from:
#   hanu_lab/tasks/manager_based/locomotion/velocity/
#      ├── velocity_env_cfg.py          (base env config)
#      └── config/hanu_a3/
#           └── rough_env_cfg.py        (HanuA3RoughEnvCfgV1)
#
# ─────────────────────────────────────────────────────────────────────────────
# OBSERVATION SPACE (99 dims, from HanuA3RoughEnvCfgV1)
# ─────────────────────────────────────────────────────────────────────────────
#   base_ang_vel        3   body-frame angular velocity,   scale = 0.25
#   projected_gravity   3   gravity vector in base frame,  scale = 1.0
#   velocity_commands   3   [vx, vy, yaw_rate],            scale = 1.0
#   joint_pos_rel      30   q - q_default,                 scale = 1.0
#   joint_vel_rel      30   dq,                            scale = 0.05
#   last_action        30   previous action sent to env,   scale = 1.0
#   ─────────────────────────────────────────
#   TOTAL              99
#
# ─────────────────────────────────────────────────────────────────────────────
# REWARD TERMS (from HanuA3RoughEnvCfgV1)
# ─────────────────────────────────────────────────────────────────────────────
#   track_lin_vel_xy_exp    +1.5   exp(-||cmd_xy - vel_xy_yaw_frame||² / 0.25)
#   track_ang_vel_z_exp     +1.0   exp(-(cmd_yaw - vel_yaw_world)²   / 0.25)
#   upright_orientation     +3.0   -||proj_gravity - [0,0,-1]||²   (higher=better)
#   feet_air_time           +0.05  min single-stance time, clipped at 0.18 s
#   feet_air_time_penalty   -0.05  penalise swing foot airtime > 0.25 s
#   feet_slide              -0.2   foot linear velocity while in contact
#   lin_vel_z_l2            -0.2   base z linear velocity²
#   ang_vel_xy_l2           -0.05  base xy angular velocity²
#   dof_torques_l2          -2e-6  joint torques² (hip/knee/ankle only)
#   dof_acc_l2              -1e-7  joint acceleration² (hip_pitch only)
#   action_rate_l2          -0.005 (action - prev_action)²
#   joint_vel_legs          -0.3   hip_yaw joint velocity²
#   joint_vel_neck          -0.5   neck joint velocity²
#   joint_deviation_arms    -0.1   |q_arm - q_arm_default| (shoulder/elbow/wrist)
#   joint_deviation_neck    -0.1   |q_neck - q_neck_default|
#   ankle_dof_pos_limits    -1.0   ankle joints beyond soft limits
#   feet_mirror             -0.02  asymmetry between left/right leg actions
#   undesired_contacts      -0.2   groin / glute body contacts
#   termination_penalty   -200.0   episode terminated (base_link contact)
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *
# from isaacgymenvs.utils.torch_jit_utils import scale, unscale, quat_mul, quat_conjugate, quat_from_angle_axis, \
#     to_torch, get_axis_params, torch_rand_float, tensor_clamp, compute_heading_and_up, compute_rot, normalize_angle

from isaacgymenvs.tasks.base.vec_task import VecTask


# ──────────────────────────────────────────────────────────────────────────────
# Helper: rotate a vector from world frame to base (body) frame
# ──────────────────────────────────────────────────────────────────────────────
def quat_rotate_inverse_batched(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Rotate vector v from world frame to body frame using quaternion q^{-1}.

    Args:
        q: (N, 4) quaternion [x, y, z, w]
        v: (N, 3) vector in world frame

    Returns:
        (N, 3) vector in body frame
    """
    # isaacgym.torch_utils.quat_rotate_inverse already does this
    return quat_rotate_inverse(q, v)


# ──────────────────────────────────────────────────────────────────────────────
# Helper: extract yaw quaternion (zero roll/pitch)
# ──────────────────────────────────────────────────────────────────────────────
def yaw_quat(q: torch.Tensor) -> torch.Tensor:
    """
    Return a quaternion that contains only the yaw component of q.
    q: (N, 4) [x, y, z, w]
    """
    # reconstruct yaw-only rotation
    yaw = torch.atan2(
        2.0 * (q[:, 3] * q[:, 2] + q[:, 0] * q[:, 1]),
        1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2),
    )
    half_yaw = yaw * 0.5
    zeros = torch.zeros_like(half_yaw)
    q_yaw = torch.stack([zeros, zeros, torch.sin(half_yaw), torch.cos(half_yaw)], dim=-1)
    return q_yaw


class Hanu(VecTask):
    """
    IsaacGym VecTask implementation of the Hanumanoid A3 locomotion environment.

    Observations and rewards mirror the Isaac Lab config HanuA3RoughEnvCfgV1
    defined in hanu_lab/tasks/manager_based/locomotion/velocity/config/hanu_a3/rough_env_cfg.py.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # Initialisation
    # ─────────────────────────────────────────────────────────────────────────

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless,
                 virtual_screen_capture, force_render):

        self.cfg = cfg

        # ── randomisation ──────────────────────────────────────────────────
        self.randomize = self.cfg["task"]["randomize"]
        # self.dof_vel_scale = self.cfg["env"]["dofVelocityScale"]
        # self.contact_force_scale = self.cfg["env"]["contactForceScale"]
        # self.power_scale = self.cfg["env"]["powerScale"]
        # self.heading_weight = self.cfg["env"]["headingWeight"]
        # self.up_weight = self.cfg["env"]["upWeight"]
        # self.actions_cost_scale = self.cfg["env"]["actionsCost"]
        # self.energy_cost_scale = self.cfg["env"]["energyCost"]
        # self.joint_at_limit_cost_scale = self.cfg["env"]["jointAtLimitCost"]
        # self.death_cost = self.cfg["env"]["deathCost"]
        # self.termination_height = self.cfg["env"]["terminationHeight"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.plane_static_friction  = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution      = self.cfg["env"]["plane"]["restitution"]
        self.max_episode_length     = self.cfg["env"]["episodeLength"]

        # ── action / DOF dims ─────────────────────────────────────────────
        self.cfg["env"]["numActions"] = 30
        self.num_dof = self.cfg["env"]["numActions"]

        # ── observation dims (HanuA3RoughEnvCfgV1) ────────────────────────
        # base_ang_vel(3) + proj_gravity(3) + commands(3)
        # + joint_pos_rel(30) + joint_vel_rel(30) + last_action(30)  →  99
        self.cfg["env"]["numObservations"]          = 3 + 3 + 3 + self.num_dof * 3   # 99
        self.cfg["env"]["numPrivilegedObservations"] = 0

        # ── observation / action scales (HanuA3RoughEnvCfgV1) ─────────────
        self.obs_scales = {
            "ang_vel":   0.25,   # base_ang_vel.scale
            "dof_pos":   1.0,    # joint_pos.scale
            "dof_vel":   0.05,   # joint_vel.scale
            "commands":  1.0,    # velocity_commands.scale
            "gravity":   1.0,    # projected_gravity.scale
            "actions":   1.0,    # actions.scale
        }
        self.action_scale = 0.25  # actions.joint_pos.scale in HanuA3RoughEnvCfgV1

        # ── reward weights (HanuA3RoughEnvCfgV1) ──────────────────────────
        self.rew_scales = {
            "track_lin_vel_xy":    1.5,     # exp kernel
            "track_ang_vel_z":     1.0,     # exp kernel
            "upright_orientation": 3.0,     # upright_orientation_l2
            "feet_air_time":       0.5,    # feet_air_time_positive_biped
            "feet_air_time_neg":  -0.0,    # feet_air_time_negative_biped (penalty)
            "feet_slide":         -0.2,     # feet_slide
            "lin_vel_z_l2":       -0.2,     # lin_vel_z_l2
            "ang_vel_xy_l2":      -0.05,    # ang_vel_xy_l2
            "dof_torques_l2":     -2.0e-6,  # dof_torques_l2 (hip/knee/ankle)
            "dof_acc_l2":         -1.0e-7,  # dof_acc_l2 (hip_pitch)
            "action_rate_l2":     -0.005,   # action_rate_l2
            "joint_vel_legs":     -0.3,     # joint_vel_l2 (hip_yaw)
            "joint_vel_neck":     -0.5,     # joint_vel_l2 (neck)
            "joint_deviation_arms": -0.1,   # joint_deviation_l1 (shoulder/elbow/wrist)
            "joint_deviation_neck": -0.1,   # joint_deviation_l1 (neck)
            "ankle_dof_pos_limits": -1.0,   # joint_pos_limits (ankle)
            "feet_mirror":        -0.0,    # action_mirror
            "undesired_contacts": -0.2,     # groin / glute contacts
            "termination_penalty": -200.0,  # is_terminated
        }

        # ── reward std for exponential kernels ────────────────────────────
        self.lin_vel_std  = 0.5   # std for track_lin_vel_xy_exp (HanuA3RoughEnvCfgV1)
        self.ang_vel_std  = 0.5   # std for track_ang_vel_z_exp  (HanuA3RoughEnvCfgV1)

        # ── feet air-time thresholds ───────────────────────────────────────
        self.air_time_threshold_pos = 0.18   # feet_air_time.threshold in V1
        self.air_time_threshold_neg = 0.25   # feet_air_time_penalty.threshold in V1

        # ── PD gains per joint class ───────────────────────────────────────
        #   (same values used in HANU_A3_CFG / previous hanu.py)
        self.pd_gains = {
            "hip_yaw":   {"stiffness": 200.0, "damping":  5.0, "effort": 300.0},
            "hip_roll":  {"stiffness": 200.0, "damping":  5.0, "effort": 300.0},
            "hip_pitch": {"stiffness": 250.0, "damping":  5.0, "effort": 300.0},
            "knee":      {"stiffness": 250.0, "damping":  5.0, "effort": 300.0},
            "ankle":     {"stiffness":  20.0, "damping":  2.0, "effort":  20.0},
            "default":   {"stiffness":  40.0, "damping": 10.0, "effort": 300.0},
        }

        # ── call parent __init__ ───────────────────────────────────────────
        super().__init__(cfg, rl_device, sim_device, graphics_device_id,
                         headless, virtual_screen_capture, force_render)

        # ── acquire simulation tensors ────────────────────────────────────
        _actor_root_state    = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state           = self.gym.acquire_dof_state_tensor(self.sim)
        _net_contact_forces  = self.gym.acquire_net_contact_force_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # ── wrap tensors ──────────────────────────────────────────────────
        self.root_states    = gymtorch.wrap_tensor(_actor_root_state)
        self.dof_states     = gymtorch.wrap_tensor(_dof_state)
        self.contact_forces = gymtorch.wrap_tensor(_net_contact_forces).view(
            self.num_envs, -1, 3
        )

        # ── convenient views ──────────────────────────────────────────────
        self.dof_pos      = self.dof_states.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel      = self.dof_states.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat    = self.root_states[:, 3:7]
        self.base_lin_vel = self.root_states[:, 7:10]
        self.base_ang_vel = self.root_states[:, 10:13]

        # ── initial root states for reset ─────────────────────────────────
        self.initial_root_states = self.root_states.clone()
        self.initial_root_states[:, 7:13] = 0.0

        # ── gravity vector (world frame, pointing down in Z) ──────────────
        # Used to compute projected_gravity in body frame
        self.gravity_vec = torch.tensor(
            [0.0, 0.0, -1.0], dtype=torch.float, device=self.device
        ).repeat((self.num_envs, 1))

        # ── velocity commands [vx, vy, yaw_rate] ─────────────────────────
        #   Initialised to zero; in training these are re-sampled by the
        #   command manager.  Here we expose a simple interface so they can
        #   be set externally (e.g., by a wrapper or curriculum).
        self.commands = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device
        )

        # ── action buffers ────────────────────────────────────────────────
        self.actions      = torch.zeros(
            self.num_envs, self.num_dof, dtype=torch.float, device=self.device
        )
        self.prev_actions = torch.zeros_like(self.actions)

        # ── previous DOF vel (for acceleration penalty) ───────────────────
        self.prev_dof_vel = torch.zeros_like(self.dof_vel)

        # ── position targets buffer (needed for PD torque approximation) ──
        # τ_approx = Kp * (target - q) - Kd * dq
        # Per-joint Kp and Kd are built later in _build_joint_index_masks.
        self.dof_pos_targets = self.default_dof_pos.unsqueeze(0).repeat(self.num_envs, 1)

        # ── feet contact tracking for air-time rewards ────────────────────
        # Populated after _create_envs once we know the body names
        self.feet_air_time  = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device
        )  # tracks how long each foot has been airborne
        self.feet_contact_time = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device
        )  # tracks current contact time

    # ─────────────────────────────────────────────────────────────────────────
    # Simulation setup
    # ─────────────────────────────────────────────────────────────────────────

    def create_sim(self):
        # Physics parameters from LocomotionVelocityRoughEnvCfg.__post_init__
        self.sim_params.dt       = 0.005   # 200 Hz physics
        self.sim_params.substeps = 1
        self.up_axis_idx = 2               # Z is up

        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id,
            self.physics_engine, self.sim_params
        )
        self._create_ground_plane()
        self._create_envs(
            self.num_envs,
            self.cfg["env"]["envSpacing"],
            int(np.sqrt(self.num_envs))
        )

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal           = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction  = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        plane_params.restitution      = self.plane_restitution
        self.gym.add_ground(self.sim, plane_params)

    # ─────────────────────────────────────────────────────────────────────────
    # Default DOF positions
    # ─────────────────────────────────────────────────────────────────────────

    def _get_default_dof_pos(self):
        """
        Map user-defined default joint angles (in YAML cfg) to the DOF order
        reported by the loaded URDF.  Unknown joints default to 0 rad.
        """
        default_dof_pos = np.zeros(self.num_dof, dtype=np.float32)
        default_angles_cfg = self.cfg["env"].get("defaultJointAngles", {})

        for i, dof_name in enumerate(self.dof_names):
            for key, target_angle in default_angles_cfg.items():
                if key in dof_name:
                    default_dof_pos[i] = target_angle
                    break
            else:
                print(f"[WARNING] Joint '{dof_name}' not in defaultJointAngles — defaulting to 0.0 rad.")

        self.default_dof_pos = to_torch(default_dof_pos, device=self.device)

    # ─────────────────────────────────────────────────────────────────────────
    # Joint index helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _build_joint_index_masks(self):
        """
        Pre-compute boolean masks / index lists for joint groups used in rewards.
        Mirrors SceneEntityCfg body_names patterns from HanuA3RewardsCfg.
        """
        self.idx_hip_yaw   = self._joint_mask("hip_yaw")
        self.idx_hip_pitch = self._joint_mask("hip_pitch")
        self.idx_knee      = self._joint_mask("knee")
        self.idx_ankle     = self._joint_mask("ankle")
        self.idx_neck      = self._joint_mask("neck")
        self.idx_shoulder  = self._joint_mask("shoulder")
        self.idx_elbow     = self._joint_mask("elbow")
        self.idx_wrist     = self._joint_mask("wrist")

        # arm = shoulder | elbow | wrist
        self.idx_arms = self.idx_shoulder | self.idx_elbow | self.idx_wrist

        # legs joints for torques penalty: hip | knee | ankle
        self.idx_leg_torque = self.idx_hip_yaw | self._joint_mask("hip_roll") | \
                              self.idx_hip_pitch | self.idx_knee | self.idx_ankle

        # foot body indices for feet air-time / slide rewards
        # We look for bodies whose name contains "_foot_"
        # The contact_forces tensor is shaped (num_envs, num_bodies, 3)
        self.feet_body_ids = []
        for i, bname in enumerate(self.body_names):
            if "_foot_" in bname:
                self.feet_body_ids.append(i)
        if len(self.feet_body_ids) == 0:
            print("[WARNING] No '_foot_' bodies found — feet rewards will be zero.")

        # groin / glute body indices for undesired_contacts
        self.undesired_body_ids = []
        for i, bname in enumerate(self.body_names):
            if "_groin_" in bname or "_glute_" in bname:
                self.undesired_body_ids.append(i)

        # Termination body ids: every link EXCEPT feet
        # Refer: HanuA3RoughEnvCfgV1: body_names = ["^(?!.*_foot_.*).*"]
        self.termination_body_ids = [
            i for i, bname in enumerate(self.body_names)
            if "_foot_" not in bname
        ]

        # mirror joint pairs for feet_mirror reward (hip/knee/ankle)
        # [[left_idx, right_idx], ...] 
        self.mirror_pairs = self._build_mirror_pairs([
            ("Joint_l_hip_pitch",  "Joint_r_hip_pitch"),
            ("Joint_l_knee_pitch", "Joint_r_knee_pitch"),
            ("Joint_l_ankle_pitch","Joint_r_ankle_pitch"),
        ])

        # Per-joint Kp / Kd tensors for PD torque approximation
        # τ_approx = Kp * (target - q) - Kd * dq
        kp = torch.zeros(self.num_dof, dtype=torch.float, device=self.device)
        kd = torch.zeros(self.num_dof, dtype=torch.float, device=self.device)
        for i, name in enumerate(self.dof_names):
            if "hip_yaw" in name:
                g = self.pd_gains["hip_yaw"]
            elif "hip_roll" in name:
                g = self.pd_gains["hip_roll"]
            elif "hip_pitch" in name:
                g = self.pd_gains["hip_pitch"]
            elif "knee" in name:
                g = self.pd_gains["knee"]
            elif "ankle" in name:
                g = self.pd_gains["ankle"]
            else:
                g = self.pd_gains["default"]
            kp[i] = g["stiffness"]
            kd[i] = g["damping"]
        # Shape: (1, num_dof) for broadcasting with (num_envs, num_dof)
        self.dof_stiffness = kp.unsqueeze(0)
        self.dof_damping   = kd.unsqueeze(0)

    def _joint_mask(self, keyword: str) -> torch.Tensor:
        """Boolean mask (num_dof,) True where joint name contains keyword."""
        mask = torch.zeros(self.num_dof, dtype=torch.bool, device=self.device)
        for i, name in enumerate(self.dof_names):
            if keyword in name:
                mask[i] = True
        return mask

    def _build_mirror_pairs(self, pairs):
        """Return list of (left_idx, right_idx) for named joint pairs."""
        result = []
        name_to_idx = {n: i for i, n in enumerate(self.dof_names)}
        for (l_name, r_name) in pairs:
            l_idx = next((i for n, i in name_to_idx.items() if l_name in n), None)
            r_idx = next((i for n, i in name_to_idx.items() if r_name in n), None)
            if l_idx is not None and r_idx is not None:
                result.append((l_idx, r_idx))
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Environment & actor creation
    # ─────────────────────────────────────────────────────────────────────────

    def _create_envs(self, num_envs, spacing, num_per_row):
        asset_root = "/home/jingjaijan/isaacgym/IsaacGymEnvs/assets"
        asset_file = "urdf/hanu_a3_description/urdf/hanu_a3.urdf"

        if "asset" in self.cfg["env"]:
            asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)

        asset_path = os.path.join(asset_root, asset_file)
        asset_root  = os.path.dirname(asset_path)
        asset_file  = os.path.basename(asset_path)

        print(f"[Hanu] Loading asset: {asset_path}")

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode      = gymapi.DOF_MODE_POS
        asset_options.collapse_fixed_joints        = self.cfg["env"]["asset"]["collapseFixedJoints"]
        asset_options.replace_cylinder_with_capsule = self.cfg["env"]["asset"]["replaceCylinderWithCapsule"]
        asset_options.flip_visual_attachments      = False
        asset_options.armature                     = 0.01
        asset_options.thickness                    = 0.001

        hanu_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        self.num_dof  = self.gym.get_asset_dof_count(hanu_asset)
        self.num_body = self.gym.get_asset_rigid_body_count(hanu_asset)

        self.dof_names  = self.gym.get_asset_dof_names(hanu_asset)
        self.body_names = self.gym.get_asset_rigid_body_names(hanu_asset)

        self._get_default_dof_pos()

        # Environment grid bounds
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        # Setup Init Pose
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.cfg["env"]["baseInitState"]["pos"])
        start_pose.r = gymapi.Quat(*self.cfg["env"]["baseInitState"]["rot"])

        self.envs = []
        self.actor_handles = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []

        for i in range(num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            actor_handle = self.gym.create_actor(
                env_ptr, hanu_asset, start_pose, "hanu_a3", i, 1, 0
            )

            # ── PD gain assignment ─────────────────────────────────────
            dof_props = self.gym.get_actor_dof_properties(env_ptr, actor_handle)

            for j, dof_name in enumerate(self.dof_names):
                dof_props["driveMode"][j] = gymapi.DOF_MODE_POS

                if "hip_yaw" in dof_name:
                    g = self.pd_gains["hip_yaw"]
                elif "hip_roll" in dof_name:
                    g = self.pd_gains["hip_roll"]
                elif "hip_pitch" in dof_name:
                    g = self.pd_gains["hip_pitch"]
                elif "knee" in dof_name:
                    g = self.pd_gains["knee"]
                elif "ankle" in dof_name:
                    g = self.pd_gains["ankle"]
                else:
                    g = self.pd_gains["default"]

                dof_props["stiffness"][j] = g["stiffness"]
                dof_props["damping"][j]   = g["damping"]
                dof_props["effort"][j]    = g["effort"]

                if i == 0:
                    self.dof_limits_lower.append(dof_props["lower"][j])
                    self.dof_limits_upper.append(dof_props["upper"][j])

            self.gym.set_actor_dof_properties(env_ptr, actor_handle, dof_props)
            self.envs.append(env_ptr)
            self.actor_handles.append(actor_handle)

        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)

        # Build joint masks now that we know the DOF names
        self._build_joint_index_masks()

        # Resize feet buffers to actual number of foot bodies
        n_feet = max(len(self.feet_body_ids), 1)
        self.feet_air_time    = torch.zeros(self.num_envs, n_feet, dtype=torch.float, device=self.device)
        self.feet_contact_time = torch.zeros(self.num_envs, n_feet, dtype=torch.float, device=self.device)
        self.feet_in_contact_prev = torch.zeros(self.num_envs, n_feet, dtype=torch.bool, device=self.device)

    # ─────────────────────────────────────────────────────────────────────────
    # Simulation step
    # ─────────────────────────────────────────────────────────────────────────

    def pre_physics_step(self, actions: torch.Tensor):
        self.prev_dof_vel[:] = self.dof_vel.clone()
        self.prev_actions[:] = self.actions.clone()
        self.actions = actions.clone().to(self.device)

        # Scale and offset: target = action_scale * action + default_pos
        targets = self.actions * self.action_scale + self.default_dof_pos
        targets = torch.clip(targets, self.dof_limits_lower, self.dof_limits_upper).contiguous()
        # Store targets so compute_reward can approximate joint torques
        self.dof_pos_targets[:] = targets

        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(targets))

    def post_physics_step(self):
        self.progress_buf += 1

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # Update feet contact timing (needed for air-time rewards)
        self._update_feet_timing()

        self.compute_observations()
        self.compute_reward()

        # DEBUG: print before reset
        n_dones = self.reset_buf.sum().item()
        if self.progress_buf[0] % 100 == 0:
            print(f"step {self.progress_buf[0]} | dones this step: {n_dones}")

        dones_snapshot = self.reset_buf.clone()

        # Reset any environments that are done
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.reset_buf[:] = dones_snapshot

        # Debug visualisation
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)

    # ─────────────────────────────────────────────────────────────────────────
    # Feet timing tracker (needed for feet_air_time positive/negative rewards)
    # ─────────────────────────────────────────────────────────────────────────

    def _update_feet_timing(self):
        """
        Maintain per-foot air-time and contact-time counters.
        Mirrors ContactSensor.data.current_air_time / current_contact_time logic
        from Isaac Lab.
        """
        if not self.feet_body_ids:
            return

        # Contact = any net force > 1 N on foot body
        foot_forces = self.contact_forces[:, self.feet_body_ids, :]          # (N, n_feet, 3)
        in_contact   = foot_forces.norm(dim=-1) > 1.0                        # (N, n_feet)

        dt = self.sim_params.dt * self.control_freq_inv  # decimated step size

        # Update air-time: increment when NOT in contact, reset when landing
        self.feet_air_time = torch.where(
            ~in_contact,
            self.feet_air_time + dt,
            torch.zeros_like(self.feet_air_time)
        )

        # Update contact-time: increment when in contact, reset when taking off
        self.feet_contact_time = torch.where(
            in_contact,
            self.feet_contact_time + dt,
            torch.zeros_like(self.feet_contact_time)
        )

        self.feet_in_contact_prev = in_contact.clone()

    # ─────────────────────────────────────────────────────────────────────────
    # Observations  (HanuA3RoughEnvCfgV1)
    # ─────────────────────────────────────────────────────────────────────────

    def compute_observations(self):
        """
        Observation order (matches hanu_lab PolicyCfg with V1 overrides):
            [0:3]     base_ang_vel        0.25
            [3:6]     projected_gravity   1.0
            [6:9]     velocity_commands   1.0
            [9:39]    joint_pos_rel       1.0    (q - q_default)
            [39:69]   joint_vel_rel       0.05
            [69:99]   last_action         1.0
        Note: base_lin_vel and height_scan are set to None in V1.
        """
        # 1. Base angular velocity in body frame (already body frame from IsaacGym)
        base_ang_vel = self.base_ang_vel * self.obs_scales["ang_vel"]

        # 2. Projected gravity: rotate world gravity vector into body frame
        projected_gravity = quat_rotate_inverse_batched(self.base_quat, self.gravity_vec) \
                            * self.obs_scales["gravity"]

        # 3. Velocity commands [vx, vy, yaw_rate]
        velocity_commands = self.commands * self.obs_scales["commands"]

        # 4. Joint positions relative to default
        dof_pos_rel = (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"]

        # 5. Joint velocities
        dof_vel = self.dof_vel * self.obs_scales["dof_vel"]

        # 6. Last action (action already recorded before physics step)
        last_action = self.actions * self.obs_scales["actions"]

        self.obs_buf = torch.cat(
            [base_ang_vel,
             projected_gravity,
             velocity_commands,
             dof_pos_rel,
             dof_vel,
             last_action],
            dim=-1
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Rewards  (HanuA3RoughEnvCfgV1)
    # ─────────────────────────────────────────────────────────────────────────

    def compute_reward(self):
        # ── shared quantities ──────────────────────────────────────────────
        base_quat    = self.base_quat
        projected_gravity = quat_rotate_inverse_batched(base_quat, self.gravity_vec)

        # Yaw-aligned velocity in body frame (for track_lin_vel_xy)
        q_yaw        = yaw_quat(base_quat)
        vel_yaw      = quat_rotate_inverse_batched(q_yaw, self.base_lin_vel)  # (N,3)

        cmd           = self.commands   # (N,3): [vx, vy, yaw_rate]
        dt            = self.sim_params.dt * self.control_freq_inv

        # ─────────────────────────────────────────────────────────────────
        # 1. Track linear velocity XY (yaw-frame exp)
        # ─────────────────────────────────────────────────────────────────
        lin_vel_err    = torch.sum(torch.square(cmd[:, :2] - vel_yaw[:, :2]), dim=1)
        rew_lin        = torch.exp(-lin_vel_err / self.lin_vel_std ** 2)
        rew_lin       *= self.rew_scales["track_lin_vel_xy"]

        # ─────────────────────────────────────────────────────────────────
        # 2. Track angular velocity Z (world frame exp)
        # ─────────────────────────────────────────────────────────────────
        ang_vel_err    = torch.square(cmd[:, 2] - self.base_ang_vel[:, 2])
        rew_ang        = torch.exp(-ang_vel_err / self.ang_vel_std ** 2)
        rew_ang       *= self.rew_scales["track_ang_vel_z"]

        # ─────────────────────────────────────────────────────────────────
        # 3. Upright orientation  (upright_orientation_l2, weight +3.0)
        #    reward = -||proj_gravity - [0,0,-1]||²
        # ─────────────────────────────────────────────────────────────────
        target_g       = torch.tensor([0.0, 0.0, -1.0], device=self.device)
        deviation      = projected_gravity - target_g
        rew_upright    = -torch.sum(torch.square(deviation), dim=1)
        rew_upright   *= self.rew_scales["upright_orientation"]

        # ─────────────────────────────────────────────────────────────────
        # 4. Feet air-time positive (feet_air_time_positive_biped)
        #    Reward single-stance phases up to threshold.
        # ─────────────────────────────────────────────────────────────────
        moving         = torch.norm(cmd[:, :2], dim=1) > 0.1
        if self.feet_body_ids:
            in_contact     = self.feet_contact_time > 0.0          # (N, n_feet)
            in_mode_time   = torch.where(in_contact, self.feet_contact_time, self.feet_air_time)
            single_stance  = torch.sum(in_contact.int(), dim=1) == 1
            raw_air        = torch.min(
                torch.where(single_stance.unsqueeze(-1), in_mode_time, torch.zeros_like(in_mode_time)),
                dim=1
            )[0]
            rew_air_pos    = torch.clamp(raw_air, max=self.air_time_threshold_pos) * moving.float()
        else:
            rew_air_pos    = torch.zeros(self.num_envs, device=self.device)
        rew_air_pos   *= self.rew_scales["feet_air_time"]

        # ─────────────────────────────────────────────────────────────────
        # 5. Feet air-time negative (feet_air_time_negative_biped)
        #    Penalise swing foot exceeding threshold.
        # ─────────────────────────────────────────────────────────────────
        if self.feet_body_ids:
            in_contact_bool = self.feet_contact_time > 0.0
            swing_air_time = torch.max(
                torch.where(~in_contact_bool, self.feet_air_time, torch.zeros_like(self.feet_air_time)),
                dim=1
            )[0]
            single_stance2 = torch.sum(in_contact_bool.int(), dim=1) == 1
            rew_air_neg    = torch.clamp(swing_air_time - self.air_time_threshold_neg, min=0.0)
            rew_air_neg   *= single_stance2.float() * moving.float()
        else:
            rew_air_neg    = torch.zeros(self.num_envs, device=self.device)
        rew_air_neg   *= self.rew_scales["feet_air_time_neg"]

        # ─────────────────────────────────────────────────────────────────
        # 6. Feet slide  (feet_slide)
        #    foot linear vel while in contact → penalise
        #    (We approximate body vel from root + relative, but here we use
        #     contact_forces as the contact signal and dof as proxy)
        # ─────────────────────────────────────────────────────────────────
        if self.feet_body_ids:
            foot_forces    = self.contact_forces[:, self.feet_body_ids, :]
            foot_in_contact = foot_forces.norm(dim=-1) > 1.0
            # proxy: base linear velocity weighted by contact (no per-body vel in IsaacGym VecTask)
            base_xy_vel_norm = self.base_lin_vel[:, :2].norm(dim=-1, keepdim=True)
            rew_slide      = (base_xy_vel_norm * foot_in_contact.float()).sum(dim=1)
        else:
            rew_slide      = torch.zeros(self.num_envs, device=self.device)
        rew_slide     *= self.rew_scales["feet_slide"]

        # ─────────────────────────────────────────────────────────────────
        # 7. Base z linear velocity penalty  (lin_vel_z_l2)
        # ─────────────────────────────────────────────────────────────────
        rew_lin_z      = torch.square(self.base_lin_vel[:, 2])
        rew_lin_z     *= self.rew_scales["lin_vel_z_l2"]

        # ─────────────────────────────────────────────────────────────────
        # 8. Base xy angular velocity penalty  (ang_vel_xy_l2)
        # ─────────────────────────────────────────────────────────────────
        rew_ang_xy     = torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
        rew_ang_xy    *= self.rew_scales["ang_vel_xy_l2"]

        # ─────────────────────────────────────────────────────────────────
        # 9. DOF torques² for legs  (dof_torques_l2, hip/knee/ankle)
        #    Approximated as τ = Kp*(target - q) - Kd*dq since IsaacGym's
        #    acquire_dof_force_tensor / refresh_dof_force_sensor_tensor are
        #    not available in this build.
        # ─────────────────────────────────────────────────────────────────
        approx_torques = (
            self.dof_stiffness * (self.dof_pos_targets - self.dof_pos)
            - self.dof_damping * self.dof_vel
        )
        leg_torques    = approx_torques[:, self.idx_leg_torque]
        rew_torques    = torch.sum(torch.square(leg_torques), dim=1)
        rew_torques   *= self.rew_scales["dof_torques_l2"]

        # ─────────────────────────────────────────────────────────────────
        # 10. DOF acceleration² for hip_pitch  (dof_acc_l2)
        # ─────────────────────────────────────────────────────────────────
        dof_acc        = (self.dof_vel - self.prev_dof_vel) / dt
        hip_pitch_acc  = dof_acc[:, self.idx_hip_pitch]
        rew_acc        = torch.sum(torch.square(hip_pitch_acc), dim=1)
        rew_acc       *= self.rew_scales["dof_acc_l2"]

        # ─────────────────────────────────────────────────────────────────
        # 11. Action rate L2  (action_rate_l2)
        # ─────────────────────────────────────────────────────────────────
        rew_action_rate = torch.sum(torch.square(self.actions - self.prev_actions), dim=1)
        rew_action_rate *= self.rew_scales["action_rate_l2"]

        # ─────────────────────────────────────────────────────────────────
        # 12. Joint velocity² – hip_yaw  (joint_vel_legs)
        # ─────────────────────────────────────────────────────────────────
        rew_joint_vel_legs = torch.sum(torch.square(self.dof_vel[:, self.idx_hip_yaw]), dim=1)
        rew_joint_vel_legs *= self.rew_scales["joint_vel_legs"]

        # ─────────────────────────────────────────────────────────────────
        # 13. Joint velocity² – neck  (joint_vel_neck)
        # ─────────────────────────────────────────────────────────────────
        rew_joint_vel_neck = torch.sum(torch.square(self.dof_vel[:, self.idx_neck]), dim=1)
        rew_joint_vel_neck *= self.rew_scales["joint_vel_neck"]

        # ─────────────────────────────────────────────────────────────────
        # 14. Joint deviation L1 – arms  (joint_deviation_arms)
        # ─────────────────────────────────────────────────────────────────
        arm_pos_dev = torch.sum(
            torch.abs((self.dof_pos - self.default_dof_pos)[:, self.idx_arms]), dim=1
        )
        rew_arm_dev = arm_pos_dev * self.rew_scales["joint_deviation_arms"]

        # ─────────────────────────────────────────────────────────────────
        # 15. Joint deviation L1 – neck  (joint_deviation_neck)
        # ─────────────────────────────────────────────────────────────────
        neck_pos_dev   = torch.sum(
            torch.abs((self.dof_pos - self.default_dof_pos)[:, self.idx_neck]), dim=1
        )
        rew_neck_dev   = neck_pos_dev * self.rew_scales["joint_deviation_neck"]

        # ─────────────────────────────────────────────────────────────────
        # 16. Ankle joint position limits  (ankle_dof_pos_limits)
        #    Penalise ankle joints that exceed 97.5% of their DOF limits.
        # ─────────────────────────────────────────────────────────────────
        ankle_mask   = self.idx_ankle
        q_ankle      = self.dof_pos[:, ankle_mask]
        lo_ankle     = self.dof_limits_lower[ankle_mask]
        hi_ankle     = self.dof_limits_upper[ankle_mask]
        soft_lo      = lo_ankle + 0.025 * (hi_ankle - lo_ankle)
        soft_hi      = hi_ankle - 0.025 * (hi_ankle - lo_ankle)
        ankle_viol   = torch.sum(
            torch.clamp(q_ankle - soft_hi, min=0.0) +
            torch.clamp(soft_lo - q_ankle, min=0.0),
            dim=1
        )
        rew_ankle_limits = ankle_viol * self.rew_scales["ankle_dof_pos_limits"]

        # ─────────────────────────────────────────────────────────────────
        # 17. Feet mirror (action_mirror) – penalise asymmetry between
        #     left/right leg joint actions (hip/knee/ankle pitch pairs)
        # ─────────────────────────────────────────────────────────────────
        rew_mirror = torch.zeros(self.num_envs, device=self.device)
        for (l_idx, r_idx) in self.mirror_pairs:
            diff = torch.square(
                torch.abs(self.actions[:, l_idx]) -
                torch.abs(self.actions[:, r_idx])
            )
            rew_mirror += diff
        if self.mirror_pairs:
            rew_mirror /= len(self.mirror_pairs)
        # scale by uprightness (mirrors Isaac Lab implementation)
        upright_scale = torch.clamp(-projected_gravity[:, 2], 0.0, 0.7) / 0.7
        rew_mirror   *= upright_scale * self.rew_scales["feet_mirror"]

        # ─────────────────────────────────────────────────────────────────
        # 18. Undesired contacts (groin / glute)
        # ─────────────────────────────────────────────────────────────────
        if self.undesired_body_ids:
            undesired_forces = self.contact_forces[:, self.undesired_body_ids, :]
            rew_undesired = (undesired_forces.norm(dim=-1) > 1.0).any(dim=1).float()
        else:
            rew_undesired = torch.zeros(self.num_envs, device=self.device)
        rew_undesired *= self.rew_scales["undesired_contacts"]

        # ─────────────────────────────────────────────────────────────────
        # 19. Termination penalty (is_terminated)
        #     Triggered when ANY non-foot link contacts the ground.
        #     Mirrors HanuA3RoughEnvCfgV1: body_names = "^(?!.*_foot_.*).*"
        # ─────────────────────────────────────────────────────────────────
        if self.termination_body_ids:
            base_contact = (
                self.contact_forces[:, self.termination_body_ids, :].norm(dim=-1) > 10.0
            ).any(dim=1)
        else:
            base_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        rew_termination = base_contact.float() * self.rew_scales["termination_penalty"]

        # ─────────────────────────────────────────────────────────────────
        # Aggregate all reward terms
        # ─────────────────────────────────────────────────────────────────
        self.rew_buf = (
            rew_lin
            + rew_ang
            + rew_upright
            + rew_air_pos
            + rew_air_neg
            + rew_slide
            + rew_lin_z
            + rew_ang_xy
            + rew_torques
            + rew_acc
            + rew_action_rate
            + rew_joint_vel_legs
            + rew_joint_vel_neck
            + rew_arm_dev
            + rew_neck_dev
            + rew_ankle_limits
            + rew_mirror
            + rew_undesired
            + rew_termination
        )

        # ─────────────────────────────────────────────────────────────────
        # Log individual reward terms to TensorBoard via rl_games extras.
        # rl_games reads self.extras["episode"] each step and writes every
        # key as a scalar under "Episode Rewards/<key>" in TensorBoard.
        # ─────────────────────────────────────────────────────────────────
        self.extras["episode"] = {
            "rew_track_lin_vel_xy":    rew_lin.mean().item(),
            "rew_track_ang_vel_z":     rew_ang.mean().item(),
            "rew_upright_orientation": rew_upright.mean().item(),
            "rew_feet_air_time":       rew_air_pos.mean().item(),
            "rew_feet_air_time_neg":   rew_air_neg.mean().item(),
            "rew_feet_slide":          rew_slide.mean().item(),
            "rew_lin_vel_z_l2":        rew_lin_z.mean().item(),
            "rew_ang_vel_xy_l2":       rew_ang_xy.mean().item(),
            "rew_dof_torques_l2":      rew_torques.mean().item(),
            "rew_dof_acc_l2":          rew_acc.mean().item(),
            "rew_action_rate_l2":      rew_action_rate.mean().item(),
            "rew_joint_vel_legs":      rew_joint_vel_legs.mean().item(),
            "rew_joint_vel_neck":      rew_joint_vel_neck.mean().item(),
            "rew_joint_dev_arms":      rew_arm_dev.mean().item(),
            "rew_joint_dev_neck":      rew_neck_dev.mean().item(),
            "rew_ankle_dof_limits":    rew_ankle_limits.mean().item(),
            "rew_feet_mirror":         rew_mirror.mean().item(),
            "rew_undesired_contacts":  rew_undesired.mean().item(),
            "rew_termination_penalty": rew_termination.mean().item(),
            "rew_total":               self.rew_buf.mean().item(),
        }

        # ─────────────────────────────────────────────────────────────────
        # Reset condition (HanuA3TerminationsCfg)
        #   • time_out  : progress >= max_episode_length
        #   • base_contact: base_link / base_* bodies touched the ground
        # ─────────────────────────────────────────────────────────────────
        timeout      = self.progress_buf >= self.max_episode_length - 1
        self.reset_buf = torch.where(
            timeout | base_contact,
            torch.ones_like(self.reset_buf),
            torch.zeros_like(self.reset_buf)
        )
        self.extras["time_outs"] = timeout.to(self.device)

    # ─────────────────────────────────────────────────────────────────────────
    # Reset
    # ─────────────────────────────────────────────────────────────────────────

    def reset_idx(self, env_ids: torch.Tensor):
        """
        Reset environments at the given indices.

        Joint reset: position_range (0.5, 1.5)  default  (HanuA3RoughEnvCfgV1)
        Base reset:  uniform pose xyr in ±0.5 m / ±π,     vel = 0
        """
        n = len(env_ids)

        # ── joint state reset ─────────────────────────────────────────────
        pos_scale  = torch_rand_float(0.5, 1.5, (n, self.num_dof), device=self.device)
        positions  = pos_scale * self.default_dof_pos.unsqueeze(0)
        positions  = torch.clip(positions, self.dof_limits_lower, self.dof_limits_upper)

        self.dof_pos[env_ids] = positions
        self.dof_vel[env_ids] = 0.0

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            n
        )

        # ── base state reset ──────────────────────────────────────────────
        self.root_states[env_ids] = self.initial_root_states[env_ids]

        # Random yaw orientation (uniform [-π, π])
        yaw_offsets = torch_rand_float(-3.14159, 3.14159, (n, 1), device=self.device).squeeze(-1)
        half_yaw    = yaw_offsets * 0.5
        zeros       = torch.zeros(n, device=self.device)
        quats       = torch.stack([zeros, zeros, torch.sin(half_yaw), torch.cos(half_yaw)], dim=-1)
        self.root_states[env_ids, 3:7] = quats

        # Random XY position offset ±0.5 m
        xy_offset   = torch_rand_float(-0.5, 0.5, (n, 2), device=self.device)
        self.root_states[env_ids, 0:2] += xy_offset

        # Zero velocity
        self.root_states[env_ids, 7:13] = 0.0

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            n
        )

        # ── buffer reset ───────────────────────────────────────────────────
        self.prev_actions[env_ids]  = 0.0
        self.prev_dof_vel[env_ids]  = 0.0

        if self.feet_body_ids:
            self.feet_air_time[env_ids]     = 0.0
            self.feet_contact_time[env_ids] = 0.0

        self.reset_buf[env_ids]    = 0
        self.progress_buf[env_ids] = 0