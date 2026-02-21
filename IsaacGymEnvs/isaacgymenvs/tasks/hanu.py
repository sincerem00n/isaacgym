import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *
# from isaacgymenvs.utils.torch_jit_utils import scale, unscale, quat_mul, quat_conjugate, quat_from_angle_axis, \
#     to_torch, get_axis_params, torch_rand_float, tensor_clamp, compute_heading_and_up, compute_rot, normalize_angle

from isaacgymenvs.tasks.base.vec_task import VecTask

class Hanu(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        self.cfg = cfg

        # self.randomization_params = self.cfg["task"]["randomization_params"]
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
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.cfg["env"]["numActions"] = 30

        # Calculate numObservations
        # base_ang_vel(3) + proj_gravity(3) + commands(3) + joint_pos(dofs) + joint_vel(dofs) + actions(dofs)
        self.num_dof = self.cfg["env"]["numActions"]
        self.cfg["env"]["numObservations"] = 9 + (self.num_dof * 3)
        self.cfg["env"]["numPrivilegedObservations"] = 0

        # Configuration Scales (from your HanuA3RoughEnvCfgV1)
        self.action_scale = 0.25
        self.obs_scales = {
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05
        }
        self.num_dof = self.cfg["env"]["numActions"]

        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

        _actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state = self.gym.acquire_dof_state_tensor(self.sim)
        _net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.root_states = gymtorch.wrap_tensor(_actor_root_state)
        self.dof_states = gymtorch.wrap_tensor(_dof_state)
        self.contact_forces = gymtorch.wrap_tensor(_net_contact_forces).view(self.num_envs, -1, 3)

        self.dof_pos = self.dof_states.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_states.view(self.num_envs, self.num_dof, 2)[..., 1]

        self.base_quat = self.root_states[:, 3:7]
        self.base_lin_vel = self.root_states[:, 7:10]
        self.base_ang_vel = self.root_states[:, 10:13]

        # initial root states for reset
        self.initial_root_states = self.root_states.clone()
        self.initial_root_states[:, 7:13] = 0.0  # Force zero starting velocity

        # Buffers (not in example)
        self.commands = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_dof,dtype=torch.float, device=self.device, requires_grad=False)
        self.prev_actions = torch.zeros_like(self.actions)
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        
        # Up vector for projected gravity
        self.gravity_vec = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

    def create_sim(self):
        self.sim_params.dt = 0.005 # from LocomotionVelocityRoughEnvCfg
        self.sim_params.substeps = 1
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        plane_params.restitution = self.plane_restitution
        self.gym.add_ground(self.sim, plane_params)
    
    def _get_default_dof_pos(self):
        """
        Parses the default joint angles from the config and maps them 
        to the exact DOF order of the loaded URDF asset.
        """
        # Initialize a flat numpy array with zeros
        default_dof_pos = np.zeros(self.num_dof, dtype=np.float32)
        
        # Extract the target angles from the YAML config
        default_angles_cfg = self.cfg["env"]["defaultJointAngles"]
        
        # Iterate over the actual URDF joint names
        for i, dof_name in enumerate(self.dof_names):
            found_match = False
            for key, target_angle in default_angles_cfg.items():
                # If the config key (e.g., 'knee_pitch') is a substring of the real joint name
                if key in dof_name:
                    default_dof_pos[i] = target_angle
                    found_match = True
                    break # Stop searching once we find the first matching key
            
            if not found_match:
                # Helpful debugging print if your URDF names change in the future
                print(f"[WARNING] Joint '{dof_name}' not found in defaultJointAngles config. Defaulting to 0.0 rad.")
        
        # Convert to a PyTorch tensor and move it to the GPU
        self.default_dof_pos = to_torch(default_dof_pos, device=self.device)

    def _create_envs(self, num_envs, spacing, num_per_row):

        # asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        asset_root = "/home/jingjaijan/isaacgym/IsaacGymEnvs/assets"
        asset_file = "urdf/hanu_a3_description/urdf/hanu_a3.urdf"
        # /home/jingjaijan/isaacgym/IsaacGymEnvs/assets/hanu_a3_description/urdf/hanu_a3.urdf

        import os
        print(f"CHECKING PATH: {os.path.join(asset_root, asset_file)}")

        if "asset" in self.cfg["env"]:
            asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.collapse_fixed_joints = self.cfg["env"]["asset"]["collapseFixedJoints"]
        asset_options.replace_cylinder_with_capsule = self.cfg["env"]["asset"]["replaceCylinderWithCapsule"]
        asset_options.flip_visual_attachments = False
        asset_options.armature = 0.01
        asset_options.thickness = 0.001

        hanu_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        self.num_dof = self.gym.get_asset_dof_count(hanu_asset)
        self.num_body = self.gym.get_asset_rigid_body_count(hanu_asset)

        dof_names = self.gym.get_asset_dof_names(hanu_asset)

        self.dof_names = dof_names
        self._get_default_dof_pos()

        # Setup env grid
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        # Setup Init Pose
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.cfg["env"]["baseInitState"]["pos"])
        start_pose.r = gymapi.Quat(*self.cfg["env"]["baseInitState"]["rot"])

        self.envs = []
        self.actor_handles = []
        
        # Lists to store limits for action clipping later
        self.dof_limits_lower = []
        self.dof_limits_upper = []

        # Create Environments and Actors
        for i in range(self.num_envs):
            # Create environment
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            
            # Create actor
            actor_handle = self.gym.create_actor(env_ptr, hanu_asset, start_pose, "hanu_a3", i, 1, 0)
            
            # Enable force sensors on feet if needed
            self.gym.enable_actor_dof_force_sensors(env_ptr, actor_handle)

            # --- PD GAIN ASSIGNMENT ---
            # Retrieve the current physical properties of the DOFs
            dof_props = self.gym.get_actor_dof_properties(env_ptr, actor_handle)

            for j, dof_name in enumerate(dof_names):
                # Enforce position control
                dof_props['driveMode'][j] = gymapi.DOF_MODE_POS
                
                # Apply HANU_A3_CFG logic via string matching
                if "hip_yaw" in dof_name or "hip_roll" in dof_name:
                    dof_props['stiffness'][j] = 200.0
                    dof_props['damping'][j] = 5.0
                    dof_props['effort'][j] = 300.0
                elif "hip_pitch" in dof_name or "knee_pitch" in dof_name:
                    dof_props['stiffness'][j] = 250.0
                    dof_props['damping'][j] = 5.0
                    dof_props['effort'][j] = 300.0
                elif "ankle" in dof_name:
                    dof_props['stiffness'][j] = 20.0
                    dof_props['damping'][j] = 2.0
                    dof_props['effort'][j] = 20.0
                elif any(x in dof_name for x in ["shoulder", "elbow", "neck", "abdomen", "E1R", "wrist"]):
                    dof_props['stiffness'][j] = 40.0
                    dof_props['damping'][j] = 10.0
                    dof_props['effort'][j] = 300.0
                else:
                    # Fallback for unclassified joints
                    dof_props['stiffness'][j] = 40.0
                    dof_props['damping'][j] = 10.0
                    dof_props['effort'][j] = 300.0

                # (Optional) Store limits for the first environment to use in tensors later
                if i == 0:
                    self.dof_limits_lower.append(dof_props['lower'][j])
                    self.dof_limits_upper.append(dof_props['upper'][j])

            # Apply the modified properties back to the simulation
            self.gym.set_actor_dof_properties(env_ptr, actor_handle, dof_props)
            # --------------------------

            self.envs.append(env_ptr)
            self.actor_handles.append(actor_handle)

        # Convert limits to tensors for fast GPU access during step
        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)

    def pre_physics_step(self, actions):
        self.prev_actions[:] = self.actions.clone()
        self.actions = actions.clone().to(self.device)
        
        # Apply action scale and clip
        targets = self.actions * self.action_scale + self.default_dof_pos
        targets = torch.clip(targets, -100.0, 100.0).contiguous()
        
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(targets))

    def post_physics_step(self):
        self.progress_buf += 1

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.compute_observations()
        self.compute_reward()

        # Handle resets
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        # debug viz
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)

            points = []
            colors = []
            for i in range(self.num_envs):
                origin = self.gym.get_env_origin(self.envs[i])
                pose = self.root_states[:, 0:3][i].cpu().numpy()
                glob_pos = gymapi.Vec3(origin.x + pose[0], origin.y + pose[1], origin.z + pose[2])
                points.append([glob_pos.x, glob_pos.y, glob_pos.z, glob_pos.x + 4 * self.heading_vec[i, 0].cpu().numpy(),
                               glob_pos.y + 4 * self.heading_vec[i, 1].cpu().numpy(),
                               glob_pos.z + 4 * self.heading_vec[i, 2].cpu().numpy()])
                colors.append([0.97, 0.1, 0.06])
                points.append([glob_pos.x, glob_pos.y, glob_pos.z, glob_pos.x + 4 * self.up_vec[i, 0].cpu().numpy(), glob_pos.y + 4 * self.up_vec[i, 1].cpu().numpy(),
                               glob_pos.z + 4 * self.up_vec[i, 2].cpu().numpy()])
                colors.append([0.05, 0.99, 0.04])

            self.gym.add_lines(self.viewer, None, self.num_envs * 2, points, colors)

    def compute_observations(self):
        # 1. Projected Gravity
        projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        
        # 2. Joint states relative to default
        dof_pos_scaled = (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"]
        dof_vel_scaled = self.dof_vel * self.obs_scales["dof_vel"]
        
        # 3. Base Angular Velocity
        base_ang_vel_scaled = self.base_ang_vel * self.obs_scales["ang_vel"]

        # Concatenate based on HanuA3RoughEnvCfgV1 (base_lin_vel and height_scan are None)
        self.obs_buf = torch.cat((
            base_ang_vel_scaled,
            projected_gravity,
            self.commands,
            dof_pos_scaled,
            dof_vel_scaled,
            self.actions
        ), dim=-1)

    def compute_reward(self):
        # 1. Tracking Linear Velocity XY (Exp)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        rew_track_lin_vel = torch.exp(-lin_vel_error / 0.5) * 1.5  # Weight = 1.5, std = 0.5

        # 2. Tracking Angular Velocity Z (Exp)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        rew_track_ang_vel = torch.exp(-ang_vel_error / 0.5) * 1.0   # Weight = 1.0

        # 3. Upright Orientation
        projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        rew_upright = torch.square(projected_gravity[:, 2]) * 3.0 # Weight = 3.0

        # 4. Action Rate L2
        rew_action_rate = torch.sum(torch.square(self.actions - self.prev_actions), dim=1) * -0.005

        # 5. Base Contact Termination
        # Assuming index 0 is the base link. Penalize if contact force is > 1.0 on base
        base_contact = torch.norm(self.contact_forces[:, 0, :], dim=-1) > 1.0
        rew_termination = base_contact.float() * -200.0
        
        # Aggregate Rewards
        self.rew_buf = rew_track_lin_vel + rew_track_ang_vel + rew_upright + rew_action_rate + rew_termination
        
        # Update Reset Buffer
        self.reset_buf = torch.where(self.progress_buf >= self.max_episode_length - 1, torch.ones_like(self.reset_buf), base_contact.long())

    def reset_idx(self, env_ids):
        # Apply position range (0.5, 1.5) scaling from HanuA3RoughEnvCfgV1 events
        positions = self.default_dof_pos.unsqueeze(0) + torch_rand_float(-0.05, 0.05, (len(env_ids), self.num_dof), device=self.device)
        self.dof_pos[env_ids] = positions
        self.dof_vel[env_ids] = 0.0

        env_ids_int32 = env_ids.to(dtype=torch.int32)

        self.gym.set_dof_state_tensor_indexed(self.sim, 
                                              gymtorch.unwrap_tensor(self.dof_states), 
                                              gymtorch.unwrap_tensor(env_ids_int32), 
                                              len(env_ids_int32))
        
        self.root_states[env_ids] = self.initial_root_states[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim, 
                                                  gymtorch.unwrap_tensor(self.root_states), 
                                                  gymtorch.unwrap_tensor(env_ids_int32), 
                                                  len(env_ids_int32))

        self.prev_actions[env_ids] = 0.0

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0