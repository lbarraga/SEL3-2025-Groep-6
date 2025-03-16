from biorobot.brittle_star.environment.directed_locomotion.shared import \
        BrittleStarDirectedLocomotionEnvironmentConfiguration
from biorobot.brittle_star.mjcf.arena.aquarium import AquariumArenaConfiguration
from biorobot.brittle_star.mjcf.morphology.specification.default import default_brittle_star_morphology_specification

morphology_specification = default_brittle_star_morphology_specification(
        num_arms=5, num_segments_per_arm=3, use_p_control=True, use_torque_control=False
)

arena_configuration = AquariumArenaConfiguration(
        size=(1.5, 1.5), sand_ground_color=False, attach_target=True, wall_height=1.5, wall_thickness=0.1
)

environment_configuration = BrittleStarDirectedLocomotionEnvironmentConfiguration(
        target_distance=1.2,
        joint_randomization_noise_scale=0.0,
        render_mode="rgb_array",
        simulation_time=20,
        num_physics_steps_per_control_step=10,
        time_scale=2,
        camera_ids=[0, 1],
        render_size=(480, 640)
)