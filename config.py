import jax.numpy as jnp
from biorobot.brittle_star.environment.directed_locomotion.dual import BrittleStarDirectedLocomotionEnvironment

from biorobot.brittle_star.environment.directed_locomotion.shared import \
    BrittleStarDirectedLocomotionEnvironmentConfiguration
from biorobot.brittle_star.mjcf.arena.aquarium import AquariumArenaConfiguration, MJCFAquariumArena
from biorobot.brittle_star.mjcf.morphology.morphology import MJCFBrittleStarMorphology
from biorobot.brittle_star.mjcf.morphology.specification.default import default_brittle_star_morphology_specification

SEED = 42
NUM_ARMS = 5
NUM_SEGMENTS_PER_ARM = 3
NUM_OSCILLATORS_PER_ARM = 2 # Defined by CPG structure assumption

CLOSE_ENOUGH_DISTANCE = 0.2 # Distance to target position to consider it reached
TARGET_REACHED_BONUS = 1.0
MAXIMUM_TIME_BONUS = 1.0 # Bonus for reaching the target in less than max time

NUM_INFERENCES_PER_TRIAL = 5 # Number of inferences per trial
NUM_STEPS_PER_INFERENCE = 30

MAX_STEPS_PER_EPISODE = 300 # Max steps in the inner loop per evaluation
NO_PROGRESS_THRESHOLD = 20 # Steps without improvement before truncating

DEFAULT_TARGET_POSITION = jnp.array([1.25, 0.75, 0.0])
NUM_EVALUATIONS_PER_INDIVIDUAL = 3
TARGET_SAMPLING_RADIUS = 1
FIXED_OMEGA = 4.5

morphology_spec = default_brittle_star_morphology_specification(
    num_arms=NUM_ARMS, num_segments_per_arm=NUM_SEGMENTS_PER_ARM,
    use_p_control=True, use_torque_control=False
)

arena_config = AquariumArenaConfiguration(
    size=(4, 4), # Example size, adjust if needed
    sand_ground_color=False,
    attach_target=True,
    wall_height=1.5,
    wall_thickness=0.1
)

env_config = BrittleStarDirectedLocomotionEnvironmentConfiguration(
    target_distance=2.0, # Example, adjust if needed
    joint_randomization_noise_scale=1.0,
    render_mode="rgb_array", # Not used in pure functions, but part of config
    simulation_time=20, # Example, adjust if needed
    num_physics_steps_per_control_step=10, # Example, adjust if needed
    time_scale=2, # Example, adjust if needed
    camera_ids=[0, 1], # Not used in pure functions
    render_size=(480, 640) # Not used in pure functions
)

def create_environment():
    """Create the environment with the specified configuration."""
    return BrittleStarDirectedLocomotionEnvironment.from_morphology_and_arena(
        morphology=MJCFBrittleStarMorphology(specification=morphology_spec),
        arena=MJCFAquariumArena(configuration=arena_config),
        configuration=env_config, backend="MJX"
    )

CONTROL_TIMESTEP = env_config.control_timestep
