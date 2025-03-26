from biorobot.brittle_star.environment.directed_locomotion.dual import BrittleStarDirectedLocomotionEnvironment
from biorobot.brittle_star.environment.directed_locomotion.shared import \
    BrittleStarDirectedLocomotionEnvironmentConfiguration
from biorobot.brittle_star.mjcf.arena.aquarium import MJCFAquariumArena
from biorobot.brittle_star.mjcf.morphology.morphology import MJCFBrittleStarMorphology
from biorobot.brittle_star.mjcf.morphology.specification.specification import BrittleStarMorphologySpecification
from moojoco.environment.mjx_env import MJXEnvState
from moojoco.mjcf.arena import ArenaConfiguration
import jax.numpy as jnp

def create_directed_environment(
        morphology_specification: BrittleStarMorphologySpecification,
        arena_configuration: ArenaConfiguration,
        environment_configuration: BrittleStarDirectedLocomotionEnvironmentConfiguration,
        backend: str = "MJX"
) -> BrittleStarDirectedLocomotionEnvironment:
    return BrittleStarDirectedLocomotionEnvironment.from_morphology_and_arena(
        morphology = MJCFBrittleStarMorphology(specification=morphology_specification),
        arena = MJCFAquariumArena(configuration=arena_configuration),
        configuration = environment_configuration,
        backend = backend
    )

def get_target_position(env: MJXEnvState):
    return jnp.concatenate([env.info["xy_target_position"], jnp.array([0.0])])

def get_brittle_star_position(env: MJXEnvState):
    return env.observations["disk_position"]

def get_direction_front_arm(env: MJXEnvState):
    return None

def get_direction_to_target(env: MJXEnvState):
    return env.observations["unit_xy_direction_to_target"]