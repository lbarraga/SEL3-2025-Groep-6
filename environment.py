from biorobot.brittle_star.environment.directed_locomotion.dual import BrittleStarDirectedLocomotionEnvironment
from biorobot.brittle_star.environment.directed_locomotion.shared import \
    BrittleStarDirectedLocomotionEnvironmentConfiguration
from biorobot.brittle_star.mjcf.arena.aquarium import MJCFAquariumArena
from biorobot.brittle_star.mjcf.morphology.morphology import MJCFBrittleStarMorphology
from biorobot.brittle_star.mjcf.morphology.specification.specification import BrittleStarMorphologySpecification
from moojoco.mjcf.arena import ArenaConfiguration


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
