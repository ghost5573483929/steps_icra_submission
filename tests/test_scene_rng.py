import numpy as np

from categorization import scene_rng


def test_scene_rng_is_deterministic():
    first = scene_rng(42, "example_scene").uniform()
    second = scene_rng(42, "example_scene").uniform()
    assert first == second


def test_scene_rng_pinned_literal():
    """The pinned draw for (42, "example_scene"):

        digest = hashlib.sha256(b"42:example_scene").digest()
        seed = int.from_bytes(digest[:8], "little")  # == 16004830596320769433
        np.random.default_rng(seed).uniform()        # == 0.8929978832292822

    Do NOT update the literal to make this pass: a failure means the seeding
    algorithm changed, and with it the accepted-scene set of published datasets.
    """
    value = scene_rng(42, "example_scene").uniform()
    assert value == 0.8929978832292822


def test_scene_rng_independence():
    base = scene_rng(42, "example_scene").uniform()
    with_suffix = scene_rng(42, "example_scene:accept").uniform()
    other_seed = scene_rng(43, "example_scene").uniform()

    assert base != with_suffix
    assert base != other_seed
    assert with_suffix != other_seed
