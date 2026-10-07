import numpy as np

from elastra import success

HOLD = 50
DECIMATION = 32


def score(stand, exit_step=None):
    return success.outcome(
        np.asarray(stand, dtype=bool),
        exit_step,
        hold=HOLD,
        physics_steps_per_control_step=DECIMATION,
    )


def test_hold_completes():
    stand = [False] * 10 + [True] * HOLD + [False] * 5
    out = score(stand)
    assert out.success and out.outcome_class == "success"
    assert out.completion_step == 10 + HOLD - 1


def test_interrupted_hold_is_not_a_success():
    stand = [True] * (HOLD - 1) + [False] + [True] * (HOLD - 1)
    out = score(stand)
    assert not out.success and out.outcome_class == "stood_not_completed"
    assert out.longest_stand_steps == HOLD - 1


def test_never_stood():
    assert score([False] * 100).outcome_class == "never_stood"


def test_leaving_the_area_before_completion_fails():
    stand = [True] * HOLD
    completion = HOLD - 1
    # an exit during the control step that completes the hold still counts
    assert not score(stand, exit_step=(completion + 1) * DECIMATION).success
    assert not score(stand, exit_step=1).success
    # an exit after the hold is complete does not
    assert score(stand, exit_step=(completion + 1) * DECIMATION + 1).success


def test_standing_flags(cfg):
    flags = success.standing_flags(
        np.array([0.69, 0.7, 0.8, 0.8]), np.array([1.0, 0.9, 0.89, 0.95]), cfg.criterion
    )
    assert flags.tolist() == [False, True, False, True]
    assert success.hold_steps(cfg.criterion, float(cfg.sim.control_dt_s)) == HOLD
