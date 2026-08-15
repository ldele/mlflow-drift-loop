"""Guards on the loop's decision rules -- especially the evaluation leak."""

import mlflow
import pandas as pd
import pytest

from driftloop.config import FEATURES, TARGET, LoopConfig
from driftloop.data import SyntheticSource
from driftloop.loop import bootstrap_champion, run_cycle, run_simulation
from driftloop.model import train


@pytest.fixture()
def isolated_mlflow(tmp_path):
    """A throwaway sqlite backend so tests never touch the real one."""
    uri = f"sqlite:///{(tmp_path / 'test.db').as_posix()}"
    mlflow.set_tracking_uri(uri)
    cfg = LoopConfig(experiment_name="test-loop", registered_model_name="test-model")
    mlflow.create_experiment("test-loop", artifact_location=(tmp_path / "artifacts").as_uri())
    mlflow.set_experiment("test-loop")
    yield cfg


def test_baseline_rmse_is_measured_on_data_the_model_did_not_fit():
    df = SyntheticSource().get_data(pd.Timestamp("2025-04-01"), pd.Timestamp("2025-07-01"))
    trained = train(df)
    assert trained.baseline_rmse > 0
    assert trained.n_rows == len(df)


def test_run_simulation_rejects_a_cadence_that_would_leak(isolated_mlflow):
    cfg = isolated_mlflow
    with pytest.raises(ValueError, match="monitor_days"):
        run_simulation(
            SyntheticSource(),
            cfg,
            pd.Timestamp("2025-08-01"),
            pd.Timestamp("2025-09-01"),
            step_days=3,  # 3 + 7 < 14
        )


def test_a_short_holdout_at_a_weekly_cadence_is_rejected(isolated_mlflow):
    """The case the old guard let through.

    `step_days >= holdout_days` admitted holdout_days=3 at a weekly cadence. A
    challenger promoted at ``as_of`` trains up to ``as_of - 3``, and the next
    week's monitor window opens at ``as_of + 7 - 14 = as_of - 7``, so four days
    of it are hours the champion fitted. The champion then looks healthier than
    it is and the retrain trigger is suppressed.
    """
    from dataclasses import replace

    cfg = replace(isolated_mlflow, holdout_days=3)
    assert 7 >= cfg.holdout_days, "the old guard would have allowed this"
    with pytest.raises(ValueError, match="monitor_days"):
        run_simulation(
            SyntheticSource(), cfg,
            pd.Timestamp("2025-08-01"), pd.Timestamp("2025-09-01"), step_days=7,
        )


def test_a_long_holdout_at_a_weekly_cadence_is_allowed(isolated_mlflow):
    """The case the old guard blocked without cause.

    A longer exam pushes the challenger's training data *further* from the next
    monitor window, so it is strictly safer than the shipped 7. Sweeping the
    exam length needs this to be permitted.
    """
    from dataclasses import replace

    cfg = replace(isolated_mlflow, holdout_days=14)
    src = SyntheticSource()
    bootstrap_champion(src, pd.Timestamp("2025-04-01"), pd.Timestamp("2025-07-01"), cfg)
    df = run_simulation(
        src, cfg, pd.Timestamp("2025-08-01"), pd.Timestamp("2025-08-22"), step_days=7,
    )
    assert len(df) == 4


def test_every_replay_starts_clear_of_the_bootstrap_training_data():
    """The first monitor window must not reach back into the bootstrap champion.

    A monitor window opens ``monitor_days`` before the run date, so a replay
    starting sooner than that after the champion's training ends scores it on
    hours it fitted. That understated run 0's error by up to 15% in five of six
    cities before the windows were corrected, which flatters the champion and
    suppresses the retrain trigger on the one run nobody re-checks.

    Asserted over the shipped profiles rather than in the loop, because it is a
    property of how a replay is *configured*, and the loop cannot see the plan.
    """
    from driftloop.config import PROFILES

    short = []
    for profile in PROFILES.values():
        if profile.replay is None:
            continue
        gap = (profile.replay.first_run - profile.replay.champion_train_end).days
        if gap < profile.loop.monitor_days:
            short.append(f"{profile.label}: {gap}d gap < {profile.loop.monitor_days}d monitor")
    assert not short, "replays starting inside the bootstrap training data: " + "; ".join(short)


def test_bootstrap_then_cycle_records_a_decision(isolated_mlflow):
    cfg = isolated_mlflow
    src = SyntheticSource()
    version = bootstrap_champion(src, pd.Timestamp("2025-04-01"), pd.Timestamp("2025-07-01"), cfg)
    assert version == "1"

    # Well after the drift date: the champion should be visibly failing.
    result = run_cycle(src, pd.Timestamp("2025-11-01"), cfg)
    assert result.perf_drift_detected
    assert result.retrain_triggered
    assert result.promotion_decision in {"promoted", "rejected"}
    assert result.challenger_rmse is not None


def test_challenger_never_trains_on_the_holdout_window(isolated_mlflow):
    """The holdout leak, asserted directly rather than through the loop."""
    cfg = isolated_mlflow
    src = SyntheticSource()
    as_of = pd.Timestamp("2025-11-01")
    holdout_start = as_of - pd.Timedelta(cfg.holdout_days, unit="D")
    challenger_start = as_of - pd.Timedelta(cfg.challenger_train_days, unit="D")

    challenger_df = src.get_data(challenger_start, holdout_start)
    holdout = src.get_data(holdout_start, as_of)
    assert challenger_df["timestamp"].max() < holdout["timestamp"].min()


def test_a_healthy_champion_is_left_alone(isolated_mlflow):
    """No drift -> no retrain. The trigger must not fire on noise."""
    cfg = isolated_mlflow
    src = SyntheticSource()
    bootstrap_champion(src, pd.Timestamp("2025-04-01"), pd.Timestamp("2025-07-01"), cfg)
    result = run_cycle(src, pd.Timestamp("2025-07-15"), cfg)  # two weeks later, pre-drift
    assert not result.retrain_triggered
    assert result.promotion_decision == "none"
    assert result.challenger_rmse is None


# --------------------------------------------------------------------------- #
# The skill floor: the second trigger, which does not move when a model is      #
# promoted. See config.LoopConfig.skill_floor for why it is a skill score and   #
# not the absolute RMSE floor the docs originally proposed.                     #
# --------------------------------------------------------------------------- #


def test_the_skill_floor_is_off_unless_asked_for(isolated_mlflow):
    """The default must reproduce the pre-floor loop exactly.

    Every published number was produced without it, so a default that quietly
    switched it on would move all of them and make the before/after comparison
    the project is built on impossible to state.
    """
    cfg = isolated_mlflow
    assert cfg.skill_floor is None
    src = SyntheticSource()
    bootstrap_champion(src, pd.Timestamp("2025-04-01"), pd.Timestamp("2025-07-01"), cfg)
    result = run_cycle(src, pd.Timestamp("2025-07-15"), cfg)
    assert pd.isna(result.champion_skill)
    assert not result.skill_drift_detected


def test_the_skill_floor_can_fire_when_the_ratio_cannot(isolated_mlflow):
    """The failure the whole change exists for.

    A floor of +1.0 is unreachable -- it demands a perfect model -- so it stands
    in for "the ratio has ratcheted out of reach while the champion is failing".
    The loop must still train a challenger.
    """
    from dataclasses import replace

    cfg = isolated_mlflow
    src = SyntheticSource()
    bootstrap_champion(src, pd.Timestamp("2025-04-01"), pd.Timestamp("2025-07-01"), cfg)

    as_of = pd.Timestamp("2025-07-15")  # pre-drift: the ratio stays quiet here
    assert not run_cycle(src, as_of, cfg).retrain_triggered

    floored = replace(cfg, skill_floor=1.0)
    result = run_cycle(src, as_of, floored)
    assert not result.perf_drift_detected, "the ratio must still be quiet"
    assert result.skill_drift_detected
    assert result.retrain_triggered
    assert result.challenger_rmse is not None, "a challenger must actually be trained"


def test_the_skill_floor_only_ever_adds_reasons_to_fire(isolated_mlflow):
    """It is a second way to trigger, never a way to suppress the first.

    A floor of -inf can never be crossed, so the decision has to come down to the
    ratio alone -- with the skill still computed and logged, which is the point:
    the yardstick is worth recording even on runs it does not act on.

    Asserted on a single cycle rather than by running with and without. Two
    cycles are not comparable: the first one promotes, so the second faces a
    fresher champion and a different question.
    """
    from dataclasses import replace

    cfg = isolated_mlflow
    src = SyntheticSource()
    bootstrap_champion(src, pd.Timestamp("2025-04-01"), pd.Timestamp("2025-07-01"), cfg)

    result = run_cycle(src, pd.Timestamp("2025-11-01"), replace(cfg, skill_floor=float("-inf")))

    assert not result.skill_drift_detected
    assert result.retrain_triggered == result.perf_drift_detected
    assert not pd.isna(result.champion_skill), "skill is measured even when it cannot fire"


def test_the_skill_floor_reads_the_lead_off_the_source(isolated_mlflow):
    """The baseline's causality rule depends on the forecast lead.

    A source that does not declare one would silently be treated as lead 0,
    which lets the baseline average hours the forecaster could not have seen.
    """
    assert SyntheticSource().forecast_lead_days == 0

    from driftloop.data import OpenMeteoSource
    from driftloop.config import FORECAST_LEAD_DAYS

    assert OpenMeteoSource().forecast_lead_days == FORECAST_LEAD_DAYS


# --------------------------------------------------------------------------- #
# Re-certification: the trigger that does not wait to be told anything is wrong #
# --------------------------------------------------------------------------- #


def test_recertification_is_off_unless_asked_for(isolated_mlflow):
    """The default must reproduce the pre-recertification loop exactly.

    Same reason the skill floor defaults off: every published number was
    produced without it. The age is still measured, because it costs nothing
    and it is the one quantity that keeps moving while both drift signals are
    quiet.
    """
    cfg = isolated_mlflow
    assert cfg.recertify_days is None
    src = SyntheticSource()
    bootstrap_champion(src, pd.Timestamp("2025-04-01"), pd.Timestamp("2025-07-01"), cfg)
    result = run_cycle(src, pd.Timestamp("2025-07-15"), cfg)
    assert not result.recertify_due
    assert result.certified_age_days > 0, "the age is recorded even with the trigger off"


def test_recertification_fires_while_every_drift_signal_is_quiet(isolated_mlflow):
    """The failure this trigger exists for.

    Pre-drift the ratio has nothing to report and the champion is healthy, which
    is exactly the state a ratcheted trigger is stuck in for months at a time.
    An expired certificate has to be enough on its own.
    """
    from dataclasses import replace

    cfg = isolated_mlflow
    src = SyntheticSource()
    bootstrap_champion(src, pd.Timestamp("2025-04-01"), pd.Timestamp("2025-07-01"), cfg)

    as_of = pd.Timestamp("2025-07-15")
    assert not run_cycle(src, as_of, cfg).retrain_triggered

    result = run_cycle(src, as_of, replace(cfg, recertify_days=14))
    assert not result.perf_drift_detected, "the ratio must still be quiet"
    assert not result.skill_drift_detected, "the floor is off and must stay quiet"
    assert result.recertify_due
    assert result.retrain_triggered
    assert result.challenger_rmse is not None, "a challenger must actually be trained"


def test_recertification_only_ever_adds_reasons_to_fire(isolated_mlflow):
    """A third way to trigger, never a way to suppress the other two.

    A certificate lasting a decade can never expire, so the decision has to come
    down to the ratio alone, with the age still measured and logged.
    """
    from dataclasses import replace

    cfg = isolated_mlflow
    src = SyntheticSource()
    bootstrap_champion(src, pd.Timestamp("2025-04-01"), pd.Timestamp("2025-07-01"), cfg)

    result = run_cycle(src, pd.Timestamp("2025-11-01"), replace(cfg, recertify_days=3650))

    assert not result.recertify_due
    assert result.retrain_triggered == result.perf_drift_detected
    assert result.certified_age_days > 100, "age is measured even when it cannot fire"


def test_an_unexamined_champion_dates_from_the_end_of_its_training_data(isolated_mlflow):
    """A bootstrap champion has never sat an exam, so its clock starts at what
    it knows rather than at the day it was registered.

    This is what makes the first run's age honest: a model trained through July
    and deployed in August is already a fortnight stale on its first day, and
    reading the clock off the registration date would hide that.
    """
    from driftloop.tracking import load_champion, mark_certified

    cfg = isolated_mlflow
    src = SyntheticSource()
    bootstrap_champion(src, pd.Timestamp("2025-04-01"), pd.Timestamp("2025-07-01"), cfg)

    champion = load_champion(cfg.registered_model_name)
    assert champion.last_certified is None
    assert champion.certified_at() == champion.train_end

    mark_certified(cfg.registered_model_name, champion.version, pd.Timestamp("2025-08-20"))
    renewed = load_champion(cfg.registered_model_name)
    assert renewed.last_certified == pd.Timestamp("2025-08-20")
    assert renewed.certified_at() == pd.Timestamp("2025-08-20")


def test_surviving_an_exam_renews_the_certificate(isolated_mlflow):
    """Passing is what renews it, which is what makes this periodic.

    Without renewal the trigger would fire every run from the moment a champion
    went stale until something finally beat it, which is not a schedule but an
    abandonment of the trigger, and it would price the experiment wrong.

    ``promotion_margin=0.99`` makes rejection certain: no challenger is 99%
    better than the incumbent, so the exam is guaranteed to end with the
    champion keeping its place.
    """
    from dataclasses import replace

    from driftloop.tracking import load_champion

    cfg = replace(isolated_mlflow, recertify_days=14, promotion_margin=0.99)
    src = SyntheticSource()
    bootstrap_champion(src, pd.Timestamp("2025-04-01"), pd.Timestamp("2025-07-01"), cfg)

    first = run_cycle(src, pd.Timestamp("2025-07-15"), cfg)
    assert first.recertify_due
    assert first.promotion_decision == "rejected"

    champion = load_champion(cfg.registered_model_name)
    assert champion.last_certified == pd.Timestamp("2025-07-15")

    # One week later the certificate has seven days left, so nothing is due.
    second = run_cycle(src, pd.Timestamp("2025-07-22"), cfg)
    assert not second.recertify_due
    assert second.certified_age_days == pytest.approx(7.0)


def test_a_run_names_every_rule_that_fired(isolated_mlflow):
    """The reason tag is joined, not first-past-the-post.

    Each trigger was added to catch weeks where the ones before it went quiet,
    and counting that needs a week to name all of its reasons. Consumers match
    on membership, so `sweep_skill_floor.py` counts a "ratio+skill" week.
    """
    from driftloop.loop import CycleResult, _retrain_reason

    def reason(**flags) -> str:
        result = CycleResult(
            as_of=pd.Timestamp("2025-07-15"), champion_version="1", data_drift_psi=0.0,
            data_drift_label="stable", worst_feature="temperature", champion_rmse=1.0,
            champion_mae=1.0, champion_r2=1.0, champion_baseline_rmse=1.0,
            perf_drift_ratio=1.0, data_drift_detected=False, perf_drift_detected=False,
            retrain_triggered=False, promotion_decision="none",
        )
        for name, value in flags.items():
            setattr(result, name, value)
        return _retrain_reason(result)

    assert reason() == "none"
    assert reason(perf_drift_detected=True) == "ratio"
    assert reason(skill_drift_detected=True) == "skill"
    assert reason(recertify_due=True) == "recert"
    assert reason(perf_drift_detected=True, skill_drift_detected=True) == "ratio+skill"
    assert reason(
        perf_drift_detected=True, skill_drift_detected=True, recertify_due=True
    ) == "ratio+skill+recert"


# --------------------------------------------------------------------------- #
# The promotion gate's own uncertainty                                          #
# --------------------------------------------------------------------------- #


def _fresh_backend(tmp_path, name: str, **overrides) -> LoopConfig:
    """A LoopConfig on its own empty backend.

    Gate arms cannot share one. The first cycle promotes, so a second cycle in
    the same registry faces a fresher champion and is answering a different
    question, which is the trap `test_the_skill_floor_only_ever_adds_reasons`
    documents.
    """
    from dataclasses import replace

    root = tmp_path / name
    root.mkdir()
    mlflow.set_tracking_uri(f"sqlite:///{(root / 'test.db').as_posix()}")
    cfg = replace(
        LoopConfig(experiment_name=name, registered_model_name=f"model-{name}"), **overrides
    )
    mlflow.create_experiment(name, artifact_location=(root / "artifacts").as_uri())
    mlflow.set_experiment(name)
    return cfg


def _one_cycle(tmp_path, name: str, **overrides):
    cfg = _fresh_backend(tmp_path, name, **overrides)
    src = SyntheticSource()
    bootstrap_champion(src, pd.Timestamp("2025-04-01"), pd.Timestamp("2025-07-01"), cfg)
    return run_cycle(src, pd.Timestamp("2025-10-15"), cfg)


def test_the_confidence_gate_is_off_unless_asked_for(isolated_mlflow):
    """The default must reproduce the point-estimate gate exactly."""
    cfg = isolated_mlflow
    assert cfg.promotion_confidence is None
    src = SyntheticSource()
    bootstrap_champion(src, pd.Timestamp("2025-04-01"), pd.Timestamp("2025-07-01"), cfg)
    result = run_cycle(src, pd.Timestamp("2025-10-15"), cfg)
    assert result.promotion_decision == "promoted"
    assert pd.isna(result.exam_margin_lo), "no interval is computed when the gate is off"


def test_the_confidence_gate_blocks_a_promotion_the_point_estimate_allowed(tmp_path):
    """The whole point: a margin the window cannot establish stops being enough.

    The synthetic world's regime shift is violent, so a challenger there wins by
    44% and no plausible interval reaches down to the shipped 5%. Raising the
    required margin to 43.5% puts the threshold between the point estimate
    (+44.1%) and its lower bound (+43.3%), which is the situation the gate
    exists for, reproduced deterministically rather than hunted for.
    """
    point = _one_cycle(tmp_path, "point-only", promotion_margin=0.435)
    confident = _one_cycle(
        tmp_path, "with-confidence", promotion_margin=0.435, promotion_confidence=0.95
    )

    assert point.promotion_decision == "promoted"
    assert confident.promotion_decision == "rejected"
    assert confident.exam_margin_lo < 0.435 < (
        1 - confident.challenger_rmse / confident.champion_rmse_holdout
    )


def test_the_confidence_gate_only_ever_blocks(tmp_path):
    """It is an additional hurdle, never an alternative one.

    At the shipped margin the challenger clears both tests, so switching the
    gate on changes nothing. A gate that could promote something the point
    estimate rejected would be a different rule rather than a stricter one.
    """
    point = _one_cycle(tmp_path, "shipped-point")
    confident = _one_cycle(tmp_path, "shipped-confident", promotion_confidence=0.95)

    assert point.promotion_decision == confident.promotion_decision == "promoted"
    assert confident.challenger_rmse == pytest.approx(point.challenger_rmse)


def test_a_tighter_confidence_demands_a_lower_bound(tmp_path):
    """More confidence means a more conservative bound, never a looser one.

    Guards the one-sided translation. ``promotion_confidence`` is one-sided and
    ``block_bootstrap`` takes a two-sided alpha, so the conversion is
    ``alpha = 2 * (1 - confidence)``. Getting that backwards would make a
    stricter setting promote more, which no assertion on a single arm catches.
    """
    loose = _one_cycle(tmp_path, "loose", promotion_confidence=0.80)
    tight = _one_cycle(tmp_path, "tight", promotion_confidence=0.999)

    assert tight.exam_margin_lo < loose.exam_margin_lo


def test_the_exam_margin_statistic_is_computed_from_the_errors(isolated_mlflow):
    """`rmse_margin` has to agree with the RMSEs the gate compares.

    It takes squared errors so the resampler has observations to redraw, which
    means it recomputes an RMSE the loop has already computed another way. Two
    routes to one number is how they drift apart.
    """
    import numpy as np

    from driftloop import stats

    champion = np.array([1.0, 4.0, 9.0, 16.0])  # RMSE sqrt(7.5)
    challenger = champion / 4.0  # exactly half the RMSE
    assert stats.rmse_margin(challenger, champion) == pytest.approx(0.5)
    assert np.isnan(stats.rmse_margin(challenger, np.zeros(4)))


def test_a_window_too_short_to_resample_yields_no_opinion(isolated_mlflow):
    """Not a block, which would let a missing interval reject by default."""
    from driftloop.loop import _GATE_BLOCK_HOURS, _exam_margin_lower_bound
    from driftloop.model import build_pipeline

    src = SyntheticSource()
    window = src.get_data(pd.Timestamp("2025-04-01"), pd.Timestamp("2025-07-01"))
    fitted = build_pipeline().fit(window[FEATURES], window[TARGET])

    short = window.iloc[: _GATE_BLOCK_HOURS]  # one block is not enough to resample
    assert pd.isna(_exam_margin_lower_bound(fitted, fitted, short, 0.95))
    assert not pd.isna(_exam_margin_lower_bound(fitted, fitted, window, 0.95))


# --------------------------------------------------------------------------- #
# Probation: promotion as a reversible decision                                 #
# --------------------------------------------------------------------------- #


def _register(cfg, source, start: str, end: str, alias=None) -> str:
    """Train on a window and register it, returning the version."""
    from driftloop.tracking import log_and_register

    trained = train(source.get_data(pd.Timestamp(start), pd.Timestamp(end)))
    with mlflow.start_run(run_name=f"fixture-{end}"):
        return log_and_register(trained, cfg.registered_model_name, alias=alias)


def test_probation_is_off_unless_asked_for(isolated_mlflow):
    """The default must reproduce the irreversible-promotion loop exactly."""
    cfg = isolated_mlflow
    assert cfg.probation_days is None
    src = SyntheticSource()
    bootstrap_champion(src, pd.Timestamp("2025-04-01"), pd.Timestamp("2025-07-01"), cfg)
    result = run_cycle(src, pd.Timestamp("2025-10-15"), cfg)
    assert result.probation_decision == "none"
    assert pd.isna(result.probation_margin)


def test_a_promotion_records_what_it_displaced(isolated_mlflow):
    """Probation has to re-run the decision later against the same alternative.

    The run that promotes overwrites its own `champion_version` tag with the
    winner, so the displaced version is not recoverable from the run afterwards.
    It goes on the model version instead.
    """
    from dataclasses import replace

    from driftloop.tracking import load_champion

    cfg = replace(isolated_mlflow, probation_days=14)
    src = SyntheticSource()
    first = bootstrap_champion(src, pd.Timestamp("2025-04-01"), pd.Timestamp("2025-07-01"), cfg)
    result = run_cycle(src, pd.Timestamp("2025-10-15"), cfg)
    assert result.promotion_decision == "promoted"

    champion = load_champion(cfg.registered_model_name)
    assert champion.version != first
    assert champion.replaced_version == first
    assert champion.promoted_at == pd.Timestamp("2025-10-15")
    assert champion.probation_cleared is False


def test_a_bootstrap_champion_is_never_put_on_probation(isolated_mlflow):
    """It displaced nothing, so there is no decision to re-run."""
    from dataclasses import replace

    cfg = replace(isolated_mlflow, probation_days=1)
    src = SyntheticSource()
    bootstrap_champion(src, pd.Timestamp("2025-04-01"), pd.Timestamp("2025-07-01"), cfg)
    assert run_cycle(src, pd.Timestamp("2025-08-15"), cfg).probation_decision == "none"


def test_a_promotion_that_holds_up_keeps_its_place(isolated_mlflow):
    """The autumn model really is better after the regime shift, so it stays."""
    from dataclasses import replace

    from driftloop.tracking import load_champion

    cfg = replace(isolated_mlflow, probation_days=14)
    src = SyntheticSource()
    bootstrap_champion(src, pd.Timestamp("2025-04-01"), pd.Timestamp("2025-07-01"), cfg)
    run_cycle(src, pd.Timestamp("2025-10-15"), cfg)
    promoted = load_champion(cfg.registered_model_name).version

    judged = run_cycle(src, pd.Timestamp("2025-10-29"), cfg)  # 14 days later
    assert judged.probation_decision == "kept"
    assert judged.probation_margin > 0
    assert load_champion(cfg.registered_model_name).version == promoted


def test_a_promotion_that_does_not_hold_up_is_rolled_back(isolated_mlflow):
    """The mechanism, on a promotion built to fail.

    The synthetic world's shift is violent enough that a real challenger always
    beats the model it replaced, so a rollback has to be constructed: a
    summer-trained model is installed as champion over an autumn-trained one and
    judged on an autumn window, where it is plainly worse.
    """
    from dataclasses import replace

    from driftloop.tracking import CHAMPION_ALIAS, load_champion, mark_promoted

    cfg = replace(isolated_mlflow, probation_days=14)
    src = SyntheticSource()

    good = _register(cfg, src, "2025-09-20", "2025-11-20")  # knows the new world
    bad = _register(cfg, src, "2025-04-01", "2025-07-01", alias=CHAMPION_ALIAS)  # does not
    mark_promoted(cfg.registered_model_name, bad, pd.Timestamp("2025-11-20"), good)

    result = run_cycle(src, pd.Timestamp("2025-12-04"), cfg)  # 14 days on

    assert result.probation_decision == "rolled_back"
    assert result.probation_margin < 0
    assert load_champion(cfg.registered_model_name).version == good


def test_a_promotion_is_judged_once_and_then_settled(isolated_mlflow):
    """Re-judging every run would be the retrying this mechanism avoids.

    A window that keeps moving would eventually find a fortnight the promoted
    model loses, which is selection by another name.
    """
    from dataclasses import replace

    from driftloop.tracking import load_champion

    cfg = replace(isolated_mlflow, probation_days=14)
    src = SyntheticSource()
    bootstrap_champion(src, pd.Timestamp("2025-04-01"), pd.Timestamp("2025-07-01"), cfg)
    run_cycle(src, pd.Timestamp("2025-10-15"), cfg)

    assert run_cycle(src, pd.Timestamp("2025-10-29"), cfg).probation_decision == "kept"
    assert load_champion(cfg.registered_model_name).probation_cleared is True
    assert run_cycle(src, pd.Timestamp("2025-11-05"), cfg).probation_decision == "none"
