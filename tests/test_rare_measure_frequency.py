import pytest

from utils.frequency import MeasurementContext, frequency_calculator


def test_full_batch_lambda_max_is_rarer_in_rare_mode():
    # For batch_size > 33, default freq = 32 (early)
    step = 32
    ctx_default = MeasurementContext(step_number=step, batch_size=64)
    ctx_rare = MeasurementContext(step_number=step, batch_size=64, rare_measure=True)

    assert frequency_calculator.should_measure('full_batch_lambda_max', ctx_default) is True
    # In rare mode (heavy scaling), should be False at step=32
    assert frequency_calculator.should_measure('full_batch_lambda_max', ctx_rare) is False


def test_full_loss_is_rarer_in_rare_mode():
    step = 32
    ctx_default = MeasurementContext(step_number=step, batch_size=64)
    ctx_rare = MeasurementContext(step_number=step, batch_size=64, rare_measure=True)

    assert frequency_calculator.should_measure('full_loss', ctx_default) is True
    assert frequency_calculator.should_measure('full_loss', ctx_rare) is False


def test_step_sharpness_unchanged_in_rare_mode():
    # step sharpness (single batch Rayleigh quotient) is cheap; keep freq = 16
    step = 16
    ctx_default = MeasurementContext(step_number=step, batch_size=64)
    ctx_rare = MeasurementContext(step_number=step, batch_size=64, rare_measure=True)

    assert frequency_calculator.should_measure('step_sharpness', ctx_default) is True
    assert frequency_calculator.should_measure('step_sharpness', ctx_rare) is True


def test_batch_sharpness_is_lightly_sparsified():
    # batch sharpness uses light sparsification; at step=256 should trigger for both
    step = 256
    ctx_default = MeasurementContext(step_number=step, batch_size=64)
    ctx_rare = MeasurementContext(step_number=step, batch_size=64, rare_measure=True)

    assert frequency_calculator.should_measure('batch_sharpness', ctx_default) is True
    assert frequency_calculator.should_measure('batch_sharpness', ctx_rare) is True


def test_gni_is_rarer_in_rare_mode():
    # base_freq is 2500, so at 2500 default True, rare mode expects False
    step = 2500
    ctx_default = MeasurementContext(step_number=step, batch_size=64)
    ctx_rare = MeasurementContext(step_number=step, batch_size=64, rare_measure=True)

    assert frequency_calculator.should_measure('gni', ctx_default) is True
    assert frequency_calculator.should_measure('gni', ctx_rare) is False
