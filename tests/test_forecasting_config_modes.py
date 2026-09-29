"""A Literal cannot be statically derived from a ``tuple[str, ...]``, so
preprocessing_views' Literal mode types are kept in sync with
forecasting_config.INPUT_MODES/TARGET_MODES by hand.
"""
from typing import get_args

import pytest

import preprocessing_views
from forecasting_config import INPUT_MODES, TARGET_MODES


@pytest.mark.parametrize(
    "modes, literal",
    [
        (INPUT_MODES, preprocessing_views.ForecastInputPreprocessing),
        (TARGET_MODES, preprocessing_views.ForecastTargetPreprocessing),
    ],
    ids=["input", "target"],
)
def test_modes_match_preprocessing_views_literal(modes, literal):
    assert set(modes) == set(get_args(literal))
