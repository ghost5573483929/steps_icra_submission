from baselines.early_stopping import EarlyStopping


def test_early_stopping_resets_patience_after_material_improvement():
    stop=EarlyStopping(patience=2,min_delta=0.1)
    assert stop.update(1.0)==(True,False)
    assert stop.update(0.95)==(False,False)
    assert stop.update(0.8)==(True,False)
    assert stop.update(0.75)==(False,False)
    assert stop.update(0.74)==(False,True)


def test_zero_patience_disables_stopping():
    stop=EarlyStopping(patience=0)
    stop.update(1.0)
    for _ in range(20):
        assert stop.update(1.0)[1] is False

