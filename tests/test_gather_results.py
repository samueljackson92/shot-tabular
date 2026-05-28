from shot_tabular.main import gather_results


def test_all_success():
    jobs = [(1, ""), (2, ""), (3, "")]
    success, errors, warnings = gather_results(iter(jobs), 3)
    assert success == 3
    assert errors == 0
    assert warnings == 0


def test_all_errors():
    jobs = [(1, "fatal error"), (2, "another error")]
    success, errors, warnings = gather_results(iter(jobs), 2)
    assert success == 0
    assert errors == 2
    assert warnings == 0


def test_partial_success_counted_as_warning():
    jobs = [(1, "Partial success with warnings: ne missing")]
    success, errors, warnings = gather_results(iter(jobs), 1)
    assert warnings == 1
    assert success == 0
    assert errors == 0


def test_mixed_results_counted_correctly():
    jobs = [
        (1, ""),
        (2, ""),
        (3, "fatal error"),
        (4, "Partial success with warnings: ne failed"),
    ]
    success, errors, warnings = gather_results(iter(jobs), 4)
    assert success == 2
    assert errors == 1
    assert warnings == 1


def test_empty_jobs():
    success, errors, warnings = gather_results(iter([]), 0)
    assert success == 0
    assert errors == 0
    assert warnings == 0
