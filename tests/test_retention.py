from tapyolo.validate import meets_map_target, score_sets


def test_perfect_detections_score_one():
    ground_truth = [[(0, 0.0, 0.0, 10.0, 10.0)]]
    predictions = [[(0, 0.9, 0.0, 0.0, 10.0, 10.0)]]
    stats = score_sets(ground_truth, predictions)
    assert stats["map50"] == 1.0
    assert stats["recall"] == 1.0
    assert meets_map_target(stats["map50"]) is True


def test_missed_detections_do_not_meet_the_target():
    ground_truth = [[(0, 0.0, 0.0, 10.0, 10.0)]]
    stats = score_sets(ground_truth, [[]])
    assert stats["map50"] == 0.0
    assert stats["recall"] == 0.0
    assert meets_map_target(stats["map50"]) is False
    assert meets_map_target(None) is False


def test_partial_match_is_between_zero_and_one():
    ground_truth = [[(0, 0.0, 0.0, 10.0, 10.0), (0, 30.0, 30.0, 40.0, 40.0)]]
    predictions = [[(0, 0.8, 0.0, 0.0, 10.0, 10.0)]]
    stats = score_sets(ground_truth, predictions)
    assert stats["recall"] == 0.5
    assert stats["map50"] == 0.5
    assert meets_map_target(stats["map50"]) is False
