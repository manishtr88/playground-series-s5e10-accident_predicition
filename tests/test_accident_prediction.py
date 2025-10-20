from Safe_road_game import accident_prediction


def test_accident_prediction_runs():
    # prepare a plausible feature vector matching the expected features list
    features = [
        'urban',   # road_type
        2,         # num_lanes (will be removed in function)
        0.3,       # curvature
        50,        # speed_limit
        'daylight',# lighting
        'clear',   # weather
        True,      # road_signs_present
        True,      # public_road
        15,        # time_of_day (will be removed)
        False,     # holiday
        True,      # school_season (will be removed)
        0          # num_reported_accidents
    ]

    risk = accident_prediction(features)
    assert hasattr(risk, '__iter__') or isinstance(risk, float)
    # ensure numeric value
    val = float(risk[0])
    assert 0.0 <= val <= 1e6  # model could return range; basic sanity check
