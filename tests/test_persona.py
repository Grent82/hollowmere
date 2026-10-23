from hollowmere.agents.persona import RANK, affinity_label, update_affinity


def test_affinity_never_overshoots_the_judged_target():
    # from neutral straight to trust: however big the step would be, it stops exactly at the target
    new = update_affinity(RANK["neutral"], "trust", importance=10)
    assert new <= RANK["trust"]


def test_affinity_moves_up_slowly_but_down_fast():
    up = update_affinity(RANK["neutral"], "fond", importance=5) - RANK["neutral"]
    down = RANK["neutral"] - update_affinity(RANK["neutral"], "fear", importance=5)
    assert 0 < up < down  # a scare moves the needle further than an equally important warm talk


def test_affinity_label_rounds_to_the_nearest_stance():
    assert affinity_label(RANK["wary"] + 0.1) == "wary"
    assert affinity_label(RANK["wary"] + 0.9) == "neutral"


def test_a_trivial_talk_moves_affinity_less_than_an_important_one():
    trivial = update_affinity(RANK["neutral"], "fond", importance=1) - RANK["neutral"]
    weighty = update_affinity(RANK["neutral"], "fond", importance=10) - RANK["neutral"]
    assert 0 < trivial < weighty


def test_the_scale_can_escalate_past_trust_into_close_friendship_and_love():
    assert RANK["trust"] < RANK["close"] < RANK["love"]
    new = update_affinity(RANK["trust"], "love", importance=10)
    assert new <= RANK["love"]  # still never overshoots, even at the new top of the scale
    assert affinity_label(RANK["love"]) == "love"
