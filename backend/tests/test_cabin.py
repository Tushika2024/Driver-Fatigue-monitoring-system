import pytest

from fatigue.alerts import CRITICAL, MILD, MODERATE, SEVERE
from fatigue.cabin import (CABIN, CREW, DISPATCH, PRIVATE, VehicleProfile, get_profile, load_profiles, route)

PROFS = load_profiles()


def test_goods_truck_table():
    p = PROFS["solo_goods_truck"]
    a = {l: route(l, "r", p) for l in (MILD, MODERATE, SEVERE, CRITICAL)}
    assert (a[MILD][0].channel, a[MILD][0].style, a[MILD][0].volume) == (CABIN, "chime", "soft")
    assert (a[MODERATE][0].style, a[MODERATE][0].channel) == ("voice", CABIN) and "r" in a[MODERATE][0].message
    assert a[SEVERE][0].style == "alarm" and a[SEVERE][0].volume == "loud" and a[SEVERE][0].repeat_until_ack
    assert [x.channel for x in a[CRITICAL]] == [CABIN]          # nothing higher exists


def test_ambulance_with_patient_table():
    p = PROFS["ambulance_patient"]
    assert route(MILD, "r", p)[0].channel == PRIVATE
    m = route(MODERATE, "r", p)[0]
    assert m.channel == PRIVATE and m.volume == "low" and "r" in m.message
    sev = route(SEVERE, "r", p)
    assert all(x.style != "alarm" for x in sev)                 # no siren next to a patient
    assert {x.channel for x in sev} == {PRIVATE, CREW}
    assert DISPATCH in {x.channel for x in route(CRITICAL, "r", p)}


def test_every_profile_every_level_is_non_empty_and_severe_is_never_suppressed():
    for p in PROFS.values():
        for lvl in (MILD, MODERATE, SEVERE, CRITICAL):
            acts = route(lvl, "reason", p)
            assert acts, (p.name, lvl)
        sev = route(SEVERE, "reason", p)
        # some person is reached at SEVERE: the driver (audio) or crew/dispatch
        assert any(a.channel in (CABIN, PRIVATE, CREW, DISPATCH) for a in sev)
        assert any(a.repeat_until_ack for a in sev)


def test_fallback_is_announced_when_nobody_else_can_be_told():
    p = VehicleProfile("x", others_present=True, second_driver=False, overhearing_harmful=True)
    sev = route(SEVERE, "r", p)
    fall = [a for a in sev if a.channel == CABIN]
    assert fall and fall[0].announce and "cabin audio" in fall[0].announce


def test_dispatch_is_the_redirect_when_no_crew():
    p = VehicleProfile("x", True, False, True, has_dispatch=True)
    assert {a.channel for a in route(SEVERE, "r", p)} == {PRIVATE, DISPATCH}


def test_default_profile_is_private_not_silent():
    p = get_profile(None)
    assert p.overhearing_harmful
    assert route(SEVERE, "r", p)


def test_unknown_profile_errors_clearly():
    with pytest.raises(KeyError, match="Available"):
        get_profile("flying_carpet")


def test_level_none_routes_nothing():
    assert route(0, "r", PROFS["bus"]) == []
