"""Pure rate-limiting logic for the public demo.

No Streamlit, no API calls, no I/O — deterministic and unit-testable via the
`if __name__ == "__main__":` smoke test below.
"""

PER_SESSION_MAX = 30
PER_DAY_MAX = 500

SESSION_LIMIT_MESSAGE = (
    "You've hit the question limit for this browser session. "
    "Please refresh the page later to keep chatting."
)

DAILY_LIMIT_MESSAGE = (
    "This public demo has hit its daily question limit (a free-tier / cost "
    "guard). Please try again tomorrow."
)


def check_rate_limit(session_count, global_counter, today,
                      per_session_max=PER_SESSION_MAX,
                      per_day_max=PER_DAY_MAX):
    """Decide whether a request should proceed.

    `global_counter` is a mutable dict `{"date": <iso str>, "count": <int>}`,
    shared process-wide (e.g. via `st.cache_resource`). This function mutates
    it in place: resetting it on a new day, and incrementing its count when a
    request is allowed.

    Returns (allowed: bool, message: str | None). The caller is responsible
    for incrementing the per-session count when allowed is True.
    """
    if global_counter.get("date") != today:
        global_counter["date"] = today
        global_counter["count"] = 0

    if session_count >= per_session_max:
        return False, SESSION_LIMIT_MESSAGE

    if global_counter["count"] >= per_day_max:
        return False, DAILY_LIMIT_MESSAGE

    global_counter["count"] += 1
    return True, None


if __name__ == "__main__":
    # 1. A fresh call is allowed and bumps the global count to 1.
    counter = {"date": "2026-07-20", "count": 0}
    allowed, msg = check_rate_limit(0, counter, "2026-07-20")
    assert allowed is True, "fresh call should be allowed"
    assert msg is None, "allowed call should have no message"
    assert counter["count"] == 1, f"expected count 1, got {counter['count']}"
    print("PASS: fresh call allowed, global count bumped to 1")

    # 2. Hitting PER_SESSION_MAX blocks with SESSION_LIMIT_MESSAGE.
    counter = {"date": "2026-07-20", "count": 0}
    allowed, msg = check_rate_limit(PER_SESSION_MAX, counter, "2026-07-20")
    assert allowed is False, "session limit should block"
    assert msg == SESSION_LIMIT_MESSAGE, "should return session limit message"
    assert counter["count"] == 0, "blocked-by-session should not bump global count"
    print("PASS: per-session max blocks with SESSION_LIMIT_MESSAGE")

    # 3. Hitting PER_DAY_MAX blocks with DAILY_LIMIT_MESSAGE.
    counter = {"date": "2026-07-20", "count": PER_DAY_MAX}
    allowed, msg = check_rate_limit(0, counter, "2026-07-20")
    assert allowed is False, "daily limit should block"
    assert msg == DAILY_LIMIT_MESSAGE, "should return daily limit message"
    assert counter["count"] == PER_DAY_MAX, "blocked-by-day should not bump global count further"
    print("PASS: per-day max blocks with DAILY_LIMIT_MESSAGE")

    # 4. A date change resets the global count (previously blocked -> allowed again).
    counter = {"date": "2026-07-20", "count": PER_DAY_MAX}
    allowed, msg = check_rate_limit(0, counter, "2026-07-21")
    assert allowed is True, "new day should reset and allow"
    assert msg is None
    assert counter["date"] == "2026-07-21", "date should be updated"
    assert counter["count"] == 1, f"expected count 1 after reset+increment, got {counter['count']}"
    print("PASS: date change resets global count, request allowed again")

    print("\nAll ratelimit smoke tests passed.")
