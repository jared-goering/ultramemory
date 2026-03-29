"""Tests for ultramemory.temporal — deterministic temporal reasoning module."""

import unittest
from datetime import date, timedelta

from ultramemory.temporal import (
    DateRange,
    compute_date_diff,
    filter_by_date_window,
    inject_temporal_context,
    resolve_temporal_expression,
)

REF = date(2025, 3, 29)  # Fixed reference date for reproducible tests


class TestResolveRelativeAgo(unittest.TestCase):
    """Test 'X days/weeks/months/years ago' patterns."""

    def test_10_days_ago(self):
        result = resolve_temporal_expression("What did I buy 10 days ago?", REF)
        self.assertEqual(result, date(2025, 3, 19))

    def test_1_day_ago(self):
        result = resolve_temporal_expression("What happened 1 day ago?", REF)
        self.assertEqual(result, date(2025, 3, 28))

    def test_3_weeks_ago(self):
        result = resolve_temporal_expression("What was I doing 3 weeks ago?", REF)
        self.assertEqual(result, date(2025, 3, 8))

    def test_2_months_ago(self):
        result = resolve_temporal_expression("What happened 2 months ago?", REF)
        self.assertEqual(result, date(2025, 1, 29))

    def test_1_year_ago(self):
        result = resolve_temporal_expression("Events from 1 year ago", REF)
        self.assertEqual(result, date(2024, 3, 29))

    def test_months_ago_crossing_year(self):
        # 6 months ago from March = September previous year
        result = resolve_temporal_expression("6 months ago", date(2025, 3, 1))
        self.assertEqual(result, date(2024, 9, 1))


class TestResolveNamedRelative(unittest.TestCase):
    """Test 'yesterday', 'today', 'day before yesterday'."""

    def test_yesterday(self):
        self.assertEqual(
            resolve_temporal_expression("What happened yesterday?", REF),
            date(2025, 3, 28),
        )

    def test_today(self):
        self.assertEqual(
            resolve_temporal_expression("What's on today?", REF),
            date(2025, 3, 29),
        )

    def test_day_before_yesterday(self):
        self.assertEqual(
            resolve_temporal_expression("the day before yesterday was fun", REF),
            date(2025, 3, 27),
        )

    def test_last_night(self):
        self.assertEqual(
            resolve_temporal_expression("What did I do last night?", REF),
            date(2025, 3, 28),
        )

    def test_tomorrow(self):
        self.assertEqual(
            resolve_temporal_expression("What's planned for tomorrow?", REF),
            date(2025, 3, 30),
        )


class TestResolveRelativePeriod(unittest.TestCase):
    """Test 'last/this/next week/month/year/quarter'."""

    def test_last_week(self):
        result = resolve_temporal_expression("What happened last week?", REF)
        self.assertIsInstance(result, DateRange)
        # REF is Saturday March 29. Start of this week = Monday March 24.
        # Last week = March 17-23
        self.assertEqual(result.start, date(2025, 3, 17))
        self.assertEqual(result.end, date(2025, 3, 23))

    def test_this_month(self):
        result = resolve_temporal_expression("What did I do this month?", REF)
        self.assertIsInstance(result, DateRange)
        self.assertEqual(result.start, date(2025, 3, 1))
        self.assertEqual(result.end, date(2025, 3, 31))

    def test_last_month(self):
        result = resolve_temporal_expression("Events from last month", REF)
        self.assertIsInstance(result, DateRange)
        self.assertEqual(result.start, date(2025, 2, 1))
        self.assertEqual(result.end, date(2025, 2, 28))

    def test_this_year(self):
        result = resolve_temporal_expression("What happened this year?", REF)
        self.assertIsInstance(result, DateRange)
        self.assertEqual(result.start, date(2025, 1, 1))
        self.assertEqual(result.end, date(2025, 12, 31))

    def test_last_year(self):
        result = resolve_temporal_expression("last year events", REF)
        self.assertIsInstance(result, DateRange)
        self.assertEqual(result.start, date(2024, 1, 1))
        self.assertEqual(result.end, date(2024, 12, 31))

    def test_next_month(self):
        result = resolve_temporal_expression("plans for next month", REF)
        self.assertIsInstance(result, DateRange)
        self.assertEqual(result.start, date(2025, 4, 1))
        self.assertEqual(result.end, date(2025, 4, 30))

    def test_last_quarter(self):
        result = resolve_temporal_expression("last quarter performance", REF)
        self.assertIsInstance(result, DateRange)
        # Q1 2025 is current (Jan-Mar). Last quarter = Q4 2024 (Oct-Dec).
        self.assertEqual(result.start, date(2024, 10, 1))
        self.assertEqual(result.end, date(2024, 12, 31))

    def test_this_week(self):
        result = resolve_temporal_expression("this week's schedule", REF)
        self.assertIsInstance(result, DateRange)
        # March 29 is Saturday, start of week = Monday March 24
        self.assertEqual(result.start, date(2025, 3, 24))
        self.assertEqual(result.end, date(2025, 3, 30))


class TestResolvePastNPeriod(unittest.TestCase):
    """Test 'the past/last N days/weeks/months'."""

    def test_past_30_days(self):
        result = resolve_temporal_expression("events in the past 30 days", REF)
        self.assertIsInstance(result, DateRange)
        self.assertEqual(result.start, date(2025, 2, 27))
        self.assertEqual(result.end, REF)

    def test_last_7_days(self):
        result = resolve_temporal_expression("the last 7 days", REF)
        self.assertIsInstance(result, DateRange)
        self.assertEqual(result.start, date(2025, 3, 22))
        self.assertEqual(result.end, REF)

    def test_past_3_months(self):
        result = resolve_temporal_expression("past 3 months", REF)
        self.assertIsInstance(result, DateRange)
        self.assertEqual(result.start, date(2024, 12, 29))
        self.assertEqual(result.end, REF)


class TestResolveLastDayOfWeek(unittest.TestCase):
    """Test 'last Monday', 'last Tuesday'."""

    def test_last_monday(self):
        # REF is Saturday March 29. Last Monday = March 24.
        result = resolve_temporal_expression("What happened last Monday?", REF)
        self.assertEqual(result, date(2025, 3, 24))

    def test_last_friday(self):
        # REF is Saturday March 29. Last Friday = March 28.
        result = resolve_temporal_expression("last Friday meeting", REF)
        self.assertEqual(result, date(2025, 3, 28))

    def test_last_saturday_on_saturday(self):
        # If today is Saturday and we say "last Saturday", it means 7 days ago.
        result = resolve_temporal_expression("last Saturday", REF)
        self.assertEqual(result, date(2025, 3, 22))


class TestResolveISODate(unittest.TestCase):
    """Test ISO date patterns."""

    def test_iso_date_dashes(self):
        result = resolve_temporal_expression("What happened on 2025-03-15?", REF)
        self.assertEqual(result, date(2025, 3, 15))

    def test_iso_date_slashes(self):
        result = resolve_temporal_expression("events on 2024/12/25", REF)
        self.assertEqual(result, date(2024, 12, 25))


class TestResolveVerbalDate(unittest.TestCase):
    """Test verbal date patterns."""

    def test_month_day_year(self):
        result = resolve_temporal_expression("since March 15, 2025", REF)
        self.assertEqual(result, date(2025, 3, 15))

    def test_month_day_no_year(self):
        result = resolve_temporal_expression("on March 15", REF)
        self.assertEqual(result, date(2025, 3, 15))

    def test_day_month_year(self):
        result = resolve_temporal_expression("on 15 March 2025", REF)
        self.assertEqual(result, date(2025, 3, 15))

    def test_month_day_ordinal(self):
        result = resolve_temporal_expression("March 15th, 2025", REF)
        self.assertEqual(result, date(2025, 3, 15))

    def test_abbreviated_month(self):
        result = resolve_temporal_expression("on Jan 5, 2025", REF)
        self.assertEqual(result, date(2025, 1, 5))


class TestResolveInMonthYear(unittest.TestCase):
    """Test 'in January [2025]' patterns."""

    def test_in_january(self):
        result = resolve_temporal_expression("What happened in January?", REF)
        self.assertIsInstance(result, DateRange)
        self.assertEqual(result.start, date(2025, 1, 1))
        self.assertEqual(result.end, date(2025, 1, 31))

    def test_in_february_2024(self):
        result = resolve_temporal_expression("events in February 2024", REF)
        self.assertIsInstance(result, DateRange)
        self.assertEqual(result.start, date(2024, 2, 1))
        self.assertEqual(result.end, date(2024, 2, 29))  # 2024 is a leap year

    def test_in_december(self):
        result = resolve_temporal_expression("What did I do in December?", REF)
        self.assertIsInstance(result, DateRange)
        self.assertEqual(result.start, date(2025, 12, 1))
        self.assertEqual(result.end, date(2025, 12, 31))

    def test_during_march(self):
        result = resolve_temporal_expression("during March", REF)
        self.assertIsInstance(result, DateRange)
        self.assertEqual(result.start, date(2025, 3, 1))
        self.assertEqual(result.end, date(2025, 3, 31))


class TestResolveInYear(unittest.TestCase):
    """Test 'in 2024' patterns."""

    def test_in_2024(self):
        result = resolve_temporal_expression("events in 2024", REF)
        self.assertIsInstance(result, DateRange)
        self.assertEqual(result.start, date(2024, 1, 1))
        self.assertEqual(result.end, date(2024, 12, 31))

    def test_since_2023(self):
        result = resolve_temporal_expression("since 2023", REF)
        self.assertIsInstance(result, DateRange)
        self.assertEqual(result.start, date(2023, 1, 1))
        self.assertEqual(result.end, date(2023, 12, 31))


class TestResolveBetween(unittest.TestCase):
    """Test 'between X and Y' patterns."""

    def test_between_iso_dates(self):
        result = resolve_temporal_expression("between 2025-01-01 and 2025-03-15", REF)
        self.assertIsInstance(result, DateRange)
        self.assertEqual(result.start, date(2025, 1, 1))
        self.assertEqual(result.end, date(2025, 3, 15))

    def test_between_verbal_dates(self):
        result = resolve_temporal_expression("between March 1 and March 15", REF)
        self.assertIsInstance(result, DateRange)
        self.assertEqual(result.start, date(2025, 3, 1))
        self.assertEqual(result.end, date(2025, 3, 15))

    def test_between_reversed_order(self):
        result = resolve_temporal_expression("between 2025-03-15 and 2025-01-01", REF)
        self.assertIsInstance(result, DateRange)
        # Should auto-swap so start < end
        self.assertEqual(result.start, date(2025, 1, 1))
        self.assertEqual(result.end, date(2025, 3, 15))


class TestResolveNoMatch(unittest.TestCase):
    """Test queries with no temporal expression."""

    def test_no_temporal(self):
        self.assertIsNone(resolve_temporal_expression("What is Alice's favorite color?", REF))

    def test_ambiguous_no_match(self):
        self.assertIsNone(resolve_temporal_expression("Tell me about the project", REF))


class TestEdgeCases(unittest.TestCase):
    """Leap years, month boundaries, etc."""

    def test_leap_year_feb_29(self):
        # Feb 29 in a leap year
        ref = date(2024, 3, 1)
        result = resolve_temporal_expression("1 day ago", ref)
        self.assertEqual(result, date(2024, 2, 29))

    def test_month_boundary_31_to_30(self):
        ref = date(2025, 4, 1)
        result = resolve_temporal_expression("yesterday", ref)
        self.assertEqual(result, date(2025, 3, 31))

    def test_end_of_year(self):
        ref = date(2025, 1, 1)
        result = resolve_temporal_expression("yesterday", ref)
        self.assertEqual(result, date(2024, 12, 31))

    def test_leap_year_month_range(self):
        result = resolve_temporal_expression("in February 2024", REF)
        self.assertIsInstance(result, DateRange)
        self.assertEqual(result.end, date(2024, 2, 29))

    def test_non_leap_year_month_range(self):
        result = resolve_temporal_expression("in February 2025", REF)
        self.assertIsInstance(result, DateRange)
        self.assertEqual(result.end, date(2025, 2, 28))


class TestComputeDateDiff(unittest.TestCase):
    """Test date difference computation."""

    def test_same_day(self):
        diff = compute_date_diff(date(2025, 3, 15), date(2025, 3, 15))
        self.assertEqual(diff.days, 0)
        self.assertEqual(diff.months, 0)
        self.assertEqual(diff.years, 0)
        self.assertIn("0 day", diff.human)

    def test_31_days(self):
        diff = compute_date_diff(date(2025, 2, 5), date(2025, 3, 8))
        self.assertEqual(diff.days, 31)
        self.assertIn("1 month", diff.human)

    def test_one_year(self):
        diff = compute_date_diff(date(2024, 1, 1), date(2025, 1, 1))
        self.assertEqual(diff.days, 366)  # 2024 is leap year
        self.assertEqual(diff.years, 1)
        self.assertIn("1 year", diff.human)

    def test_reversed_order(self):
        """compute_date_diff should auto-swap if d1 > d2."""
        diff = compute_date_diff(date(2025, 3, 15), date(2025, 1, 1))
        self.assertEqual(diff.days, 73)
        self.assertGreater(diff.months, 0)

    def test_mixed_years_months_days(self):
        diff = compute_date_diff(date(2023, 6, 15), date(2025, 3, 29))
        self.assertEqual(diff.years, 1)
        self.assertIn("year", diff.human)
        self.assertIn("month", diff.human)

    def test_days_across_months(self):
        diff = compute_date_diff(date(2025, 1, 28), date(2025, 3, 5))
        self.assertEqual(diff.days, 36)


class TestInjectTemporalContext(unittest.TestCase):
    """Test temporal context injection for answer prompts."""

    def test_empty_results(self):
        ctx = inject_temporal_context("What happened?", [], REF)
        self.assertEqual(ctx, "")

    def test_with_resolved_date(self):
        ctx = inject_temporal_context(
            "What did I buy 10 days ago?",
            [{"content": "Bought shoes", "document_date": "2025-03-19"}],
            REF,
        )
        self.assertIn("2025-03-19", ctx)
        self.assertIn("Temporal context", ctx)

    def test_with_date_range(self):
        ctx = inject_temporal_context(
            "What happened last month?",
            [{"content": "Meeting", "document_date": "2025-02-15"}],
            REF,
        )
        self.assertIn("2025-02-01", ctx)
        self.assertIn("2025-02-28", ctx)

    def test_multiple_results_shows_span(self):
        results = [
            {"content": "Event A", "document_date": "2025-01-15"},
            {"content": "Event B", "document_date": "2025-03-10"},
        ]
        ctx = inject_temporal_context("What happened this year?", results, REF)
        self.assertIn("Time span", ctx)
        self.assertIn("2025-01-15", ctx)
        self.assertIn("2025-03-10", ctx)

    def test_ago_computation(self):
        results = [
            {"content": "Bought a book", "document_date": "2025-03-19"},
        ]
        ctx = inject_temporal_context("recent purchases", results, REF)
        self.assertIn("10 days ago", ctx)

    def test_event_date_preferred(self):
        """event_date should be used if present."""
        results = [
            {
                "content": "Wedding",
                "document_date": "2025-03-20",
                "event_date": "2025-03-15",
            },
        ]
        ctx = inject_temporal_context("When was the wedding?", results, REF)
        self.assertIn("2025-03-15", ctx)


class TestFilterByDateWindow(unittest.TestCase):
    """Test date-window filtering of search results."""

    def test_filter_single_date(self):
        results = [
            {"date": "2025-03-19", "content": "match"},
            {"date": "2025-01-01", "content": "no match"},
            {"date": "2025-03-20", "content": "close match"},
        ]
        filtered = filter_by_date_window(results, date(2025, 3, 19), window_days=2)
        self.assertEqual(len(filtered), 2)
        contents = [r["content"] for r in filtered]
        self.assertIn("match", contents)
        self.assertIn("close match", contents)

    def test_filter_date_range(self):
        results = [
            {"date": "2025-02-01", "content": "in range"},
            {"date": "2025-02-15", "content": "in range 2"},
            {"date": "2025-04-01", "content": "out of range"},
        ]
        target = DateRange(date(2025, 2, 1), date(2025, 2, 28))
        filtered = filter_by_date_window(results, target, window_days=1)
        self.assertEqual(len(filtered), 2)

    def test_filter_with_document_date(self):
        results = [
            {"document_date": "2025-03-19", "content": "match"},
        ]
        filtered = filter_by_date_window(results, date(2025, 3, 19), window_days=0)
        self.assertEqual(len(filtered), 1)

    def test_filter_no_dates(self):
        results = [
            {"content": "no date field"},
        ]
        filtered = filter_by_date_window(results, date(2025, 3, 19))
        self.assertEqual(len(filtered), 0)

    def test_window_days_expansion(self):
        results = [
            {"date": "2025-03-15", "content": "5 days away"},
        ]
        # window_days=3 should NOT include 5 days away
        filtered = filter_by_date_window(results, date(2025, 3, 19), window_days=3)
        self.assertEqual(len(filtered), 0)
        # window_days=5 should include it
        filtered = filter_by_date_window(results, date(2025, 3, 19), window_days=5)
        self.assertEqual(len(filtered), 1)


class TestDefaultReferenceDate(unittest.TestCase):
    """Test that reference_date defaults to today when not specified."""

    def test_default_reference(self):
        # "yesterday" without explicit reference should use today
        result = resolve_temporal_expression("yesterday")
        today = date.today()
        self.assertEqual(result, today - timedelta(days=1))

    def test_inject_default_reference(self):
        ctx = inject_temporal_context(
            "yesterday",
            [{"content": "test", "document_date": date.today().isoformat()}],
        )
        self.assertIn(date.today().isoformat(), ctx)


if __name__ == "__main__":
    unittest.main()
