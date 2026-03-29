"""Deterministic temporal reasoning module.

Resolves temporal expressions to dates/ranges, computes date differences,
and injects pre-computed temporal context into answer prompts so the LLM
doesn't have to do date arithmetic.

No LLM calls — uses dateutil + custom regex patterns only.
"""

import re
from datetime import date, timedelta
from typing import NamedTuple

from dateutil.relativedelta import relativedelta

# ── Data types ───────────────────────────────────────────────────────────────


class DateRange(NamedTuple):
    """A resolved date range (start inclusive, end inclusive)."""

    start: date
    end: date


class DateDiff(NamedTuple):
    """Difference between two dates."""

    days: int
    months: int
    years: int
    human: str  # e.g. "31 days" or "2 months and 5 days"


# ── Temporal expression patterns ─────────────────────────────────────────────

# "X days/weeks/months/years ago"
_RELATIVE_AGO = re.compile(
    r"(\d+)\s+(day|week|month|year)s?\s+ago",
    re.IGNORECASE,
)

# "last week/month/year", "this week/month/year", "next week/month/year"
_RELATIVE_PERIOD = re.compile(
    r"(last|this|next|past|previous)\s+(week|month|year|quarter)",
    re.IGNORECASE,
)

# "in January", "in March 2025", "in 2024"
_MONTH_NAMES_MAP = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "sept": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}
_MONTH_PATTERN = "|".join(_MONTH_NAMES_MAP.keys())
_IN_MONTH_YEAR = re.compile(
    rf"(?:in|during)\s+({_MONTH_PATTERN})(?:\s+(\d{{4}}))?",
    re.IGNORECASE,
)

# "in 2024", "since 2023", "from 2022"
_IN_YEAR = re.compile(
    r"(?:in|during|since|from)\s+((?:19|20)\d{2})\b",
    re.IGNORECASE,
)

# "between X and Y" (dates)
_BETWEEN_DATES = re.compile(
    r"between\s+(.+?)\s+and\s+(.+?)(?:\s*[?.!]|$)",
    re.IGNORECASE,
)

# "yesterday", "today", "day before yesterday"
_NAMED_RELATIVE = re.compile(
    r"\b(yesterday|today|day before yesterday|the day before yesterday|"
    r"last night|tonight|tomorrow)\b",
    re.IGNORECASE,
)

# "last Monday", "last Tuesday", etc.
_DAY_NAMES_MAP = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}
_DAY_PATTERN = "|".join(_DAY_NAMES_MAP.keys())
_LAST_DAY = re.compile(
    rf"(?:last|past|previous)\s+({_DAY_PATTERN})",
    re.IGNORECASE,
)

# ISO date: "2025-03-15", "2025/03/15"
_ISO_DATE = re.compile(r"\b((?:19|20)\d{2})[-/](0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])\b")

# "March 15, 2025" or "March 15th 2025" or "15 March 2025"
# The day part uses a lookahead to ensure it's not followed by more digits
# (otherwise "February 2024" would match as "February 20" + leftover "24")
_VERBAL_DATE = re.compile(
    rf"(?:({_MONTH_PATTERN})\s+(\d{{1,2}})(?:st|nd|rd|th)?(?!\d),?\s*(\d{{4}})?|"
    rf"(\d{{1,2}})(?:st|nd|rd|th)?(?!\d)\s+({_MONTH_PATTERN}),?\s*(\d{{4}})?)",
    re.IGNORECASE,
)

# "the past/last N days/weeks/months"
_PAST_N_PERIOD = re.compile(
    r"(?:the\s+)?(?:past|last)\s+(\d+)\s+(day|week|month|year)s?",
    re.IGNORECASE,
)


# ── Core functions ───────────────────────────────────────────────────────────


def resolve_temporal_expression(
    query: str,
    reference_date: date | None = None,
) -> DateRange | date | None:
    """Resolve a temporal expression in a query to a date or date range.

    Args:
        query: The search query containing temporal language.
        reference_date: The "now" date for resolving relative expressions.
            Defaults to today.

    Returns:
        A single date, a DateRange, or None if no temporal expression found.
    """
    if reference_date is None:
        reference_date = date.today()

    # Try each pattern in order of specificity

    # 1. "between X and Y" — must come before individual date patterns
    m = _BETWEEN_DATES.search(query)
    if m:
        d1 = _parse_single_date(m.group(1).strip(), reference_date)
        d2 = _parse_single_date(m.group(2).strip(), reference_date)
        if d1 and d2:
            start, end = (d1, d2) if d1 <= d2 else (d2, d1)
            return DateRange(start, end)

    # 2. ISO dates
    m = _ISO_DATE.search(query)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass

    # 3. Verbal dates: "March 15, 2025" or "15 March 2025"
    m = _VERBAL_DATE.search(query)
    if m:
        try:
            if m.group(1):  # "March 15, 2025"
                month = _MONTH_NAMES_MAP[m.group(1).lower()]
                day_num = int(m.group(2))
                year = int(m.group(3)) if m.group(3) else reference_date.year
            else:  # "15 March 2025"
                day_num = int(m.group(4))
                month = _MONTH_NAMES_MAP[m.group(5).lower()]
                year = int(m.group(6)) if m.group(6) else reference_date.year
            return date(year, month, day_num)
        except (ValueError, KeyError):
            pass

    # 4. Named relative: "yesterday", "today"
    m = _NAMED_RELATIVE.search(query)
    if m:
        word = m.group(1).lower()
        if word == "today" or word == "tonight":
            return reference_date
        elif word in ("yesterday", "last night"):
            return reference_date - timedelta(days=1)
        elif word in ("day before yesterday", "the day before yesterday"):
            return reference_date - timedelta(days=2)
        elif word == "tomorrow":
            return reference_date + timedelta(days=1)

    # 5. "X days/weeks/months/years ago"
    m = _RELATIVE_AGO.search(query)
    if m:
        n = int(m.group(1))
        unit = m.group(2).lower()
        if unit == "day":
            return reference_date - timedelta(days=n)
        elif unit == "week":
            return reference_date - timedelta(weeks=n)
        elif unit == "month":
            return reference_date - relativedelta(months=n)
        elif unit == "year":
            return reference_date - relativedelta(years=n)

    # 6. "the past/last N days/weeks/months"
    m = _PAST_N_PERIOD.search(query)
    if m:
        n = int(m.group(1))
        unit = m.group(2).lower()
        if unit == "day":
            start = reference_date - timedelta(days=n)
        elif unit == "week":
            start = reference_date - timedelta(weeks=n)
        elif unit == "month":
            start = reference_date - relativedelta(months=n)
        elif unit == "year":
            start = reference_date - relativedelta(years=n)
        else:
            return None
        return DateRange(start, reference_date)

    # 7. "last/this/next week/month/year/quarter"
    m = _RELATIVE_PERIOD.search(query)
    if m:
        modifier = m.group(1).lower()
        period = m.group(2).lower()
        return _resolve_relative_period(modifier, period, reference_date)

    # 8. "last Monday", "last Tuesday"
    m = _LAST_DAY.search(query)
    if m:
        target_day = _DAY_NAMES_MAP[m.group(1).lower()]
        current_day = reference_date.weekday()
        days_back = (current_day - target_day) % 7
        if days_back == 0:
            days_back = 7  # "last Monday" when today is Monday means 7 days ago
        return reference_date - timedelta(days=days_back)

    # 9. "in January [2025]"
    m = _IN_MONTH_YEAR.search(query)
    if m:
        month = _MONTH_NAMES_MAP[m.group(1).lower()]
        year = int(m.group(2)) if m.group(2) else reference_date.year
        start = date(year, month, 1)
        if month == 12:
            end = date(year, 12, 31)
        else:
            end = date(year, month + 1, 1) - timedelta(days=1)
        return DateRange(start, end)

    # 10. "in 2024"
    m = _IN_YEAR.search(query)
    if m:
        year = int(m.group(1))
        return DateRange(date(year, 1, 1), date(year, 12, 31))

    return None


def _parse_single_date(text: str, reference_date: date) -> date | None:
    """Parse a single date expression (used within 'between X and Y')."""
    # Try ISO
    m = _ISO_DATE.search(text)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None

    # Try verbal date
    m = _VERBAL_DATE.search(text)
    if m:
        try:
            if m.group(1):
                month = _MONTH_NAMES_MAP[m.group(1).lower()]
                day_num = int(m.group(2))
                year = int(m.group(3)) if m.group(3) else reference_date.year
            else:
                day_num = int(m.group(4))
                month = _MONTH_NAMES_MAP[m.group(5).lower()]
                year = int(m.group(6)) if m.group(6) else reference_date.year
            return date(year, month, day_num)
        except (ValueError, KeyError):
            return None

    # Try named relative
    text_lower = text.lower().strip()
    if text_lower == "today":
        return reference_date
    elif text_lower == "yesterday":
        return reference_date - timedelta(days=1)

    # Try "X days/weeks/months ago"
    m = _RELATIVE_AGO.search(text)
    if m:
        n = int(m.group(1))
        unit = m.group(2).lower()
        if unit == "day":
            return reference_date - timedelta(days=n)
        elif unit == "week":
            return reference_date - timedelta(weeks=n)
        elif unit == "month":
            return reference_date - relativedelta(months=n)
        elif unit == "year":
            return reference_date - relativedelta(years=n)

    return None


def _resolve_relative_period(modifier: str, period: str, reference_date: date) -> DateRange:
    """Resolve 'last/this/next week/month/year/quarter' to a DateRange."""
    if period == "week":
        # ISO week: Monday=0
        start_of_this_week = reference_date - timedelta(days=reference_date.weekday())
        if modifier in ("last", "past", "previous"):
            start = start_of_this_week - timedelta(weeks=1)
            end = start_of_this_week - timedelta(days=1)
        elif modifier == "this":
            start = start_of_this_week
            end = start_of_this_week + timedelta(days=6)
        else:  # next
            start = start_of_this_week + timedelta(weeks=1)
            end = start + timedelta(days=6)

    elif period == "month":
        if modifier in ("last", "past", "previous"):
            first_of_this = reference_date.replace(day=1)
            end = first_of_this - timedelta(days=1)
            start = end.replace(day=1)
        elif modifier == "this":
            start = reference_date.replace(day=1)
            if reference_date.month == 12:
                end = date(reference_date.year, 12, 31)
            else:
                end = date(reference_date.year, reference_date.month + 1, 1) - timedelta(days=1)
        else:  # next
            if reference_date.month == 12:
                start = date(reference_date.year + 1, 1, 1)
                end = date(reference_date.year + 1, 1, 31)
            else:
                start = date(reference_date.year, reference_date.month + 1, 1)
                if reference_date.month + 1 == 12:
                    end = date(reference_date.year, 12, 31)
                else:
                    end = date(reference_date.year, reference_date.month + 2, 1) - timedelta(days=1)

    elif period == "year":
        if modifier in ("last", "past", "previous"):
            start = date(reference_date.year - 1, 1, 1)
            end = date(reference_date.year - 1, 12, 31)
        elif modifier == "this":
            start = date(reference_date.year, 1, 1)
            end = date(reference_date.year, 12, 31)
        else:  # next
            start = date(reference_date.year + 1, 1, 1)
            end = date(reference_date.year + 1, 12, 31)

    elif period == "quarter":
        current_q = (reference_date.month - 1) // 3  # 0-based quarter
        if modifier in ("last", "past", "previous"):
            q = current_q - 1
            year = reference_date.year
            if q < 0:
                q = 3
                year -= 1
        elif modifier == "this":
            q = current_q
            year = reference_date.year
        else:  # next
            q = current_q + 1
            year = reference_date.year
            if q > 3:
                q = 0
                year += 1

        q_start_month = q * 3 + 1
        start = date(year, q_start_month, 1)
        q_end_month = q_start_month + 2
        if q_end_month == 12:
            end = date(year, 12, 31)
        else:
            end = date(year, q_end_month + 1, 1) - timedelta(days=1)
    else:
        # Fallback: return reference date
        return DateRange(reference_date, reference_date)

    return DateRange(start, end)


def compute_date_diff(d1: date, d2: date) -> DateDiff:
    """Compute the difference between two dates.

    Args:
        d1: First date.
        d2: Second date.

    Returns:
        DateDiff with days, months, years, and a human-readable string.
    """
    # Ensure d1 <= d2 for consistent output
    if d1 > d2:
        d1, d2 = d2, d1

    delta = d2 - d1
    total_days = delta.days

    # Use relativedelta for month/year computation
    rd = relativedelta(d2, d1)
    months = rd.years * 12 + rd.months
    years = rd.years

    # Build human-readable string
    parts = []
    if years > 0:
        parts.append(f"{years} year{'s' if years != 1 else ''}")
    remaining_months = rd.months
    if remaining_months > 0:
        parts.append(f"{remaining_months} month{'s' if remaining_months != 1 else ''}")
    remaining_days = rd.days
    if remaining_days > 0 or not parts:
        parts.append(f"{remaining_days} day{'s' if remaining_days != 1 else ''}")

    human = " and ".join(parts) if len(parts) <= 2 else ", ".join(parts[:-1]) + f", and {parts[-1]}"

    return DateDiff(
        days=total_days,
        months=months,
        years=years,
        human=human,
    )


def inject_temporal_context(
    query: str,
    search_results: list[dict],
    reference_date: date | None = None,
) -> str:
    """Pre-compute temporal context and return an enriched context string.

    This string is injected into the answer prompt so the LLM doesn't have
    to do date arithmetic.

    Args:
        query: The user's original query.
        search_results: List of memory dicts with 'document_date' or 'event_date'.
        reference_date: The "now" date for resolving relative expressions.

    Returns:
        A context string with pre-computed date math, or empty string if nothing useful.
    """
    if reference_date is None:
        reference_date = date.today()

    lines = []

    # 1. Resolve the temporal expression in the query
    resolved = resolve_temporal_expression(query, reference_date)
    if resolved:
        if isinstance(resolved, DateRange):
            lines.append(
                f"[Temporal context] Query refers to the period "
                f"{resolved.start.isoformat()} to {resolved.end.isoformat()} "
                f"(reference date: {reference_date.isoformat()})."
            )
        else:
            lines.append(
                f"[Temporal context] Query refers to the date "
                f"{resolved.isoformat()} "
                f"(reference date: {reference_date.isoformat()})."
            )

    # 2. Compute "ago" for each result relative to reference_date
    dated_results = []
    for r in search_results:
        d_str = r.get("event_date") or r.get("document_date")
        if not d_str:
            continue
        try:
            d = date.fromisoformat(d_str[:10])
        except (ValueError, TypeError):
            continue
        diff = compute_date_diff(d, reference_date)
        dated_results.append((d, diff, r))

    if dated_results:
        # Sort by date
        dated_results.sort(key=lambda x: x[0])

        lines.append(f"[Temporal context] Reference date (today): {reference_date.isoformat()}")
        for d, diff, r in dated_results:
            content_preview = (r.get("content") or "")[:80]
            lines.append(
                f'  - {d.isoformat()}: "{content_preview}" '
                f"({diff.human} {'ago' if d < reference_date else 'from now'})"
            )

        # 3. If there are 2+ dated results, compute pairwise diffs for the extremes
        if len(dated_results) >= 2:
            earliest = dated_results[0]
            latest = dated_results[-1]
            span = compute_date_diff(earliest[0], latest[0])
            lines.append(
                f"[Temporal context] Time span of results: "
                f"{span.human} (from {earliest[0].isoformat()} to {latest[0].isoformat()})."
            )

    return "\n".join(lines)


def filter_by_date_window(
    results: list[dict],
    target: date | DateRange,
    window_days: int = 3,
) -> list[dict]:
    """Filter search results to those within a date window of the target.

    Args:
        results: List of memory/fact dicts with 'date', 'document_date', or 'event_date'.
        target: The target date or date range.
        window_days: How many days of slack to allow around the target.

    Returns:
        Filtered list of results.
    """
    if isinstance(target, DateRange):
        window_start = target.start - timedelta(days=window_days)
        window_end = target.end + timedelta(days=window_days)
    else:
        window_start = target - timedelta(days=window_days)
        window_end = target + timedelta(days=window_days)

    filtered = []
    for r in results:
        d_str = r.get("date") or r.get("event_date") or r.get("document_date")
        if not d_str:
            continue
        try:
            d = date.fromisoformat(d_str[:10])
        except (ValueError, TypeError):
            continue
        if window_start <= d <= window_end:
            filtered.append(r)

    return filtered
