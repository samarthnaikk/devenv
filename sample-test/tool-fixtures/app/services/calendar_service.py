"""Business logic for the demo calendar service."""


class CalendarService:
    """Coordinates schedule summaries for the demo project."""

    def describe_backend(self) -> str:
        return "Calendar backend exposes job scheduling and reminder summaries."

    def list_features(self) -> list[str]:
        return [
            "job reminders",
            "calendar sync",
            "availability windows",
        ]
