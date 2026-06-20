"""Demo calendar backend entrypoint."""

from app.services.calendar_service import CalendarService


def build_summary() -> str:
    service = CalendarService()
    return service.describe_backend()


if __name__ == "__main__":
    print(build_summary())
