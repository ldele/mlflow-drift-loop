from driftloop.data.base import DataSource, validate_frame
from driftloop.data.openmeteo import OpenMeteoSource
from driftloop.data.synthetic import SyntheticSource


def replayable_source(profile) -> DataSource | None:
    """The source a profile's history can be re-read from, if any.

    Scoring old models on old windows, which is what the retrospective on both
    UIs does, means re-reading those windows. That only works for a fixed cached
    span, so it is limited to the profiles tied to a place.

    The live schedule is excluded even though it reads Kraków too: it fetches a
    window rolling with the current date rather than the committed cache, so
    re-reading a window from months ago would go to the network for data the
    repository does not hold. Callers degrade rather than fail, and both UIs
    apply this one rule so they cannot disagree about which profiles qualify.
    """
    if getattr(profile, "location", None) is None:
        return None
    source = OpenMeteoSource(profile.location)
    return source if source._cache_path().exists() else None


__all__ = [
    "DataSource",
    "OpenMeteoSource",
    "SyntheticSource",
    "replayable_source",
    "validate_frame",
]
