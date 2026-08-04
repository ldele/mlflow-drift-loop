from driftloop.data.base import DataSource, validate_frame
from driftloop.data.openmeteo import OpenMeteoSource
from driftloop.data.synthetic import SyntheticSource


def replayable_source(profile) -> DataSource | None:
    """The source a profile's history can be re-read from, if any.

    Anything that scores old models on old windows -- the retrospective the site
    and the dashboard both show -- needs to fetch those windows again, which only
    works where the data is a fixed, cached span, which means the profiles
    tied to a place.

    The live schedule is excluded by design even though it reads Kraków too:
    it fetches a window rolling with the current date rather than the committed
    cache, so re-reading a window from months ago would mean going to the network
    for data the repository does not hold. Callers degrade rather than fail --
    one rule, so the site and the dashboard cannot disagree about which profiles
    get a retrospective.
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
