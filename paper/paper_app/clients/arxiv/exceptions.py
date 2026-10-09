"""ArXiv client exceptions."""


class ArxivError(RuntimeError):
    """Base class for every ArXiv failure."""


class ArxivTransportError(ArxivError):
    """Network/timeout problem while talking to ArXiv."""


class ArxivParseError(ArxivError):
    """The Atom feed was malformed or unexpected."""


class ArxivEmptyResponse(ArxivParseError):
    """ArXiv returned zero entries for a query."""


class ArxivRateLimited(ArxivError):
    """ArXiv replied with HTTP 429."""


class ArxivNotFound(ArxivError):
    """No entry exists for the requested id."""


class ContentUnavailable(ArxivError):
    """Neither an HTML rendering nor a PDF could be retrieved."""