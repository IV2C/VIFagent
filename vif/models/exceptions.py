class ParsingError(Exception):
    """Indicates a parsing error when processing LLM's output
    """
    pass

class JsonFormatError(Exception):
    """Indicates a parsing error when processing LLM's generated Json
    """
    pass

class InvalidMasksError(Exception):
    """Indicates that a generated mask is invalid
    """