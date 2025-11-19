class LLMDetectionException(Exception):

    def __init__(self, *args, token_data):
        self.token_data = token_data
        super().__init__(*args)


class ParsingError(LLMDetectionException):
    """Indicates a parsing error when processing LLM's output"""

    def __init__(self, *args, token_data=None):
        super().__init__(*args, token_data=token_data)


class JsonFormatError(LLMDetectionException):
    """Indicates a parsing error when processing LLM's generated Json"""

    def __init__(self, *args, token_data=None):
        super().__init__(*args, token_data=token_data)


class InvalidMasksError(LLMDetectionException):
    """Indicates that a generated mask is invalid"""
    
    def __init__(self, *args, token_data=None):
        super().__init__(*args, token_data=token_data)
