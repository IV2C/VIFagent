from dataclasses import dataclass
@dataclass
class CodeEdit:
    start: int
    end: int
    content: str = None