class CleanLine:
    """
    Class representing a cleaned line of code.

    Attributes:
        raw_line (str): The original line of code.
        line_number (int): The line number in the source file.
        content (str): The cleaned line of code, stripped of leading and trailing whitespace.
    """

    def __init__(self, raw_line: str, line_number: int):
        """
        Initialize the MyClass instance.

        Args:
            raw_line (str): The original line of code.
            line_number (int): The line number in the source file.
        """
        self.raw_line = raw_line
        self.line_number = line_number
        self.content = raw_line.strip()

    def __str__(self) -> str:
        return f"{self.line_number}: {self.content}"


# Backwards-compatible alias (legacy name).
Clean_line = CleanLine
