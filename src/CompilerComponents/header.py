### Generated Code ###
# This code was generated from CIE pseudo-code to Python.

# Use of python keywords as variable names in Pseudocode WILL cause errors.
#
# NOTE (identifier normalization): the compiler lowercases all user-defined identifiers when
# generating Python. Runtime helpers use Capitalized names so they cannot collide with user variables.
# Reserved runtime names:
#   CURRENT_OPEN_FILES
#   InputAndConvert
#   IsEndOfFile

# Comments in the original code have been omitted.

# Needs python 3.10 or higher to run.

# Note: Some constructs may not have direct equivalents in Python.
#   Arrays are represented using the runtime helper class `CIEArray`.
#   Date literals are represented as strings in "DD/MM/YYYY" format.
#   Inputs are read as strings and type conversion is attempted.
#   RAND(high) is implemented using `random.uniform(0, high)`.


from random import uniform
from io import TextIOWrapper


class CIEArray:
    """CIE array runtime helper.

    Supports 1D and 2D arrays with arbitrary lower/upper bounds.

    - 1D: CIEArray(low, high, default)
      Access via: arr[i]
    - 2D: CIEArray(low1, high1, default, low2, high2)
      Access via: arr[i, j]
    """

    def __init__(
        self,
        low1: int,
        high1: int,
        default,
        low2: int | None = None,
        high2: int | None = None,
    ):
        if low2 is None and high2 is None:
            self._dims = 1
            self.low1 = low1
            self.high1 = high1
            self._data = [default] * (high1 - low1 + 1)
            return

        if low2 is None or high2 is None:
            raise ValueError("CIEArray: low2/high2 must be both provided for 2D arrays")

        self._dims = 2
        self.low1 = low1
        self.high1 = high1
        self.low2 = low2
        self.high2 = high2
        width = high2 - low2 + 1
        height = high1 - low1 + 1
        self._data = [[default] * width for _ in range(height)]

    def _index_1d(self, i: int) -> int:
        return i - self.low1

    def _index_2d(self, i: int, j: int) -> tuple[int, int]:
        return i - self.low1, j - self.low2

    def __getitem__(self, key):
        if self._dims == 1:
            i = key
            return self._data[self._index_1d(i)]

        # 2D: Python passes a tuple when indexing like arr[i, j]
        i, j = key
        ii, jj = self._index_2d(i, j)
        return self._data[ii][jj]

    def __setitem__(self, key, value) -> None:
        if self._dims == 1:
            i = key
            self._data[self._index_1d(i)] = value
            return

        i, j = key
        ii, jj = self._index_2d(i, j)
        self._data[ii][jj] = value

    def __repr__(self) -> str:
        if self._dims == 1:
            return f"CIEArray1D([{self.low1}:{self.high1}])"
        return f"CIEArray2D([{self.low1}:{self.high1}], [{self.low2}:{self.high2}])"

CURRENT_OPEN_FILES : dict[str, TextIOWrapper] = dict()

def InputAndConvert():
    user_input = input()
    # Attempt to convert input to appropriate type
    if user_input.lower() == 'true' or user_input.lower() == 'false':
        user_input = user_input.title()
    else:
        try:
            user_input = int(user_input)
        except ValueError:
            try:
                user_input = float(user_input)
            except ValueError:
                pass
    return user_input

def IsEndOfFile(filename):
    # Check if the end of file has been reached for the given file
    file = CURRENT_OPEN_FILES.get(filename)
    if file is None:
        raise Exception(f"File {filename} is not open.")
    current_pos = file.tell()
    file.seek(0, 2)  # Move to end of file
    end_pos = file.tell()
    file.seek(current_pos)
    return current_pos == end_pos