### Generated Code ###
# This code was generated from CIE pseudo-code to Python.

# Use of python keywords as variable names in Pseudocode WILL cause errors.
# The name currently_open_files is reserved for file handling.
# The following functions are also definied by default and thus their identifier cannot be used:
#   input_and_convert
#   is_end_of_file

# Comments in the original code have been omitted.

# Needs python 3.10 or higher to run.

# Note: Some constructs may not have direct equivalents in Python.
#   Arrays are represented as tuples with start index and list of elements.
#   Date literals are represented as strings in "DD/MM/YYYY" format.
#   Inputs are read as strings and type conversion is attempted.
#   RAND(high) is implemented using `random.uniform(0, high)`.


from random import uniform
from io import TextIOWrapper

current_open_files : dict[str, TextIOWrapper] = dict()

def input_and_convert():
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

def is_end_of_file(filename):
    # Check if the end of file has been reached for the given file
    file = current_open_files.get(filename)
    if file is None:
        raise Exception(f"File {filename} is not open.")
    current_pos = file.tell()
    file.seek(0, 2)  # Move to end of file
    end_pos = file.tell()
    file.seek(current_pos)
    return current_pos == end_pos