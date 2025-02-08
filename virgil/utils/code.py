import sys
from io import StringIO

start_tag = "<code>"
end_tag = "</code>"


def execute_code_in_string(input_string):
    """
    Execute Python code within  tags and return the output as a string.
    Returns None if parsing fails.
    """

    start_index = input_string.find(start_tag)
    if start_index == -1:
        return None

    end_index = input_string.find(end_tag, start_index + len(start_tag))
    if end_index == -1:
        return None

    code_block = input_string[start_index + len(start_tag):end_index]
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()

    try:
        exec(code_block)
    except Exception as e:
        print(e)
        return None
    finally:
        sys.stdout = old_stdout

    output = mystdout.getvalue()
    return output


if __name__ == '__main__':
    # Example usage
    input_string = """<code>print("Hello, World!"); print(2 + 2)</code>"""

    output = execute_code_in_string(input_string)
    if output:
        print(output)
    else:
        print("Parsing or execution failed.")
