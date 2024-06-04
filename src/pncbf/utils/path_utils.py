import inspect
import pathlib


def mkdir(path: pathlib.Path) -> pathlib.Path:
    """Helper function to reduce number of lines of code."""
    path.mkdir(exist_ok=True, parents=True)
    return path


def current_file_dir(up: int = 0) -> pathlib.Path:
    caller_file = pathlib.Path(inspect.stack()[up + 1].filename)
    assert caller_file.exists()
    return caller_file.parent
