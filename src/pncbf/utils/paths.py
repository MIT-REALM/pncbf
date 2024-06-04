import pathlib

from pncbf.utils.path_utils import current_file_dir, mkdir


def get_src_dir() -> pathlib.Path:
    path = current_file_dir().parent.parent
    assert path.name == "src"
    return path


def get_root_dir() -> pathlib.Path:
    return get_src_dir().parent


def get_runs_dir() -> pathlib.Path:
    return mkdir(get_root_dir() / "runs")


def get_plots_dir() -> pathlib.Path:
    return mkdir(get_root_dir() / "plots")


def get_script_plot_dir() -> pathlib.Path:
    return mkdir(current_file_dir(1) / "plots")


def get_data_dir() -> pathlib.Path:
    return get_root_dir() / "data"


def get_paper_data_dir() -> pathlib.Path:
    return get_root_dir() / "paper_data"


def get_paper_plot_dir() -> pathlib.Path:
    return get_root_dir() / "paper_plots"
