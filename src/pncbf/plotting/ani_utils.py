import pathlib

from matplotlib.animation import FuncAnimation
from rich.progress import Progress


def save_anim(ani: FuncAnimation, path: pathlib.Path):
    pbar = Progress()
    pbar.start()
    task = pbar.add_task("Animating", total=ani._save_count)

    def progress_callback(curr_frame: int, total_frames: int):
        pbar.update(task, advance=1)

    pbar.log("Saving anim to {}...".format(path))
    ani.save(path, progress_callback=progress_callback)
    pbar.stop()
