import functools as ft

from loguru import logger
from mpire import WorkerPool
from mpire.async_result import AsyncResult


class log_err_wrapper:
    def __init__(self, fn, *args, **kwargs):
        self.inner_fn = fn
        self.args = args
        self.kwargs = kwargs

    def __call__(self):
        try:
            # Use CPU only.
            self.inner_fn(*self.args, **self.kwargs)
        except:
            logger.opt(exception=True).debug("Error occurred in AsyncRun for fn {}!".format(self.inner_fn))


class MPRun:
    def __init__(self):
        self._pool = WorkerPool(n_jobs=4)
        self._results: list[AsyncResult] = []

    @property
    def pool(self):
        return self._pool

    def apply_async(self, fn, *args, **kwargs):
        future = self._pool.apply_async(log_err_wrapper(fn, *args, **kwargs))
        self._results.append(future)
        return future

    def wrap(self, plot_fn):
        @ft.wraps(plot_fn)
        def fn(*args, **kwargs):
            logger.info("len(results): {}, adding!".format(len(self._results)))
            future = self._pool.apply_async(log_err_wrapper(plot_fn, *args, **kwargs))
            self._results.append(future)
            return future

        return fn

    def remove_finished(self):
        self._results = [result for result in self._results if result.ready()]

    def shutdown(self):
        self._pool.stop_and_join()

    def __del__(self):
        self._pool.stop_and_join()
