import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')
logger = logger = logging.getLogger(__name__)


class ProgressLogger:
    def __init__(self, process_name, total_iters, log_time_interval=60):
        self.total_iters = total_iters
        self.name = process_name
        if log_time_interval < 1:
            log_time_interval = 1
            logger.info('log_time_interval must be >= 1, setting to 1')
        self.log_time_interval = log_time_interval
        self.start_time = None
        self.last_log_time = None
        self.last_log_ix = 0

    def __call__(self, ix):
        if ix == 0:
            self.start_time = time.time()
            self.last_log_time = time.time()
            self.log(ix)
        elif ix == self.total_iters - 1:
            self.log(ix)
        elif time.time() - self.last_log_time > self.log_time_interval:
            self.log(ix)

    def log(self, ix):
        pct_complete = ix / self.total_iters * 100
        iters_per_sec = (ix - self.last_log_ix) / (time.time() - self.last_log_time)
        remaining = (time.time() - self.start_time) / (ix + 1) * (self.total_iters - ix - 1)
        remaining = time.strftime('%H:%M:%S', time.gmtime(remaining))
        logger.info('{}: {:_} / {:_} ({:.2f}%) ({:_.2f} iter/sec) (remaining: {})'.format(
            self.name,
            ix,
            self.total_iters,
            pct_complete,
            iters_per_sec,
            remaining,
        ))
        self.last_log_time = time.time()
        self.last_log_ix = ix

    @classmethod
    def wrap_iter(
        cls,
        process_name,
        iterable,
        total_iters,
        log_time_interval=60,
        return_ix=False,
        ):
        progress = cls(process_name, total_iters, log_time_interval)
        if return_ix:
            for ix, item in enumerate(iterable):
                progress(ix)
                yield ix, item
        else:
            for ix, item in enumerate(iterable):
                progress(ix)
                yield item

