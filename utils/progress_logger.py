import time
import torch
from transformers import TrainerCallback
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')

logger = logging.getLogger(__name__)


class LogCallback(TrainerCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.log_time_interval = 0
        self.current_step = 0
        self.is_training = False
        self.max_steps = -1
        self.first_step_of_run = 0

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            args.logging_steps = 1
            args.logging_strategy = 'steps'
            self.log_time_interval = args.log_time_interval
            if self.log_time_interval > 0:
                logger.info(f'Using log_time_interval {self.log_time_interval} s. This may override logging_step.')
            self.is_training = True
            self.current_step = 0
            self.start_time = time.time()
            self.last_log_time = self.start_time
            self.max_steps = state.max_steps
            self.first_step_of_run = state.global_step
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop('total_flos', None)
        if state.is_local_process_zero:
            if self.is_training:
                current_time = time.time()
                time_diff = current_time - self.last_log_time
                force = len(logs) > 3 or any([k.endswith('runtime') for k in logs])
                if time_diff > self.log_time_interval or self.current_step >= self.max_steps - 1 or force:
                    self.last_log_time = current_time
                    steps_completed = max(self.current_step, 1)
                    steps_since_first = max(1, self.current_step - self.first_step_of_run)
                    remaining_steps = self.max_steps - steps_completed
                    pct_completed = (steps_completed / self.max_steps) * 100
                    time_since_start = current_time - self.start_time
                    remaining_time = (time_since_start / steps_since_first) * remaining_steps
                    update = {'completed': f'{pct_completed:.2f}% ({steps_completed:_} / {self.max_steps:_})', 'remaining time': self.format_duration(remaining_time)}
                    logger.info(str({**logs, **update}))
            else:
                logger.info(str(logs))

    def on_step_end(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            self.current_step = state.global_step
    
    def on_train_end(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            self.is_training = False

    @staticmethod
    def format_duration(seconds):
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f'{int(hours)}:{int(minutes):02}:{int(seconds):02}'
