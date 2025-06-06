"""Open loop trajectory executor

Copyright 2020 Wenzhao Lian. All rights reserved.

"""
import logging
import time
import threading
from typing import Callable
import numpy as np

from motion import trajectory_planner


class OpenTrajectoryExecutor(object):
  """Open loop trajectory execution.
  """

  def __init__(self, rate: float = 200,
               pos_setter: Callable[[np.ndarray], None] = None,
               vel_setter: Callable[[np.ndarray], None] = None,
               time_handover: float = 1e-8,
               time_func: Callable[[], float] = time.time,
               logging: bool = False):
    self._traj = None
    if pos_setter is not None:
      assert vel_setter is None,\
        "Either position or velocity setter needs to be provided!"
      self._pos_setter = pos_setter
      self._vel_setter = None
    elif vel_setter is not None:
      assert pos_setter is None,\
        "Either position or velocity setter needs to be provided!"
      self._vel_setter = vel_setter
      self._pos_setter = None
    else:
      raise ValueError(
        "Either position or velocity setter needs to be provided!")

    self._rate = rate
    self._dt = 1 / self._rate
    self._time_func = time_func
    self._time_handover = time_handover
    self._should_run = False
    # Adds a small slack additional time on top of execution duration.
    self._slack_time = 0.5
    self._lock = threading.Lock()
    self._logging = logging
    self.log_command = []
    self.finished = True

  @property
  def start_time(self) -> float:
    """Returns the start time of the last trajectory."""
    return self._start_time

  def follow_trajectory(
    self, traj: trajectory_planner.TimeOptimalTrajectoryWrapper,
    timeout: float = None) -> None:
    """Follows a predefined trajectory.

    Parameters
    ----------
    traj : TimeOptimalTrajectoryWrapper
    """
    self._traj = traj
    duration = self._traj.get_duration()
    if np.isnan(duration):
      logging.warning('nan duration in follow_trajectory!')
      return
    if self._pos_setter is None:
      self._ref_getter = self._traj.get_velocity
      self._ref_setter = self._vel_setter
    else:
      self._ref_getter = self._traj.get_position
      self._ref_setter = self._pos_setter

    timeout = duration + self._slack_time if timeout is None else timeout
    assert duration < timeout, "Trajectory " +\
        "execution will take {}, longer than timeout {}!".format(
            duration, timeout)
    self._should_run = True
    self._thread = threading.Thread(target=self._execute)
    self._thread.daemon = True
    self._start_time = self._time_func()
    self._max_time = self._start_time + timeout
    self._thread.start()

    self.log_command = []

  def wait(self, timeout: float = None,
           callback: Callable[[float], None] = None) -> bool:
    """Blocks the exeuction thread until timeout."""
    if self.finished:
      return True
    timeout = (self._traj.get_duration() + self._slack_time
               if timeout is None else timeout)
    cur_time = self._time_func()
    while cur_time - self._start_time < timeout:
      if callback is not None:
        callback(cur_time)
      time.sleep(self._time_handover)
      cur_time = self._time_func()
      if self.finished:
        return True
    self.abort()
    return False

  def _execute(self):
    """Streams the reference points.
    """
    self.finished = None
    t_elapsed = 0

    while self._should_run and t_elapsed < self._traj.get_duration():
      cur_t = self._time_func()
      t = cur_t - self._start_time
      j_target = self._ref_getter(t)
      self._ref_setter(np.array(j_target))
      if self._logging:
        self.log_command.append((t, j_target))
      while self._time_func() < cur_t + self._dt:
        time.sleep(self._time_handover)
      t_elapsed = self._time_func() - self._start_time
    with self._lock:
      self.finished = True

  def get_status(self) -> bool:
    return self.finished

  def abort(self) -> None:
    self._should_run = False
    self._thread.join()
    with self._lock:
      self.finished = False
