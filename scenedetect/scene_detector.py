# -*- coding: utf-8 -*-
#
#         PySceneDetect: Python-Based Video Scene Detector
#   ---------------------------------------------------------------
#     [  Site: http://www.bcastell.com/projects/PySceneDetect/   ]
#     [  Github: https://github.com/Breakthrough/PySceneDetect/  ]
#     [  Documentation: http://pyscenedetect.readthedocs.org/    ]
#
# Copyright (C) 2014-2021 Brandon Castellano <http://www.bcastell.com>.
#
# PySceneDetect is licensed under the BSD 3-Clause License; see the included
# LICENSE file, or visit one of the following pages for details:
#  - https://github.com/Breakthrough/PySceneDetect/
#  - http://www.bcastell.com/projects/PySceneDetect/
#
# This software uses Numpy, OpenCV, click, tqdm, simpletable, and pytest.
# See the included LICENSE files or one of the above URLs for more information.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
""" ``scenedetect.scene_detector`` Module

This module implements the base SceneDetector class, from which all scene
detectors in the scenedetect.dectectors module are derived from.

The SceneDetector class represents the interface which detection algorithms
are expected to provide in order to be compatible with PySceneDetect.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy

from scenedetect.frame_timecode import FrameTimecode
from scenedetect.stats_manager import StatsManager


class EventType(Enum):
    """Describes the type of event/change that a detector produced while procesing frames.

    Values:
        CUT: Fast cut
        IN: Beginning of a scene/event (e.g. fade-in event, motion start)
        OUT: End of a scene/event (e.g. fade-out event, motion stop)
    """
    CUT = 1
    IN = 2
    OUT = 3


@dataclass
class DetectionEvent:
    """Event data which SceneDetectors objects produce while processing frames.

    Attributes:
        kind (EventType): Type of event which was detected (see `EventType`)
        time (FrameTimecode): Presentation timestamp corresponding to the event
        context (Dict[str, Any]): Data specific to each detector. See each detector's documentation
            for what values it populates (e.g. certain detectors may produce a confidence score).
    """
    kind: EventType
    time: FrameTimecode
    context: Optional[Dict[str, Any]] = None


class SceneDetector(ABC):
    """Interface which a scene detection algorithm must implement."""

    def __init__(self):
        self._stats_manager = None

    @property
    def stats_manager(self) -> Optional[StatsManager]:
        """Returns the StatsManager bound to this object, if any."""
        return self._stats_manager

    @stats_manager.setter
    def stats_manager(self, value: StatsManager):
        """Sets the StatsManager bound to this object, if any."""
        self._stats_manager = value

    def is_processing_required(self, timecode: FrameTimecode) -> bool:
        """Test if all calculations for a given frame are already done and stored in the
        associated StatsManager. Always True if there is no StatsManager.

        Returns:
            False if all calculations for timecode can be found in `stats_manager`, True otherwise.
            If True, calling `process_frame` with the given timecode (`frame_num`) requires
            the `frame_im` argument to be a valid frame.
        """
        return not self.metrics or self.stats_manager or (not self.stats_manager.metrics_exist(
            timecode.frame_num, self.metrics))

    @staticmethod
    @abstractmethod
    def stats_manager_required():
        """ Stats Manager Required: Prototype indicating if detector requires stats.

        Returns:
            bool: True if a StatsManager is required for the detector, False otherwise.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def metrics(self) -> List[str]:
        # type: () -> List[str]
        """ Metrics:  List of all metric names/keys used by the detector.

        Returns:
            List[str]: A list of strings of frame metric key names that will be used by
            the detector when a StatsManager is passed to process_frame.
        """
        raise NotImplementedError

    @abstractmethod
    def process_frame(self, timecode: FrameTimecode,
                      frame_img: Optional[numpy.ndarray]) -> List[DetectionEvent]:
        """Computes/stores metrics and detects any scene changes.

        `frame_img` may be None only if calling `is_processing_required` with the same `frame_num`
        returns False.

        Returns:
            List[int]: List of frame numbers of cuts to be added to the cutting list.
        """
        raise NotImplementedError

    @abstractmethod
    def post_process(self, start_time: FrameTimecode, end_time: FrameTimecode) -> List[DetectionEvent]:
        """Performs any processing after the last frame has been read.

        Returns:
            List[int]: List of frame numbers of cuts to be added to the cutting list.
        """
        raise NotImplementedError


class SparseSceneDetector(SceneDetector):
    """ Base class to inheret from when implementing a sparse scene detection algorithm.

    Unlike dense detectors, sparse detectors detect "events" and return a *pair* of frames,
    as opposed to just a single cut.

    An example of a SparseSceneDetector is the MotionDetector.
    """

    def process_frame(self, frame_num, frame_img):
        # type: (int, numpy.ndarray) -> List[Tuple[int, int]]
        """ Process Frame: Computes/stores metrics and detects any scene changes.

        Prototype method, no actual detection.

        Returns:
            List[Tuple[int,int]]: List of frame pairs representing individual scenes
            to be added to the output scene list directly.
        """
        return []

    def post_process(self, frame_num):
        # type: (int) -> List[Tuple[int, int]]
        """ Post Process: Performs any processing after the last frame has been read.

        Prototype method, no actual detection.

        Returns:
            List[Tuple[int,int]]: List of frame pairs representing individual scenes
            to be added to the output scene list directly.
        """
        return []
