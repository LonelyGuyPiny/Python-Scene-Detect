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
""" ``scenedetect.detectors.threshold_detector`` Module

This module implements the :py:class:`ThresholdDetector`, which uses a set intensity
level as a threshold, to detect cuts when the average frame intensity exceeds or falls
below this threshold.

This detector is available from the command-line interface by using the
`detect-threshold` command.
"""

from typing import List, Optional, Union

import numpy

from scenedetect.frame_timecode import FrameTimecode
from scenedetect.scene_detector import DetectionEvent, EventType, SceneDetector

##
## ThresholdDetector Helper Functions
##


def compute_frame_average(frame):
    """Computes the average pixel value/intensity for all pixels in a frame.

    The value is computed by adding up the 8-bit R, G, and B values for
    each pixel, and dividing by the number of pixels multiplied by 3.

    Returns:
        Floating point value representing average pixel intensity.
    """
    num_pixel_values = float(frame.shape[0] * frame.shape[1] * frame.shape[2])
    avg_pixel_value = numpy.sum(frame[:, :, :]) / num_pixel_values
    return avg_pixel_value


##
## ThresholdDetector Class Implementation
##


class ThresholdDetector(SceneDetector):
    """Detects fast cuts/slow fades in from and out to a given threshold level.

    Detects both fast cuts and slow fades so long as an appropriate threshold
    is chosen (especially taking into account the minimum grey/black level).

    Attributes:
        threshold:  8-bit intensity value that each pixel value (R, G, and B)
            must be <= to in order to trigger a fade in/out.

        TODO(v1.0): The following arguments are being moved to `SceneManager.transform_events`:
        min_scene_len:  FrameTimecode object or integer greater than 0 of the
            minimum length, in frames, of a scene (or subsequent scene cut).
        time_before: FrameTimecode representing max. amount to shift each IN event backwards
            in time by (default is None for no shift).
        time_after: FrameTimecode representing amount of time to shift each OUT event forwards
            in time by (default is None for no shift).
        cut_mode: If True, will emit `EventType.CUT` events instead of `EventType.IN`
            and `EventType.OUT` events.
    """

    THRESHOLD_VALUE_KEY = 'delta_rgb'

    def __init__(self,
                 threshold: int = 12,
                 min_scene_len: Union[int, FrameTimecode] = 15,
                 time_before: Optional[FrameTimecode] = None,
                 time_after: Optional[FrameTimecode] = None,
                 cut_mode: bool = False,
                 fade_bias: Optional[float] = 0.0,
                 add_cut_on_last_out=False):
        """Initializes threshold-based scene detector object."""

        super().__init__()
        # TODO(v1.0): Allow negative values for threshold to indicate the opposite (e.g. output only frames that
        # the threshold is below instead of above), e.g. if set to -10, will output an IN event
        # when the average frame value goes under 10, and will output an OUT when it goes back above.
        # TODO(v1.0): Make it a float.
        self._threshold = int(threshold)
        self._metric_keys = [ThresholdDetector.THRESHOLD_VALUE_KEY]
        self._last_frame_avg = None

    @staticmethod
    def stats_manager_required() -> bool:
        return False

    @property
    def metrics(self) -> List[str]:
        return self._metric_keys

    def process_frame(self, timecode: FrameTimecode,
                      frame_img: Optional[numpy.ndarray]) -> List[DetectionEvent]:
        # Calculate the current frame's average value.
        frame_num = timecode.frame_num
        if (self.stats_manager is not None) and (self.stats_manager.metrics_exist(
                frame_num, self._metric_keys)):
            curr_frame_avg = self.stats_manager.get_metrics(frame_num, self._metric_keys)[0]
        else:
            curr_frame_avg = compute_frame_average(frame_img)
            if self.stats_manager is not None:
                self.stats_manager.set_metrics(frame_num, {self._metric_keys[0]: curr_frame_avg})

        # Swap the average with the last frame's average for use on the next frame.
        last_frame_avg = self._last_frame_avg
        self._last_frame_avg = curr_frame_avg

        # Compare the current frame's average against the previous frame, if any.
        if last_frame_avg is not None:
            # Detect fade in.
            if last_frame_avg < self._threshold and curr_frame_avg >= self._threshold:
                return [DetectionEvent(kind=EventType.IN, time=timecode)]
            # Detect fade out.
            if last_frame_avg >= self._threshold and curr_frame_avg < self._threshold:
                return [DetectionEvent(kind=EventType.OUT, time=timecode)]

        return []  # No previous frames to compare against yet.

    def post_process(self) -> List[DetectionEvent]:
        return []
