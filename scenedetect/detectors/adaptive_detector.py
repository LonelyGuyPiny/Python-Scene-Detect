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
""" Module: ``scenedetect.detectors.adaptive_detector``

This module implements the :py:class:`AdaptiveDetector`, which compares the
difference in content between adjacent frames similar to `ContentDetector` except the
threshold isn't fixed, but is a rolling average of adjacent frame changes. This can
help mitigate false detections in situations such as fast camera motions.

This detector is available from the command-line interface by using the
`detect-adaptive` command.
"""

from typing import List, Optional

import numpy

from scenedetect.detectors import ContentDetector
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.scene_detector import DetectionEvent, EventType


# TODO(v1.0): This class should own and create it's own ContentDetector rather
# than using inheritance.
class AdaptiveDetector(ContentDetector):
    """Detects cuts using HSV changes similar to ContentDetector, but with a
    rolling average that can help mitigate false detections in situations such
    as camera moves.
    """

    ADAPTIVE_RATIO_KEY_TEMPLATE = "adaptive_ratio{luma_only} (w={window_width})"

    def __init__(self,
                 adaptive_threshold=3.0,
                 luma_only=False,
                 min_scene_len=15,
                 min_delta_hsv=15.0,
                 window_width=2):
        super().__init__()
        # Minimum length of any given scene, in frames (int) or FrameTimecode
        self.min_scene_len = min_scene_len
        self.adaptive_threshold = adaptive_threshold
        self.min_delta_hsv = min_delta_hsv
        self.window_width = window_width
        self._luma_only = luma_only
        self._adaptive_ratio_key = AdaptiveDetector.ADAPTIVE_RATIO_KEY_TEMPLATE.format(
            window_width=window_width, luma_only='' if not luma_only else '_lum')

        self._start_time = None
        self._end_time = None

    @staticmethod
    def stats_manager_required() -> bool:
        # type: () -> bool
        """ Overload to indicate that this detector requires a StatsManager.

        Returns:
            True as AdaptiveDetector requires stats.
        """
        return True

    @property
    def metrics(self) -> List[str]:
        # type: () -> List[str]
        """ Combines base ContentDetector metric keys with the AdaptiveDetector one. """

        return super().metrics + [self._adaptive_ratio_key]

    def process_frame(self, timecode: FrameTimecode,
                      frame_img: Optional[numpy.ndarray]) -> List[DetectionEvent]:
        """ Similar to ThresholdDetector, but using the HSV colour space DIFFERENCE instead
        of single-frame RGB/grayscale intensity (thus cannot detect slow fades with this method).

        Arguments:
            frame_num (int): Frame number of frame that is being passed.

            frame_img (Optional[int]): Decoded frame image (numpy.ndarray) to perform scene
                detection on, or None if `is_processing_required()` returns False.

        Returns:
            Empty list
        """

        # Call the process_frame function of ContentDetector but ignore any
        # returned cuts
        if self.is_processing_required(timecode):
            super().process_frame(timecode, frame_img)

        if self._start_time is None:
            self._start_time = timecode
        self._end_time = timecode

        return []

    def get_content_val(self, frame_num: int):
        """
        Returns the average content change for a frame.
        """
        metric_key = (
            ContentDetector.FRAME_SCORE_KEY if not self._luma_only else ContentDetector.DELTA_V_KEY)
        return self.stats_manager.get_metrics(frame_num, [metric_key])[0]

    def post_process(self) -> List[DetectionEvent]:
        """
        After an initial run through the video to detect content change
        between each frame, we try to identify fast cuts as short peaks in the
        `content_val` value. If a single frame has a high `content-val` while
        the frames around it are low, we can be sure it's fast cut. If several
        frames in a row have high `content-val`, it probably isn't a cut -- it
        could be fast camera movement or a change in lighting that lasts for
        more than a single frame.
        """
        if self._start_time is None:
            return []

        cut_list = []
        adaptive_threshold = self.adaptive_threshold
        window_width = self.window_width
        last_cut = None
        start_frame = self._start_time.frame_num
        end_frame = self._end_time.frame_num
        base_timecode = FrameTimecode(0, self._start_time)


        assert self.stats_manager is not None

        if self.stats_manager is not None:
            # Loop through the stats, building the adaptive_ratio metric
            for frame_num in range(start_frame + window_width + 1, end_frame - window_width):
                # If the content-val of the frame is more than
                # adaptive_threshold times the mean content_val of the
                # frames around it, then we mark it as a cut.
                denominator = 0
                for offset in range(-window_width, window_width + 1):
                    if offset == 0:
                        continue
                    else:
                        denominator += self.get_content_val(frame_num + offset)

                denominator = denominator / (2.0 * window_width)
                denominator_is_zero = abs(denominator) < 0.00001

                if not denominator_is_zero:
                    adaptive_ratio = self.get_content_val(frame_num) / denominator
                elif denominator_is_zero and self.get_content_val(frame_num) >= self.min_delta_hsv:
                    # if we would have divided by zero, set adaptive_ratio to the max (255.0)
                    adaptive_ratio = 255.0
                else:
                    # avoid dividing by zero by setting adaptive_ratio to zero if content_val
                    # is still very low
                    adaptive_ratio = 0.0

                self.stats_manager.set_metrics(frame_num,
                                               {self._adaptive_ratio_key: adaptive_ratio})

            # Loop through the frames again now that adaptive_ratio has been calculated to detect
            # cuts using adaptive_ratio
            for frame_num in range(start_frame + window_width + 1, end_frame - window_width):
                # Check to see if adaptive_ratio exceeds the adaptive_threshold as well as there
                # being a large enough content_val to trigger a cut
                if (self.stats_manager.get_metrics(
                        frame_num, [self._adaptive_ratio_key])[0] >= adaptive_threshold
                        and self.get_content_val(frame_num) >= self.min_delta_hsv):
                    cut_time = base_timecode + frame_num
                    if last_cut is None or (frame_num - last_cut) >= self.min_scene_len:
                        cut_list.append(DetectionEvent(kind=EventType.CUT, time=cut_time))
                        last_cut = frame_num

            return cut_list

        # Stats manager must be used for this detector
        return None
