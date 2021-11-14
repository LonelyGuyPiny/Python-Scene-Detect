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
""" ``scenedetect.detectors.content_detector`` Module

This module implements the :py:class:`ContentDetector`, which compares the
difference in content between adjacent frames against a set threshold/score,
which if exceeded, triggers a scene cut.

This detector is available from the command-line interface by using the
`detect-content` command.
"""

from typing import List, Optional, Union

import numpy
import cv2

from scenedetect.frame_timecode import FrameTimecode
from scenedetect.scene_detector import DetectionEvent, EventType, SceneDetector


class ContentDetector(SceneDetector):
    """Detects fast cuts using changes in colour and intensity between frames.

    Since the difference between frames is used, unlike the ThresholdDetector,
    only fast cuts are detected with this method.  To detect slow fades between
    content scenes still using HSV information, use the DissolveDetector.
    """

    FRAME_SCORE_KEY = 'content_val'
    DELTA_H_KEY, DELTA_S_KEY, DELTA_V_KEY = ('delta_hue', 'delta_sat', 'delta_lum')
    METRIC_KEYS = [FRAME_SCORE_KEY, DELTA_H_KEY, DELTA_S_KEY, DELTA_V_KEY]

    def __init__(self, threshold: float = 30.0, min_scene_len: int = 15, luma_only: bool = False):
        super().__init__()
        # Threshold representing detector sensitivity (lower = more sensitive).
        self._threshold: float = threshold
        # Minimum length of any given scene, in frames (int) or as a FrameTimecode.
        # TODO(v1.0): Consider moving this to SceneManager.transform_events.
        self._min_scene_len: Union[int, FrameTimecode] = min_scene_len
        # Only considers lightness information. Set to True if input video is greyscale.
        self._luma_only: bool = luma_only
        # Either the last frame that was processed, or True if the last frame has statistics for it
        # already calculated. Set to None until the first frame is processed.
        self._last_hsv: Optional[Union[bool, numpy.ndarray]] = None
        # Frame index of the last time a cut was detected.
        self._last_scene_cut: Optional[FrameTimecode] = None

    @staticmethod
    def stats_manager_required() -> bool:
        return False

    @property
    def metrics(self) -> List[str]:
        return ContentDetector.METRIC_KEYS

    def calculate_frame_score(self, frame_num, curr_hsv, last_hsv):
        # type: (int, List[numpy.ndarray], List[numpy.ndarray]) -> float
        curr_hsv = [x.astype(numpy.int32) for x in curr_hsv]
        last_hsv = [x.astype(numpy.int32) for x in last_hsv]
        delta_hsv = [0, 0, 0, 0]
        for i in range(3):
            num_pixels = curr_hsv[i].shape[0] * curr_hsv[i].shape[1]
            delta_hsv[i] = numpy.sum(numpy.abs(curr_hsv[i] - last_hsv[i])) / float(num_pixels)

        delta_hsv[3] = sum(delta_hsv[0:3]) / 3.0
        delta_h, delta_s, delta_v, delta_content = delta_hsv

        if self.stats_manager is not None:
            self.stats_manager.set_metrics(
                frame_num, {
                    self.FRAME_SCORE_KEY: delta_content,
                    self.DELTA_H_KEY: delta_h,
                    self.DELTA_S_KEY: delta_s,
                    self.DELTA_V_KEY: delta_v
                })
        return delta_content if not self._luma_only else delta_v

    def _get_cached_score(self, frame_num: int) -> Optional[float]:
        metric_key = (
            ContentDetector.DELTA_V_KEY if self._luma_only else ContentDetector.FRAME_SCORE_KEY)
        if (self.stats_manager is not None
                and self.stats_manager.metrics_exist(frame_num, [metric_key])):
            return self.stats_manager.get_metrics(frame_num, [metric_key])[0]
        return None

    def process_frame(self, timecode: FrameTimecode,
                      frame_img: Optional[numpy.ndarray]) -> List[DetectionEvent]:
        """ Similar to ThresholdDetector, but using the HSV colour space DIFFERENCE instead
        of single-frame RGB/grayscale intensity (thus cannot detect slow fades with this method).

        Arguments:
            timecode: Presentation timestamp of the frame being processed.

            frame_img: Decoded frame image (numpy.ndarray) to perform scene
                detection on. Can be None *only* if the self.is_processing_required() method
                (inhereted from the base SceneDetector class) returns True.

        Returns:
            List[int]: List of frames where scene cuts have been detected. There may be 0
            or more frames in the list, and not necessarily the same as frame_num.
        """

        cut_list = []
        frame_num = timecode.frame_num

        # Initialize last scene cut point at the beginning of the frames of interest.
        if self._last_scene_cut is None or self._last_scene_cut > frame_num:
            self._last_scene_cut = frame_num

        # Can only begin processing once we have at least one frame:
        if self._last_hsv is None:
            # If we have the next frame computed, don't copy the current frame since we won't use
            # it on the next call anyways.
            if (self.stats_manager is not None
                    and self.stats_manager.metrics_exist(frame_num + 1, self.metrics)):
                # Since we can just lookup the value on the next call to process_frame, replace the
                # last frame with a sentinel value instead of copying & converting the colorspace.
                self._last_hsv = True
            else:
                # We need the frame on the next iteration to calculate the delta.
                self._last_hsv = cv2.split(cv2.cvtColor(frame_img.copy(), cv2.COLOR_BGR2HSV))
        else:
            # Since process_frame was called previously, we can calculate a score from the last
            # frame (or just get it from the StatsManager if it was previously calculated).
            frame_score: Optional[float] = self._get_cached_score(frame_num)
            if frame_score is None:
                curr_hsv = cv2.split(cv2.cvtColor(frame_img, cv2.COLOR_BGR2HSV))
                frame_score = self.calculate_frame_score(frame_num, curr_hsv, self._last_hsv)
                self._last_hsv = curr_hsv

            # We consider any frame over the threshold a new scene, but only if
            # the minimum scene length has been reached (otherwise it is ignored).
            if frame_score >= self._threshold and (
                (frame_num - self._last_scene_cut) >= self._min_scene_len):
                cut_list.append(
                    DetectionEvent(
                        kind=EventType.CUT, time=timecode, context={'confidence': 'TODO(v1.0)'}))
                self._last_scene_cut = frame_num

        # If we have the next frame computed, don't copy the current frame
        # into last_frame since we won't use it on the next call anyways.
        if (self.stats_manager is not None
                and self.stats_manager.metrics_exist(frame_num + 1, self.metrics)):
            self._last_hsv = True
        else:
            self._last_hsv = cv2.split(cv2.cvtColor(frame_img.copy(), cv2.COLOR_BGR2HSV))

        return cut_list

    def post_process(self, start_time: FrameTimecode,
                     end_time: FrameTimecode) -> List[DetectionEvent]:
        """Performs any processing after the last frame has been read.

        TODO: Based on the parameters passed to the ContentDetector constructor,
            ensure that the last scene meets the minimum length requirement,
            otherwise it should be merged with the previous scene.
        """
        return []
