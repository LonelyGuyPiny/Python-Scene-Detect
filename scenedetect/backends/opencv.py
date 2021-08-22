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
""" ``scenedetect.backends.opencv`` Module

This module contains the :py:class:`VideoStreamCv2` class, which provides an OpenCV
based video decoder (based on cv2.VideoCapture).
"""

import math
from typing import Tuple, Union, Optional
import os.path

import cv2
from numpy import ndarray

from scenedetect.frame_timecode import FrameTimecode, MINIMUM_FRAMES_PER_SECOND_FLOAT
from scenedetect.platform import get_aspect_ratio, logger
from scenedetect.video_stream import VideoStream, SeekError, VideoOpenFailure


class VideoStreamCv2(VideoStream):
    """ OpenCV VideoCapture backend. """

    def __init__(self, path_or_device: Union[str, int], override_framerate: Optional[float] = None):
        """Opens a new OpenCV backend."""
        super().__init__()
        self._path_or_device = path_or_device
        self._is_device = isinstance(self._path_or_device, int)
        self._cap, self._frame_rate = self._open_capture(override_framerate)

    #
    # VideoStream Methods/Properties
    #

    @property
    def frame_rate(self) -> float:
        """Get framerate in frames/sec."""
        return self._frame_rate


    @property
    def path(self) -> str:
        """Returns video or device path."""
        if self._is_device:
            return "Device %d" % self._path_or_device
        return self._path_or_device

    @property
    def is_seekable(self) -> bool:
        """True if seek() is allowed, False otherwise.

        Always False if opening a device/webcam."""
        return not self._is_device

    @property
    def frame_size(self) -> Tuple[int, int]:
        """Size of each video frame in pixels as a tuple of (width, height)."""
        return [(math.trunc(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                 math.trunc(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))]

    @property
    def duration(self) -> Optional[FrameTimecode]:
        """Returns duration of the stream as a FrameTimecode, or None if non terminating."""
        if self._is_device:
            return None
        return self.base_timecode + math.trunc(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def aspect_ratio(self) -> float:
        """Returns display/pixel aspect ratio as a float (1.0 represents square pixels)."""
        return get_aspect_ratio(self._cap)

    @property
    def position(self) -> FrameTimecode:
        """Returns current position within stream as FrameTimecode."""
        # TODO
        raise NotImplementedError

    def seek(self, target: Union[FrameTimecode, float, int]):
        """Seeks to the given timecode. Will be the next frame returned by read().

        May not be supported on all backends/types of videos (e.g. cameras).

        Arguments:
            target: Target position in video stream to seek to. Interpreted based on type.
              If FrameTimecode, backend can seek using any representation (preferably native when
              VFR support is added).
              If float, interpreted as time in seconds.
              If int, interpreted as frame number.
        Raises:
            SeekError if an unrecoverable error occurs while seeking, or seeking is not
            supported (either by the backend entirely, or if the input is a stream).
        """
        # TODO
        raise NotImplementedError


    def reset(self):
        """ Close and re-open the VideoStream (equivalent to seeking back to beginning). """
        self._cap, self._frame_rate = self._open_capture(self._frame_rate)


    def read(self, decode: bool = True, advance: bool = True) -> Optional[ndarray]:
        """ Returns next frame (or current if advance = False), or None if end of video.

        If decode = False, None will be returned, but will be slightly faster.

        If decode and advance are both False, equivalent to a no-op.
        """
        if advance:
            self._cap.grab()
        if decode:
            _, frame = self._cap.retrieve()
            return frame
        return None



    #
    # Private Methods
    #

    def _open_capture(self, override_framerate: Optional[float] = None) -> cv2.VideoCapture:
        if self._is_device and self._path_or_device < 0:
            raise ValueError("Invalid/negative device ID specified.")
        # Check if files exist if passed video file is not an image sequence
        # (checked with presence of % in filename) or not a URL (://).
        if not self._is_device and not ('%' in self._path_or_device
                                        or '://' in self._path_or_device):
            if not os.path.exists(self._path_or_device):
                raise IOError("Video file not found.")

        cap = cv2.VideoCapture(self._path_or_device)
        if not cap.isOpened():
            raise VideoOpenFailure("isOpened() returned False when opening OpenCV VideoCapture!")

        # Display a warning if the video codec type seems unsupported (#86).
        if int(abs(cap.get(cv2.CAP_PROP_FOURCC))) == 0:
            logger.error(
                "Video codec detection failed, output may be incorrect.\nThis could be caused"
                " by using an outdated version of OpenCV, or using codecs that currently are"
                " not well supported (e.g. VP9).\n"
                "As a workaround, consider re-encoding the source material before processing.\n"
                "For details, see https://github.com/Breakthrough/PySceneDetect/issues/86")

        # Ensure the framerate is correct to avoid potential divide by zero errors. This can be
        # addressed in the PyAV backend if required since it supports integer timebases.
        if override_framerate is not None:
            return cap, override_framerate

        framerate = cap.get(cv2.CAP_PROP_FPS)
        if framerate < MINIMUM_FRAMES_PER_SECOND_FLOAT:
            raise VideoOpenFailure(
                "Unable to obtain video framerate! Check the video file, or set override_framerate"
                " to assume a given framerate.")
        return cap, framerate
