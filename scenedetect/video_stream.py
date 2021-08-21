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

""" ``scenedetect.video_stream`` Module

This module contains the :py:class:`VideoStream` class, which provides a consistent
interface to reading videos which is library agnostic.  This allows PySceneDetect to
support multiple video backends.

"""

from abc import ABC, abstractmethod
from typing import Tuple, Union, Optional
import os.path

from numpy import ndarray

from scenedetect.platform import logger
from scenedetect.frame_timecode import FrameTimecode


class VideoStream(ABC):
    """ Interface which all video backends must implement. """
    def __init__(self):
        self._downscale: Optional[int] = None

    @property
    def downscale(self) -> int:
        """Factor to downscale each frame by. If 0, 1, or None, has no effect.
        If 2, effectively divides the video into 1/4 it's original resolution."""
        return self._downscale

    @downscale.setter
    def downscale(self, downscale_factor: Optional[int] = None):
        # TODO: If None, calculate based on frame size.
        if not isinstance(downscale_factor, int):
            logger.warning("Downscale factor will be truncated to integer!")
        self._downscale = downscale_factor

    @property
    @abstractmethod
    def path(self) -> str:
        """Returns video or device path."""
        raise NotImplementedError

    @property
    @abstractmethod
    def base_timecode(self) -> FrameTimecode:
        """Returns base FrameTimecode object to use as a timebase."""
        raise NotImplementedError

    @property
    @abstractmethod
    def position(self) -> FrameTimecode:
        """Returns current position within stream as FrameTimecode."""
        raise NotImplementedError

    @property
    @abstractmethod
    def duration(self) -> Optional[FrameTimecode]:
        """Returns duration of the stream as a FrameTimecode, or None if non terminating."""
        raise NotImplementedError

    @property
    @abstractmethod
    def frame_size(self) -> Tuple[int, int]:
        """Size of each video frame in pixels."""
        raise NotImplementedError

    @property
    @abstractmethod
    def aspect_ratio(self) -> Tuple[float, float]:
        """Returns display aspect ratio in form numerator/denominator."""
        raise NotImplementedError

    @abstractmethod
    def read(self, decode: bool=True, advance: bool=True) -> Optional[ndarray]:
        """ Returns next frame (or current if advance = False), or None if end of video.

        If decode = False, None will be returned, but will be slightly faster.

        If decode and advance are both False, equivalent to a no-op.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """ Close and re-open the VideoStream (equivalent to seeking back to beginning). """
        raise NotImplementedError

    @abstractmethod
    def seek(self, timecode: FrameTimecode):
        """ Seeks to the given timecode. Will be the next frame returned by read(). """
        raise NotImplementedError

    @property
    def frame_rate(self) -> float:
        """Get framerate in frames/sec."""
        return self.base_timecode.get_framerate()

    @property
    def frame_size_effective(self) -> Tuple[int, int]:
        """Get effective framesize taking into account downscale if set."""
        if self.downscale is None:
            return self.frame_size
        return (self.frame_size[0] / self.downscale, self.frame_size[1] / self.downscale)





