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
"""PySceneDetect scenedetect.scene_manager Tests

This file includes unit tests for the scenedetect.scene_manager.SceneManager class,
which applies SceneDetector algorithms on VideoStream backends.
"""

# Standard project pylint disables for unit tests using pytest.
# pylint: disable=protected-access, invalid-name, unused-argument, redefined-outer-name

import glob
import os
import os.path
from typing import Iterable, Tuple

import pytest

from scenedetect.backends.opencv import VideoStreamCv2
from scenedetect.detectors import ContentDetector
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.scene_detector import DetectionEvent, EventType
from scenedetect.scene_manager import SceneManager, save_images

# TODO(v1.0): Add hard-coded results for test files.

# TODO: Add tests that combine detector types.


def _set_events(scene_manager: SceneManager, events: Iterable[Tuple[int, EventType]],
                base_timecode: FrameTimecode) -> None:
    scene_manager.clear()
    scene_manager.add_events([
        DetectionEvent(kind=event, time=base_timecode + frame_num) for (frame_num, event) in events
    ])


def _create_sm_from_events(events: Iterable[Tuple[int, EventType]],
                           start_frame=0,
                           end_frame=0,
                           fps=10.0) -> Tuple[SceneManager, FrameTimecode]:
    """Helper function that creates a SceneManager populated with the given events, where `events`
    is a list of frame number and event type pairs. The FrameTimecodes are created with `fps`,"""
    sm = SceneManager()
    sm.add_detector(ContentDetector())
    bt = FrameTimecode(0, fps)
    _set_events(scene_manager=sm, events=events, base_timecode=bt)
    sm._start_pos = bt + start_frame
    sm._last_pos = bt + end_frame
    return sm, bt


def scene_with_gaps() -> Tuple[SceneManager, FrameTimecode]:
    """Start: 5, End: 99, Events: (5, IN), (20, OUT), (30, IN), (40, OUT), (60, IN), (80, OUT)"""
    IN, OUT = EventType.IN, EventType.OUT
    EVENTS = [(5, IN), (20, OUT), (30, IN), (40, OUT), (60, IN), (80, OUT)]
    return _create_sm_from_events(events=EVENTS, start_frame=5, end_frame=99)


def scene_with_gaps_v2() -> Tuple[SceneManager, FrameTimecode]:
    """Start: 5, End: 99, Events: (5, IN), (20, OUT), (40, IN), (50, OUT), (60, IN), (80, OUT)"""
    # Same as scene_with_gaps but the middle scene is shifted more towards the last scene.
    IN, OUT = EventType.IN, EventType.OUT
    EVENTS = [(5, IN), (20, OUT), (40, IN), (50, OUT), (60, IN), (80, OUT)]
    return _create_sm_from_events(events=EVENTS, start_frame=5, end_frame=99)


def scene_with_no_gaps() -> Tuple[SceneManager, FrameTimecode]:
    """Start: 5, End: 59, Events: (10, IN), (20, CUT), (40, OUT)

                   IN     CUT    OUT
           START                       END
Frame   0    5     10     20     40      59
-----------------------------------------------

    """
    # Test case has IN at frame 10 (t=1.0s), CUT at 20 (t=2.0s), OUT at 40 (t=4.0s), and has a
    # total of 59 frames (thus the video length is 6.0s).
    EVENTS = [(10, EventType.IN), (20, EventType.CUT), (40, EventType.OUT)]
    return _create_sm_from_events(events=EVENTS, start_frame=5, end_frame=59)


def scene_with_no_gaps_cuts_only() -> Tuple[SceneManager, FrameTimecode]:
    """Start: 5, End: 59, Events: (10, CUT), (20, CUT), (40, CUT)"""
    # Test case has CUT at frame 10 (t=1.0s), CUT at 20 (t=2.0s), CUT at 40 (t=4.0s), and has a
    # total of 59 frames (thus the video length is 6.0s and there are 4 scenes).
    EVENTS = [(10, EventType.CUT), (20, EventType.CUT), (40, EventType.CUT)]
    return _create_sm_from_events(events=EVENTS, start_frame=5, end_frame=59)


def test_empty():

    sm, _ = _create_sm_from_events(events=[], end_frame=59)

    assert len(sm.get_cut_list()) == 0
    assert len(sm.get_scene_list()) == 0

    sm, _ = _create_sm_from_events(events=[(20, EventType.CUT), (40, EventType.OUT)], end_frame=59)
    assert len(sm.get_cut_list()) == 1
    assert len(sm.get_scene_list()) == 0


def test_get_cut_list():
    """ Test get_cut_list with only CUT events. """
    # Test case has CUT at frame 10 (t=1.0s), CUT at 20 (t=2.0s), CUT at 40 (t=4.0s), and has a
    # total of 59 frames (thus the video length is 6.0s and there are 4 scenes).
    EVENTS = [(10, EventType.CUT), (20, EventType.CUT), (40, EventType.CUT)]
    sm, bt = _create_sm_from_events(events=EVENTS, end_frame=59)

    assert sm.get_cut_list(min_time_between_cuts=0) == [bt + 10, bt + 20, bt + 40]
    assert sm.get_cut_list(min_time_between_cuts=10) == [bt + 10, bt + 20, bt + 40]
    assert sm.get_cut_list(min_time_between_cuts=11) == [bt + 10, bt + 40]
    assert sm.get_cut_list(min_time_between_cuts=30) == [bt + 10, bt + 40]
    assert sm.get_cut_list(min_time_between_cuts=31) == [bt + 10]


def test_scene_list_cuts_only():
    """ Test get_scene_list with only CUT events. """
    sm, bt = scene_with_no_gaps_cuts_only()
    assert sm.get_scene_list() == [(bt + 5, bt + 10), (bt + 10, bt + 20), (bt + 20, bt + 40),
                                   (bt + 40, bt + 60)]


def test_get_scene_list():
    """Test basic functionality of `get_scene_list` with no/default transformations."""
    sm, bt = scene_with_no_gaps()
    assert sm.get_scene_list() == [(bt + 10, bt + 20), (bt + 20, bt + 40)]
    sm, bt = scene_with_gaps()
    assert sm.get_scene_list() == [(bt + 5, bt + 20), (bt + 30, bt + 40), (bt + 60, bt + 80)]


def test_get_scene_list_always_include_end():
    """Test the `always_include_end` option in `get_scene_list`."""
    sm, bt = scene_with_no_gaps()
    assert sm.get_scene_list(always_include_end=True) == [(bt + 10, bt + 20), (bt + 20, bt + 40),
                                                          (bt + 40, bt + 60)]
    # Ensure a CUT outside of the last OUT event does not trigger a new scene.
    sm.events[bt + 45] = [DetectionEvent(kind=EventType.CUT, time=bt + 45)]
    assert sm.get_scene_list(always_include_end=True) == [(bt + 10, bt + 20), (bt + 20, bt + 40),
                                                          (bt + 40, bt + 60)]
    # Ensure an OUT outside of the last OUT event does not modify the last scene.
    sm.events[bt + 50] = [DetectionEvent(kind=EventType.OUT, time=bt + 50)]
    assert sm.get_scene_list(always_include_end=True) == [(bt + 10, bt + 20), (bt + 20, bt + 40),
                                                          (bt + 40, bt + 60)]
    # Lastly, check that the result is consistent with another set.
    sm, bt = scene_with_gaps()
    assert sm.get_scene_list(always_include_end=True) == [(bt + 5, bt + 20), (bt + 30, bt + 40),
                                                          (bt + 60, bt + 80), (bt + 80, bt + 100)]


def test_get_scene_list_min_scene_len():
    """Test the `min_scene_len` option in `get_scene_list`."""
    sm, bt = scene_with_no_gaps()
    assert sm.get_scene_list(min_scene_len=10) == [(bt + 10, bt + 20), (bt + 20, bt + 40)]

    assert sm.get_scene_list(min_scene_len=10) == [(bt + 10, bt + 20), (bt + 20, bt + 40)]

    assert sm.get_scene_list(min_scene_len=20) == [(bt + 10, bt + 40)]


def test_get_scene_list_merge_len():
    """Test the `merge_len` option in `get_scene_list`."""
    sm, bt = scene_with_no_gaps()
    # If merge_len is None, we should drop all scenes shorter than min_scene_len.
    assert sm.get_scene_list(min_scene_len=20, merge_len=None) == [(bt + 20, bt + 40)]
    # If merge_len is >= 0, then they should be merged instead since they are directly adjacent.
    assert sm.get_scene_list(min_scene_len=20, merge_len=0) == [(bt + 10, bt + 40)]
    assert sm.get_scene_list(min_scene_len=20, merge_len=200) == [(bt + 10, bt + 40)]

    sm, bt = scene_with_gaps()
    # Increase merge_len until the first two scenes get merged.
    assert sm.get_scene_list(min_scene_len=20, merge_len=None) == [(bt + 60, bt + 80)]
    assert sm.get_scene_list(min_scene_len=20, merge_len=9) == [(bt + 60, bt + 80)]
    assert sm.get_scene_list(
        min_scene_len=20, merge_len=10) == [(bt + 5, bt + 40), (bt + 60, bt + 80)]
    # Test that all get merged properly once min_scene_len is increased.
    assert sm.get_scene_list(min_scene_len=30, merge_len=20) == [(bt + 5, bt + 80)]
    # Test that the predecessor is preferred over the successor for merging.
    assert sm.get_scene_list(
        min_scene_len=15, merge_len=20) == [(bt + 5, bt + 40), (bt + 60, bt + 80)]

    # Test that the successor can still be merged to if it is the only candidate.
    sm, bt = scene_with_gaps_v2()
    assert sm.get_scene_list(
        min_scene_len=15, merge_len=10) == [(bt + 5, bt + 20), (bt + 40, bt + 80)]


def test_get_scene_list_shift_start_offset():
    """Test the `shift_scene_start`/`shift_scene_end`/`overlap_bias` options in `get_scene_list`."""

    sm, bt = scene_with_no_gaps()
    # Test shift_scene_start with overlap_bias=None (i.e. merge), which is the default
    assert sm.get_scene_list(shift_scene_start=-5) == [(bt + 5, bt + 40)]
    assert sm.get_scene_list(shift_scene_start=-5, overlap_bias=None) == [(bt + 5, bt + 40)]
    assert sm.get_scene_list(
        always_include_end=True, shift_scene_start=-5, overlap_bias=None) == [(bt + 5, bt + 60)]
    # Test shift_scene_start with overlap_bias=0.0
    assert sm.get_scene_list(
        shift_scene_start=-5, overlap_bias=0.0) == [(bt + 5, bt + 17), (bt + 17, bt + 40)]
    assert sm.get_scene_list(
        always_include_end=True, shift_scene_start=-5, overlap_bias=0.0) == [(bt + 5, bt + 17),
                                                                             (bt + 17, bt + 37),
                                                                             (bt + 37, bt + 60)]
    # Test shift_scene_start with overlap_bias=1.0
    assert sm.get_scene_list(
        shift_scene_start=-5, overlap_bias=1.0) == [(bt + 5, bt + 20), (bt + 20, bt + 40)]
    assert sm.get_scene_list(
        always_include_end=True, shift_scene_start=-5, overlap_bias=1.0) == [(bt + 5, bt + 20),
                                                                             (bt + 20, bt + 40),
                                                                             (bt + 40, bt + 60)]
    # Test shift_scene_start with overlap_bias=-1.0
    assert sm.get_scene_list(
        shift_scene_start=-5, overlap_bias=-1.0) == [(bt + 5, bt + 15), (bt + 15, bt + 40)]
    assert sm.get_scene_list(
        always_include_end=True, shift_scene_start=-5, overlap_bias=-1.0) == [(bt + 5, bt + 15),
                                                                              (bt + 15, bt + 35),
                                                                              (bt + 35, bt + 60)]

    # Test shifting more than the entire first two scenes behind time 0
    assert sm.get_scene_list(shift_scene_start=-25, overlap_bias=None) == [(bt, bt + 40)]
    assert sm.get_scene_list(
        shift_scene_start=-45, overlap_bias=None, always_include_end=True) == [(bt, bt + 60)]
    assert sm.get_scene_list(shift_scene_start=-25, overlap_bias=0.0) == [(bt, bt + 40)]
    assert sm.get_scene_list(shift_scene_start=-25, overlap_bias=1.0) == [(bt, bt + 40)]
    assert sm.get_scene_list(shift_scene_start=-25, overlap_bias=-1.0) == [(bt, bt + 40)]

    # Test shifting more than the entire last two scenes beyond the end of the video
    assert sm.get_scene_list(shift_scene_end=45, overlap_bias=None) == [(bt + 10, bt + 60)]
    assert sm.get_scene_list(shift_scene_end=45, overlap_bias=0.0) == [(bt + 10, bt + 60)]
    assert sm.get_scene_list(shift_scene_end=45, overlap_bias=1.0) == [(bt + 10, bt + 60)]
    assert sm.get_scene_list(shift_scene_end=45, overlap_bias=-1.0) == [(bt + 10, bt + 60)]



    # Basic test using a video containing gaps instead of a contiguous segment.
    sm, bt = scene_with_gaps()
    # TODO(v1.0)


def test_save_images(test_video_file):
    """ Test scenedetect.scene_manager.save_images function.  """
    video = VideoStreamCv2(test_video_file)
    sm = SceneManager()
    sm.add_detector(ContentDetector())

    image_name_glob = 'scenedetect.tempfile.*.jpg'
    image_name_template = 'scenedetect.tempfile.$SCENE_NUMBER.$IMAGE_NUMBER'

    try:
        video_fps = video.frame_rate
        start_time = FrameTimecode('00:00:05', video_fps)
        end_time = FrameTimecode('00:00:15', video_fps)

        video.seek(start_time)
        sm.auto_downscale = True

        sm.detect_scenes(video=video, end_time=end_time)

        scene_list = sm.get_scene_list()
        assert scene_list

        image_filenames = save_images(
            scene_list=scene_list,
            video=video,
            num_images=3,
            image_extension='jpg',
            image_name_template=image_name_template)

        # Ensure images got created, and the proper number got created.
        total_images = 0
        for scene_number in image_filenames:
            for path in image_filenames[scene_number]:
                assert os.path.exists(path)
                total_images += 1

        assert total_images == len(glob.glob(image_name_glob))

    finally:
        for path in glob.glob(image_name_glob):
            os.remove(path)


class FakeCallback(object):
    """ Fake callback used for testing purposes only. Currently just stores
    the number of times the callback was invoked."""

    def __init__(self):
        self.num_invoked: int = 0

    def get_callback_lambda(self):
        """Returns a callback which consumes a frame image and timecode. The `num_invoked` property
        is incremented each time the callback is invoked."""
        return lambda image, frame_num: self._callback(image, frame_num)

    def get_callback_func(self):
        """Returns a callback which consumes a frame image and timecode. The `num_invoked` property
        is incremented each time the callback is invoked."""

        def callback(image, frame_num):
            nonlocal self
            self._callback(image, frame_num)

        return callback

    def _callback(self, image, frame_num):
        self.num_invoked += 1


def test_detect_scenes_callback(test_video_file):
    """ Test SceneManager detect_scenes method with a callback function.

    Note that the API signature of the callback will undergo breaking changes in v0.6.
    """
    video = VideoStreamCv2(test_video_file)
    sm = SceneManager()
    sm.add_detector(ContentDetector())

    fake_callback = FakeCallback()

    video_fps = video.frame_rate
    start_time = FrameTimecode('00:00:05', video_fps)
    end_time = FrameTimecode('00:00:15', video_fps)
    video.seek(start_time)
    sm.auto_downscale = True

    _ = sm.detect_scenes(
        video=video, end_time=end_time, callback=fake_callback.get_callback_lambda())
    assert fake_callback.num_invoked == (len(sm.get_scene_list()) - 1)

    sm.clear()
    fake_callback.num_invoked = 0
    video.seek(start_time)

    _ = sm.detect_scenes(video=video, end_time=end_time, callback=fake_callback.get_callback_func())
    assert fake_callback.num_invoked == (len(sm.get_scene_list()) - 1)
