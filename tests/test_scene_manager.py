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
# pylint: disable=protected-access, invalid-name, unused-argument

# Standard Library Imports
import glob
import os
import os.path
from typing import Iterable, Tuple

# PySceneDetect Library Imports
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
    scene_manager.events.update({
        base_timecode + frame_num: [DetectionEvent(kind=event, time=base_timecode + frame_num)]
        for (frame_num, event) in events
    })


def test_get_cut_list():
    """ Test get_cut_list with only CUT events. """
    sm = SceneManager()
    sm.add_detector(ContentDetector())
    fps = 10.0
    bt = FrameTimecode(0, fps)
    # Test case has CUT at frame 10 (t=1.0s), CUT at 20 (t=2.0s), CUT at 40 (t=4.0s), and has a
    # total of 59 frames (thus the video length is 6.0s and there are 4 scenes).
    EVENTS = [(10, EventType.CUT), (20, EventType.CUT), (40, EventType.CUT)]
    sm._start_pos = bt
    _set_events(scene_manager=sm, events=EVENTS, base_timecode=bt)
    sm._start_pos = bt
    sm._last_pos = bt + 59

    assert sm.get_cut_list(min_time_between_cuts=0) == [bt + 10, bt + 20, bt + 40]
    assert sm.get_cut_list(min_time_between_cuts=10) == [bt + 10, bt + 20, bt + 40]
    assert sm.get_cut_list(min_time_between_cuts=11) == [bt + 10, bt + 40]
    assert sm.get_cut_list(min_time_between_cuts=30) == [bt + 10, bt + 40]
    assert sm.get_cut_list(min_time_between_cuts=31) == [bt + 10]


def test_scene_list_cuts_only():
    """ Test get_scene_list with only CUT events. """
    sm = SceneManager()
    sm.add_detector(ContentDetector())
    fps = 10.0
    bt = FrameTimecode(0, fps)
    # Test case has CUT at frame 10 (t=1.0s), CUT at 20 (t=2.0s), CUT at 40 (t=4.0s), and has a
    # total of 59 frames (thus the video length is 6.0s and there are 4 scenes).
    EVENTS = [(10, EventType.CUT), (20, EventType.CUT), (40, EventType.CUT)]
    _set_events(scene_manager=sm, events=EVENTS, base_timecode=bt)
    sm._start_pos = bt + 5 # First scene should start at 0.5s.
    sm._last_pos = bt + 59 # Last scene should end at 6.0s.

    assert sm.get_scene_list() == [(bt + 5, bt + 10), (bt + 10, bt + 20), (bt + 20, bt + 40),
                                   (bt + 40, bt + 60)]


def test_scene_list_no_gaps():
    """ Test transformations that get_scene_list can apply with a small two scene test case with
    no gaps between scenes. """
    sm = SceneManager()
    sm.add_detector(ContentDetector())
    fps = 10.0
    bt = FrameTimecode(0, fps)
    # Test case has IN at frame 10 (t=1.0s), CUT at 20 (t=2.0s), OUT at 40 (t=4.0s), and has a
    # total of 59 frames (thus the video length is 6.0s).
    EVENTS = [(10, EventType.IN), (20, EventType.CUT), (40, EventType.OUT)]
    _set_events(scene_manager=sm, events=EVENTS, base_timecode=bt)
    sm._start_pos = bt + 5 # Start position is before first IN event, should not affect results.
    sm._last_pos = bt + 59 # Last scene if created should end at 6.0s.

    assert sm.get_scene_list() == [(bt + 10, bt + 20), (bt + 20, bt + 40)]

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


def test_scene_list_with_gaps():
    """ Test transformations that get_scene_list can apply with a small three scene test case with
    gaps between scenes. """
    sm = SceneManager()
    sm.add_detector(ContentDetector())
    fps = 10.0
    bt = FrameTimecode(0, fps)
    # Test case has three scenes: (10, 20), (30, 40), and (60, 80) with a length of 99
    # and a framerate of 10.
    EVENTS = [
        (10, EventType.IN),
        (20, EventType.OUT),
        (30, EventType.IN),
        (40, EventType.OUT),
        (60, EventType.IN),
        (80, EventType.OUT),
    ]
    _set_events(scene_manager=sm, events=EVENTS, base_timecode=bt)
    sm._start_pos = bt + 5 # Start position is before first IN event, should not affect results.
    sm._last_pos = bt + 99 # Last scene if created should end at 10.0s.

    assert sm.get_scene_list() == [(bt + 10, bt + 20), (bt + 30, bt + 40), (bt + 60, bt + 80)]
    assert sm.get_scene_list(always_include_end=True) == [(bt + 10, bt + 20), (bt + 30, bt + 40),
                                                          (bt + 60, bt + 80), (bt + 80, bt + 100)]


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

    _ = sm.detect_scenes(video=video, end_time=end_time, callback=fake_callback.get_callback_lambda())
    assert fake_callback.num_invoked == (len(sm.get_scene_list()) - 1)

    sm.clear()
    fake_callback.num_invoked = 0
    video.seek(start_time)

    _ = sm.detect_scenes(video=video, end_time=end_time, callback=fake_callback.get_callback_func())
    assert fake_callback.num_invoked == (len(sm.get_scene_list()) - 1)
