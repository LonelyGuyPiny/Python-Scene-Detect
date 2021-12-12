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
"""PySceneDetect scenedetect.scene_list SceneList Tests"""

# Standard project pylint disables for unit tests using pytest.
# pylint: disable=protected-access, invalid-name, unused-argument, redefined-outer-name

import glob
import os
import os.path
from typing import Iterable, Tuple

import pytest

from scenedetect.frame_timecode import FrameTimecode
from scenedetect.scene_list import SceneList

BASE_TIME = FrameTimecode(timecode=0, fps=10.0)

def make_scene_list(start: int, end: int, scenes: Iterable[Tuple[int, int]]) -> SceneList:
    return SceneList(
        start_time=BASE_TIME + start,
        end_time=BASE_TIME + end,
        scenes=[(BASE_TIME + start, BASE_TIME + end) for start, end in scenes])

def test_drop_short_scenes():
    """Test `drop_short_scenes`"""
    scenes = make_scene_list(start=5, end=59, scenes=[(2, 2), (10, 20), (20, 40)])

    # The presentation time of the last frame of the scene is included in min_scene_len. This means
    # setting min_scene_len to 1 will have no effect, since scenes can be no smaller than 1 frame.
    assert scenes.drop_short_scenes(min_scene_len=0) == [(2, 2), (10, 20), (20, 40)]
    assert scenes.drop_short_scenes(min_scene_len=1) == [(2, 2), (10, 20), (20, 40)]
    assert scenes.drop_short_scenes(min_scene_len=2) == [(10, 20), (20, 40)]

    # The scene that starts on frame 10 and ends on frame 20 is technically 11 frames long,
    # since we need to account for the presentation time of the final frame.
    assert scenes.drop_short_scenes(min_scene_len=10) == [(10, 20), (20, 40)]
    assert scenes.drop_short_scenes(min_scene_len=11) == [(10, 20), (20, 40)]
    assert scenes.drop_short_scenes(min_scene_len=12) == [(20, 40)]

    # Same as above for the last scene.
    assert scenes.drop_short_scenes(min_scene_len=20) == [(20, 40)]
    assert scenes.drop_short_scenes(min_scene_len=21) == [(20, 40)]
    assert scenes.drop_short_scenes(min_scene_len=22) == []

def test_merge_short_scenes():
    """Test `merge_short_scenes`"""
    scenes = make_scene_list(start=5, end=59, scenes=[(10, 20), (20, 40)])
    assert scenes.merge_short_scenes(min_scene_len=10) == [(10, 20), (20, 40)]
    assert scenes.merge_short_scenes(min_scene_len=20) == [(10, 40)]
