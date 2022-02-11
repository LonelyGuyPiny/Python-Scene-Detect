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
"""PySceneDetect scenedetect.scene_list Tests

This file includes unit tests for the SceneList object which applies a set of transformations to
Scene objects, which are essentially pairs of FrameTimecodes representing a scene's start/end."""

from typing import Iterable, Tuple

import pytest

from scenedetect.frame_timecode import FrameTimecode
from scenedetect.scene_list import SceneList


def make_scene_list(scenes: Iterable[Tuple[int, int]]) -> SceneList:
    """Create a SceneList from a list of pairs of ints representing frame numbers.
    The framerate is set at 10 FPS to make timecode calculations easier."""
    base_time = FrameTimecode(timecode=0, fps=10.0)
    return SceneList(scenes=[(base_time + start, base_time + end) for start, end in scenes])


def test_drop():
    """Test `drop`"""
    scenes = make_scene_list(scenes=[(2, 2), (10, 20), (20, 40)])

    # The presentation time of the last frame of the scene is included in min_len. This means
    # setting min_len to 1 will have no effect, since scenes can be no smaller than 1 frame.
    assert scenes.drop(min_len=0) == [(2, 2), (10, 20), (20, 40)]
    assert scenes.drop(min_len=1) == [(2, 2), (10, 20), (20, 40)]
    assert scenes.drop(min_len=2) == [(10, 20), (20, 40)]

    # The scene that starts on frame 10 and ends on frame 20 is technically 11 frames long,
    # since we need to account for the presentation time of the final frame.
    assert scenes.drop(min_len=10) == [(10, 20), (20, 40)]
    assert scenes.drop(min_len=11) == [(10, 20), (20, 40)]
    assert scenes.drop(min_len=12) == [(20, 40)]

    # Same as above for the last scene.
    assert scenes.drop(min_len=20) == [(20, 40)]
    assert scenes.drop(min_len=21) == [(20, 40)]
    assert scenes.drop(min_len=22) == []


def test_merge():
    """Test `merge`."""
    scenes = make_scene_list(scenes=[(10, 20), (30, 40), (40, 60)])
    # Test changing `min_len`. Note that only adjacent scenes are considered for
    # merging as the default value of `max_dist` is 0.
    assert scenes.merge(min_len=10, max_dist=9) == [(10, 20), (30, 40), (40, 60)]
    assert scenes.merge(min_len=11, max_dist=9) == [(10, 20), (30, 60)]
    assert scenes.merge(min_len=20, max_dist=9) == [(10, 20), (30, 60)]
    assert scenes.merge(min_len=21, max_dist=9) == [(10, 20), (30, 60)]

    # Test `max_dist`.
    assert scenes.merge(min_len=11, max_dist=9) == [(10, 20), (30, 60)]
    assert scenes.merge(min_len=11, max_dist=10) == [(10, 40), (40, 60)]
    assert scenes.merge(min_len=21, max_dist=9) == [(10, 20), (30, 60)]
    assert scenes.merge(min_len=21, max_dist=10) == [(10, 60)]


def test_contract():
    """Test `contract`"""
    scenes = make_scene_list(scenes=[(10, 40), (40, 50), (50, 60)])
    assert scenes.contract(start=5) == [(15, 40), (45, 50), (55, 60)]
    assert scenes.contract(end=5) == [(10, 35), (40, 45), (50, 55)]
    assert scenes.contract(start=5, end=5) == [(15, 35)]
    assert scenes.contract(start=10, end=10) == [(20, 30)]


@pytest.mark.skip(reason="TODO(v1.0): Finish SceneList expand().")
def test_expand_merge():
    """Test `expand` with merge=True (default)."""
    scenes = make_scene_list(scenes=[(10, 20), (30, 40), (40, 60)])
    assert scenes.expand(start=5) == [(5, 20), (25, 60)]
    assert scenes.expand(end=5) == [(10, 25), (30, 65)]
    assert scenes.expand(start=5, end=5) == [(5, 25), (25, 65)]
    # Note that these values now result in some overlap which was still not merged.
    assert scenes.expand(start=10, end=10) == [(0, 30), (20, 70)]
    # Once the expanded regions cover the original boundaries, they should be merged.
    # As the SceneList doesn't know the original video's start/end times, they may be
    # extended past the original video boundaries.
    assert scenes.expand(start=10, end=11) == [(0, 71)]
    assert scenes.expand(start=11, end=10) == [(0, 70)]
    assert scenes.expand(start=11, end=11) == [(0, 71)]


@pytest.mark.skip(reason="TODO(v1.0): Finish SceneList expand().")
def test_expand_no_merge():
    """Test `expand` with merge=False."""
    scenes = make_scene_list(scenes=[(10, 20), (30, 40), (40, 60)])
    assert scenes.expand(start=5, merge=False) == [(5, 20), (25, 40), (35, 60)]
    assert scenes.expand(end=5, merge=False) == [(10, 25), (30, 45), (40, 65)]
    assert scenes.expand(start=5, end=10, merge=False) == [(5, 30), (25, 50), (35, 70)]
    assert scenes.expand(start=15, end=10, merge=False) == [(0, 30), (15, 50), (25, 70)]
