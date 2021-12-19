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
""" ``scenedetect.scene_list`` Module

This module contains the `Scene` and `SceneList` types.

TODO(v1.0): Documentation.

Example usage:

scenes = sm.get_scene_list()

processed_scenes = scenes.merge(10).drop(10)

"""

from enum import Enum
from typing import Sequence, List, Tuple, Optional, Union

from scenedetect.frame_timecode import FrameTimecode


class Scene(tuple):
    """Scene objects are tuples of two FrameTimecode objects with named accessors for
    [0] as `start` and [1] as `end`. Also provides `duration` as a property.

    For example:

    .. code:: python
        scene = Scene(start=5, end=10)
        assert scene.start == scene[0]
        assert scene.end == scene[1]
    """

    def __new__(self, start: FrameTimecode, end: FrameTimecode):
        return super().__new__(Scene, (start, end))

    @property
    def start(self) -> FrameTimecode:
        return self[0]

    @property
    def end(self) -> FrameTimecode:
        return self[1]

    @property
    def duration(self) -> FrameTimecode:
        return self[1] - self[0]


# Helper for type hints of references to `self` within `SceneList` to allow autocomplete
# to work properly.
_SceneListTypeHint = Union['SceneList', List[Scene]]


class SceneList(list):
    """A list of sorted non-overlapping scenes with helper methods to transform the scenes.

    Note that the following invariants required for every scene `n` in the SceneList:

        scenes[n].start < scenes[n].end
        scenes[n].start < scenes[n+1].start
        scenes[n].end <= scenes[n+1].start

    These invariants are not enforced at runtime and should be guaranteed by the caller during
    construction of the SceneList object.

    Where certain methods may break these invariants (e.g. `expand`), they will return a list of
    Scene objects rather than a SceneList.
    """
    def __init__(self,
                 scenes: Sequence[Union[Tuple[FrameTimecode, FrameTimecode], Scene]] = None):
        super().__init__([Scene(start=start, end=end) for start, end in scenes])

    def drop(self, min_len: FrameTimecode) -> 'SceneList':
        """
            min_len: Minimum length each scene must be. If a given scene is smaller than
                this amount, it will be dropped/removed from the resulting SceneList.
        """
        # Range check inputs before proceeding.
        if min_len < 0:
            raise ValueError('min_len is out of range! Value must be 0 or greater.')
        if min_len <= 1:
            return self
        return SceneList([scene for scene in self if ((scene[1] - scene[0]) + 1) >= min_len])


    def merge(self: _SceneListTypeHint,
                    min_len: Optional[FrameTimecode],
                    max_dist: Union[FrameTimecode, int] = 0) -> 'SceneList':
        """Merge scenes meeting the specified criteria with their closest neighbor. When both
        neighbors are the same distance apart from a merge candidate, it will be merged with the
        next scene.

        Arguments:
            min_len:
                If None, all scenes will be considered for merging.

                Minimum length each scene must be to be considered for merging. If a given scene is smaller than
                this amount, the scene will be merged with the first successor less than
                `max_dist` away, if one exists, otherwise any predecessor less than `max_dist`.

                Scenes shorter than this value but with no merge candidates are unmodified.
                They can be removed by calling `drop` afterwards.

            max_dist:
                The maximum distance scenes can be apart to be considered for merging.
                A value of 0 (default) will only merge scenes which are directly adjacent (the end
                frame of the current scene is the same as the start frame of the next one).
        """
        # Range check inputs before proceeding.
        if min_len < 0:
            raise ValueError('min_len is out of range! Value must be 0 or greater.')
        if max_dist < 0:
            raise ValueError('max_dist is out of range!')
        scene_list = SceneList(self)

        # If any scenes are < min_len, first try to merge it with a predecessor,
        # otherwise a successor, that is at most max_dist away.
        i = 0
        while i < len(scene_list):
            # If the scene is too short, attempt merging it.
            if scene_list[i].duration < min_len:
                # Attempt to merge with the next scene.
                if (i + 1) < len(scene_list) and (scene_list[i+1].start - scene_list[i].end) <= max_dist:
                    scene_list[i] = Scene(scene_list[i].start, scene_list[i+1].end)
                    del scene_list[i+1]
                    continue
                # Attempt to merge with the predecessor, which must be in new_scene_list already.
                if i > 0 and (scene_list[i].start - scene_list[i-1].end) <= max_dist:
                    scene_list[i-1] = Scene(scene_list[i-1].start, scene_list[i].end)
                    del scene_list[i]
                    continue
            i += 1
        return scene_list

    def expand(self: _SceneListTypeHint,
               start: Optional[FrameTimecode] = None,
               end: Optional[FrameTimecode] = None,
               merge: bool = True) -> List[Scene]:
        """Expands each scene in the SceneList by shifting each scene's start/end timecodes by the
        specified amounts, merging if any overlap occurs.

            start: Amount to shift each scene's start timecode backwards by.
        Arguments:
            end: Amount to shift each scene's end timecode forwards by.
            merge: If True (default), will merge any scenes which, after expansion, overlap existing
                scenes from the original list. The resulting output may still have overlaps if the
                expand amounts are large enough. If False, resulting scenes may overlap entirely. See
                the documentation for specific examples.
        Returns:
            List of Scene objects, instead of a SceneList, as the result may contain overlaps.
        """

        raise NotImplementedError()

    def contract(self: _SceneListTypeHint,
               start: Union[FrameTimecode, int] = 0,
               end: Union[FrameTimecode, int] = 0) -> 'SceneList':
        """Contract each scene in the SceneList by shifting each scene's start/end timecodes by the
        specified amounts.

        Arguments:
            start: Amount to shift each scene's start timecode forwards by.
            end: Amount to shift each scene's end timecode backwards by.
        """
        return SceneList([
            Scene(start=(scene.start + start) if start is not None else scene.start,
            end=scene.end - end) for scene in self
            if scene.start + start < scene.end - end])

