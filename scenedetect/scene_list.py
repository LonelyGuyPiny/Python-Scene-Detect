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

TODO(v1.0): Documentation.

Example usage:

scenes = sm.get_scene_list()

processed_scenes = scenes.merge_short_scenes(20, 10).drop_short_scenes(10)

"""

from string import Template
from typing import Iterable, List, Tuple, Optional, Dict, Callable, Union, Set
from enum import Enum
import logging
from scenedetect.frame_timecode import FrameTimecode

logger = logging.getLogger('pyscenedetect')


class MergeStrategy(Enum):
    # TODO(v1.0)

    # NEAREST_PREFER_PREDECESSOR = 0
    # """Use the nearest scene, preferring the predecessor if both are the same distance away."""
    #
    # NEAREST_PREFER_SUCCESSOR = 1
    # """Use the nearest scene, preferring the successor if both are the same distance away."""
    #
    PREFER_PREDECESSOR = 2
    """Use the predecessor, if any, otherwise the successor."""

    # PREFER_SUCCESSOR = 3
    # """Use the successor, if any, otherwise the predecessor."""
    #
    # ONLY_PREDECESSOR = 4
    # """Only use the predecessor, if any."""
    #
    # ONLY_SUCCESSOR = 5
    # """Only use the successor, if any."""


class Scene(tuple):

    def __new__(self, start: FrameTimecode, end: FrameTimecode):
        return tuple.__new__(Scene, (start, end))

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

#
# Assumptions for every scene `n` in the SceneList:
#   scenes[n].start < scenes[n].end
#   scenes[n].start < scenes[n+1].start
#   scenes[n].end <= scenes[n+1].start
#
class SceneList(list):

    def __init__(self,
                 start_time: FrameTimecode,
                 end_time: FrameTimecode,
                 scenes: Iterable[Tuple[FrameTimecode, FrameTimecode]] = None):
        super().__init__([Scene(start=start, end=end) for start, end in scenes])
        self._start = start_time
        self._end = end_time

    def clone(self, scenes: 'SceneList' = None):
        # TODO: Test shallow copy ops.
        return SceneList(self._start, self._end, list(self) if scenes is None else scenes)

    def drop_short_scenes(self, min_scene_len: FrameTimecode) -> 'SceneList':
        """
            min_scene_len: Minimum length each scene must be. If a given scene is smaller than
                this amount, it will be dropped/removed from the resulting SceneList.
                TODO(v1.0): Make default value 0.3s after FrameTimecode -> Timecode refactor.
        """
        # Range check inputs before proceeding.
        if min_scene_len < 0:
            raise ValueError('min_scene_len is out of range! Value must be 0 or greater.')
        if min_scene_len <= 1:
            return self.clone()
        return self.clone([scene for scene in self if ((scene[1] - scene[0]) +1) >= min_scene_len])

    def merge_short_scenes(self: _SceneListTypeHint,
                           min_scene_len: FrameTimecode,
                           merge_len: Union[FrameTimecode, int] = 0,
                           merge_strategy: MergeStrategy = MergeStrategy.PREFER_PREDECESSOR) -> 'SceneList':
        """
            min_scene_len: Minimum length each scene must be. If a given scene is smaller than
                this amount, the scene will be merged with the first predecessor less than
                `merge_len` away, if one exists, otherwise the first successor less than `merge_len`.

                Scenes which are shorter than min_scene_len but were not merged are unmodified.
                They can be removed by calling `drop_short_scenes` afterwards.

                TODO(v1.0): Make default value 0.3s after FrameTimecode -> Timecode refactor.
            merge_len: Specifies the maximum distance an adjacent scene can be, in time, from scenes
                smaller than `min_scene_len` before they are dropped instead of merged. The default
                (0) only merges scenes shorter than `min_scene_len` if there is a scene directly
                adjacent to it.

            merge_strategy: Strategy to use for merging scenes when there are multiple candidates.
                See `MergeStrategy` for details on each strategy.
        """
        # Range check inputs before proceeding.
        if min_scene_len < 0:
            raise ValueError('min_scene_len is out of range! Value must be 0 or greater.')
        if merge_len is not None and merge_len < 0:
            raise ValueError('merge_len is out of range!')
        # TODO(v1.0): Implement merge_strategy.
        if merge_strategy != MergeStrategy.PREFER_PREDECESSOR:
            raise NotImplementedError('TODO(v1.0)')

        # Empty clone of self to store the result.
        scene_list = self.clone([])

        # If any scenes are < min_scene_len, first try to merge it with a predecessor,
        # otherwise a successor, that is at most merge_len away.
        for i, scene in enumerate(self):
            # If the scene is long enough, add it and continue.
            if (scene.end - scene.start) >= min_scene_len:
                scene_list.append(scene)
                continue
            next_scene = self[i + 1] if (i + 1) < len(self) else None
            # Attempt to merge with the predecessor, which must be in new_scene_list already.
            if scene_list and (scene.start - scene_list[-1].end) <= merge_len:
                scene_list[-1] = Scene(scene_list[-1].start, scene.end)
            # Attempt to merge with the next scene. We do this by modifying the next scene
            # in-place before we process it.
            elif next_scene and (next_scene[0] - scene[1]) <= merge_len:
                self[i + 1] = Scene(scene[0], next_scene[1])
            # If we get here, the scene will be dropped since it could not be merged.
        return scene_list

    def shift_scenes(self: _SceneListTypeHint,
        shift_start: Union[FrameTimecode, int] = 0,
        shift_end: Union[FrameTimecode, int] = 0,
        merge_overlap = True,
        overlap_bias: float = 0.0) -> 'SceneList':
        """
        Note that the original scene boundaries will still be respected (i.e. there will never be
        any overlapping scenes). If overlapping is desired, use list comprehension instead, e.g.:

            maybe_overlaps = [(start + start_shift, end + end_shift) for start, end in scene_list]

        Arguments:
            shift_start: Amount to add to each scene's start timecode. Negative values
                imply shifting the start time backwards (e.g. making each scene longer).
                If the shifted time overlaps with another scene, they will be merged unless
                overlap_bias is set.
                TODO(v1.0): Allow negative FrameTimecode values in addition to frame numbers.
                Do as part of the FrameTimecode refactor; for now just test using frame numbers,
                and don't expose any CLI options for this under the `post-process` command yet.
            shift_end: Amount to add to each scene's end timecode. Negative values
                imply shifting the end time backwards (e.g. making each scene shorter).
                Overlapping scenes will be merged unless overlap_bias is set.
            merge_overlap: If True (the default), any overlapping scene boundaries which occur due
                to the specified shift values will be merged. If False, overlap_bias will be used
                to calculate the location of the resulting scene boundary.
            overlap_bias: Value between -1.0 and +1.0 representing where the resulting scene
                boundary should be placed when the shift amounts cause scenes to overlap. Note that
                the bias only applies to the contested area, so the original scene boundaries are
                still respected.
        """
        raise NotImplementedError("TODO(v1.0)")




        # Handle shift_start / shift_end / overlap_bias.
        if shift_start != 0 or shift_end != 0:

            # TODO: Represent this as an interval tree of integers that can be negative.
            # First calculate the new splits, THEN worry about video boundaries.

            scene_list = [(start + shift_start, end + shift_end) for (start, end)
                            in scene_list]
            # Merge all scenes outside of the video boundaries.
            #need to represent as interval tree. merge then trim.

            # Drop all scenes that were shifted before the start of the video.
            all_start_at_zero = True
            for i in range(len(scene_list)):
                if scene_list[i][0] > 0 and i > 0:
                    scene_list = scene_list[i-1:]
                    all_start_at_zero = False
                    break
            if all_start_at_zero:
                scene_list = scene_list[-1:]

            # Drop all scenes after we find one that starts beyond the end of the video.
            for i in range(len(scene_list)):
                if scene_list[i][0] > self._last_pos and i > 0:
                    scene_list = scene_list[:i]
                    break

            # Merge any scenes that overlap (assumes scene_list is sorted by scene start time).
            new_scene_list = []
            base_timecode = None if not scene_list else FrameTimecode(0, scene_list[0][0])
            for i in range(len(scene_list) - 1):
                if scene_list[i][1] > scene_list[i+1][0]:
                    if overlap_bias is None:
                        # Merge with next scene by modifying it's start time.
                        scene_list[i+1] = (scene_list[i][0], scene_list[i+1][1])
                    else:
                        # THIS IS WRONG - This can cause the start/end times to be shifted beyond
                        # the original scene boundaries.  Handle the shrinking case first since
                        # those are easier to handle.
                        #
                        # This should not allow scenes to shrink if, for example, shift_start/_end
                        # are negative/positive respectively.
                        #
                        #

                        delta = scene_list[i][1].frame_num - scene_list[i+1][0].frame_num
                        # Translate bias from -1.0-+1.0 to 0.0-1.0 before multiplying.
                        overlap_bias_ratio = (overlap_bias + 1.0) / 2.0
                        new_offset = scene_list[i+1][0].frame_num + round(delta * overlap_bias_ratio)
                        new_timecode = base_timecode + new_offset
                        scene_list[i] = (scene_list[i][0], new_timecode)
                        scene_list[i+1] = (new_timecode, scene_list[i+1][1])
                        new_scene_list.append(scene_list[i])
            scene_list = new_scene_list + scene_list[-1:]