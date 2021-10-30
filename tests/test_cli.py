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

import subprocess

import pytest


def invoke_command(command: str):
    return subprocess.call(command.strip().split(' '))


VIDEO_PATH = 'tests/goldeneye/goldeneye.mp4'
SCENEDETECT_CMD = 'python -m scenedetect '
SET_TIME = 'time -s 2s -d 6s '    # Seek forward a bit but limit the amount we process.
DEFAULT_DETECTOR = 'detect-content '
GE_CMD = SCENEDETECT_CMD + '-i ' + VIDEO_PATH + ' '
GE_STATS_CMD = GE_CMD + '-s goldeneye.csv ' + SET_TIME

ALL_DETECTORS = ['detect-content', 'detect-adaptive', 'detect-threshold']


def test_cli_no_args():
    assert invoke_command(SCENEDETECT_CMD) == 0


@pytest.mark.parametrize('info_command', ['help', 'about', 'version'])
def test_cli_info_commands(info_command):
    for command in ['help', 'version', 'about']:
        assert invoke_command(SCENEDETECT_CMD + command) == 0


@pytest.mark.parametrize('detector_command', ALL_DETECTORS)
def test_cli_detectors(detector_command: str):
    # Ensure all detectors work with and without a statsfile.
    assert invoke_command(GE_CMD + SET_TIME + detector_command) == 0
    # Run with a statsfile twice to ensure the file is populated with those metrics and re-loaded.
    assert invoke_command(GE_STATS_CMD + detector_command) == 0
    assert invoke_command(GE_STATS_CMD + detector_command) == 0


def test_cli_time():
    # TODO: Add test for timecode formats.
    base_command = GE_CMD + DEFAULT_DETECTOR + 'time '
    # Test setting start, end, and duration.
    assert invoke_command(base_command + '-s 2s -e 8s') == 0    # start/end
    assert invoke_command(base_command + '-s 2s -d 6s') == 0    # start/duration
    # Ensure cannot set end and duration at the same time.
    assert invoke_command(base_command + '-s 2s -d 6s -e 8s') != 0    # start/duration


def test_cli_list_scenes():
    base_command = GE_STATS_CMD + DEFAULT_DETECTOR + 'list-scenes '
    assert invoke_command(base_command) == 0
    assert invoke_command(base_command + '-n') == 0    # no output file
    # TODO: Check for existence of scene list file, remove after.


@pytest.mark.skip(reason="TODO(v1.0): Command not functional yet.")
def test_cli_split_video():
    base_command = GE_STATS_CMD + DEFAULT_DETECTOR + 'split-video '
    assert invoke_command(base_command) == 0
    # TODO: Check for existence of split video files, remove after.


def test_cli_save_images():
    base_command = GE_STATS_CMD + DEFAULT_DETECTOR + 'save-images '
    assert invoke_command(base_command) == 0
    # TODO: Check for existence of images, remove after.


@pytest.mark.skip(reason="TODO(v1.0): Command not functional yet.")
def test_cli_export_html():
    base_command = GE_STATS_CMD + DEFAULT_DETECTOR
    assert invoke_command(base_command + 'save-images export-html') == 0
    assert invoke_command(base_command + 'export-html --no-images') == 0
    # TODO: Check for existence of HTML & image files, remove after.
