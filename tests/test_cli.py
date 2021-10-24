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

""" PySceneDetect CLI Tests"""

# Standard project pylint disables for unit tests using pytest.
# pylint: disable=no-self-use, protected-access, multiple-statements, invalid-name
# pylint: disable=redefined-outer-name

from scenedetect.platform import invoke_command

# TODO: Add more test cases:
# - time (e.g. can't specify -e and -d)
# - save-images
# - list-scenes
# - detectors
#

def test_basic_usage():
    assert invoke_command('python -m scenedetect -i goldeneye.mp4 time -s 60s detect-content') == 0

def test_missing_input():
    # Certain commands should gracefully allow no input
    assert invoke_command('python -m scenedetect help') == 0
    assert invoke_command('python -m scenedetect version') == 0
    assert invoke_command('python -m scenedetect about') == 0
    # All other commands should require an input param.
    assert invoke_command('python -m scenedetect detect-content')
    assert invoke_command('python -m scenedetect time -s 60s')
    assert invoke_command('python -m scenedetect list-scenes')
    assert invoke_command('python -m scenedetect -s stats.csv')

def test_missing_command():
    assert invoke_command('python -m scenedetect -i goldeneye.mp4')


