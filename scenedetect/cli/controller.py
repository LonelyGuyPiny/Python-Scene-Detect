# -*- coding: utf-8 -*-
#
#         PySceneDetect: Python-Based Video Scene Detector
#   ---------------------------------------------------------------
#     [  Site: http://www.bcastell.com/projects/PySceneDetect/   ]
#     [  Github: https://github.com/Breakthrough/PySceneDetect/  ]
#     [  Documentation: http://pyscenedetect.readthedocs.org/    ]
#
# Copyright (C) 2014-2022 Brandon Castellano <http://www.bcastell.com>.
#
# PySceneDetect is licensed under the BSD 3-Clause License; see the included
# LICENSE file, or visit one of the above pages for details.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
""" ``scenedetect.cli.controller`` Module

This file contains the implementation of the PySceneDetect command-line logic.
"""

import logging
import os
from string import Template
import time
from typing import Dict, List, Tuple, Optional

import click
import cv2

from scenedetect.backends import VideoStreamCv2
from scenedetect.cli.context import CliContext
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.platform import (get_and_create_path)
from scenedetect.scene_manager import (save_images, write_scene_list, write_scene_list_html)
from scenedetect.video_splitter import (is_mkvmerge_available, is_ffmpeg_available,
                                        split_video_mkvmerge, split_video_ffmpeg)

logger = logging.getLogger('pyscenedetect')


# TODO(v0.6): Move this function and it's associated helper into a separate module, and
# pass the CliContext in explicitly. That way CliContext is only responsible for processing
# and validating the specified input options / user configuration, and the actual control
# flow has it's own place (call it scenedetect.cli.controller).
def run_scenedetect(context: CliContext):
    """Run scenedetect: Perform main CLI application control logic.

    Arguments:
        context: Validated command-line option context to use for processing.
    """
    if not context.process_input_flag:
        logger.debug('Skipping processing (process_input_flag is False).')
        return
    if not context.options_processed:
        logger.debug('Skipping processing, CLI options were not parsed successfully.')
        return
    logger.debug('Processing input...')
    context.check_input_open()
    if context.scene_manager.get_num_detectors() == 0:
        logger.error('No scene detectors specified (detect-content, detect-threshold, etc...),\n'
                     ' or failed to process all command line arguments.')
        return

    # Display a warning if the video codec type seems unsupported (#86).
    if isinstance(context.video_stream, VideoStreamCv2):
        if int(abs(context.video_stream.capture.get(cv2.CAP_PROP_FOURCC))) == 0:
            logger.error(
                'Video codec detection failed, output may be incorrect.\nThis could be caused'
                ' by using an outdated version of OpenCV, or using codecs that currently are'
                ' not well supported (e.g. VP9).\n'
                'As a workaround, consider re-encoding the source material before processing.\n'
                'For details, see https://github.com/Breakthrough/PySceneDetect/issues/86')

    logger.info('Detecting scenes...')
    perf_start_time = time.time()
    if context.start_time is not None:
        context.video_stream.seek(target=context.start_time)
    num_frames = context.scene_manager.detect_scenes(
        video=context.video_stream,
        duration=context.duration,
        end_time=context.end_time,
        frame_skip=context.frame_skip,
        show_progress=not context.quiet_mode)

    # Handle case where video failure is most likely due to multiple audio tracks (#179).
    if num_frames <= 0:
        logger.critical(
            'Failed to read any frames from video file. This could be caused by the video'
            ' having multiple audio tracks. If so, try installing the PyAV backend:\n'
            '      pip install av\n'
            'Or remove the audio tracks by running either:\n'
            '      ffmpeg -i input.mp4 -c copy -an output.mp4\n'
            '      mkvmerge -o output.mkv input.mp4\n'
            'For details, see https://pyscenedetect.readthedocs.io/en/latest/faq/')
        return

    perf_duration = time.time() - perf_start_time
    logger.info('Processed %d frames in %.1f seconds (average %.2f FPS).', num_frames,
                perf_duration,
                float(num_frames) / perf_duration)

    # Handle -s/--stats option.
    _save_stats(context)

    # Get list of detected cuts/scenes from the SceneManager to generate the required output
    # files, based on the given commands (list-scenes, split-video, save-images, etc...).
    cut_list = context.scene_manager.get_cut_list()
    scene_list = context.scene_manager.get_scene_list(start_in_scene=True)

    # Handle --drop-short-scenes.
    if context.drop_short_scenes and context.min_scene_len > 0:
        scene_list = [s for s in scene_list if (s[1] - s[0]) >= context.min_scene_len]

    # Ensure we don't divide by zero.
    if scene_list:
        logger.info(
            'Detected %d scenes, average shot length %.1f seconds.', len(scene_list),
            sum([(end_time - start_time).get_seconds() for start_time, end_time in scene_list]) /
            float(len(scene_list)))
    else:
        logger.info('No scenes detected.')

    # Handle list-scenes command.
    _list_scenes(context, scene_list, cut_list)

    # Handle save-images command.
    image_filenames = _save_images(context, scene_list)

    # Handle export-html command.
    _export_html(context, scene_list, cut_list, image_filenames)

    # Handle split-video command.
    _split_video(context, scene_list)


def _save_stats(context: CliContext) -> None:
    """Handles saving the statsfile if -s/--stats was specified."""
    if context.stats_file_path is not None:
        # We check if the save is required in order to reduce unnecessary log messages.
        if context.stats_manager.is_save_required():
            logger.info('Saving frame metrics to stats file: %s',
                        os.path.basename(context.stats_file_path))
            context.stats_manager.save_to_csv(
                path=context.stats_file_path, base_timecode=context.video_stream.base_timecode)
        else:
            logger.debug('No frame metrics updated, skipping update of the stats file.')


def _list_scenes(context: CliContext, scene_list: List[Tuple[FrameTimecode, FrameTimecode]],
                 cut_list: List[FrameTimecode]) -> None:
    """Handles the `list-scenes` command."""
    if context.scene_list_output:
        scene_list_filename = Template(
            context.scene_list_name_format).safe_substitute(VIDEO_NAME=context.video_stream.name)
        if not scene_list_filename.lower().endswith('.csv'):
            scene_list_filename += '.csv'
        scene_list_path = get_and_create_path(
            scene_list_filename, context.scene_list_directory
            if context.scene_list_directory is not None else context.output_directory)
        logger.info('Writing scene list to CSV file:\n  %s', scene_list_path)
        with open(scene_list_path, 'wt') as scene_list_file:
            write_scene_list(
                output_csv_file=scene_list_file,
                scene_list=scene_list,
                include_cut_list=not context.skip_cuts,
                cut_list=cut_list)

    if context.print_scene_list:
        logger.info(
            """Scene List:
-----------------------------------------------------------------------
| Scene # | Start Frame |  Start Time  |  End Frame  |   End Time   |
-----------------------------------------------------------------------
%s
-----------------------------------------------------------------------
""", '\n'.join([
                ' |  %5d  | %11d | %s | %11d | %s |' %
                (i + 1, start_time.get_frames(), start_time.get_timecode(), end_time.get_frames(),
                 end_time.get_timecode()) for i, (start_time, end_time) in enumerate(scene_list)
            ]))

    if cut_list:
        logger.info('Comma-separated timecode list:\n  %s',
                    ','.join([cut.get_timecode() for cut in cut_list]))


# TODO(v0.6): Test save-images output matches the frames from v0.5.x.
def _save_images(
        context: CliContext,
        scene_list: List[Tuple[FrameTimecode, FrameTimecode]]) -> Optional[Dict[int, List[str]]]:
    """Handles the `save-images` command."""
    if not context.save_images:
        return None

    image_output_dir = context.output_directory
    if context.image_directory is not None:
        image_output_dir = context.image_directory

    return save_images(
        scene_list=scene_list,
        video=context.video_stream,
        num_images=context.num_images,
        frame_margin=context.frame_margin,
        image_extension=context.image_extension,
        encoder_param=context.image_param,
        image_name_template=context.image_name_format,
        output_dir=image_output_dir,
        show_progress=not context.quiet_mode,
        scale=context.scale,
        height=context.height,
        width=context.width)


def _export_html(context: CliContext, scene_list: List[Tuple[FrameTimecode, FrameTimecode]],
                 cut_list: List[FrameTimecode], image_filenames: Optional[Dict[int,
                                                                               List[str]]]) -> None:
    """Handles the `export-html` command."""
    if not context.export_html:
        return

    html_filename = Template(
        context.html_name_format).safe_substitute(VIDEO_NAME=context.video_stream.name)
    if not html_filename.lower().endswith('.html'):
        html_filename += '.html'
    html_path = get_and_create_path(
        html_filename, context.image_directory
        if context.image_directory is not None else context.output_directory)
    logger.info('Exporting to html file:\n %s:', html_path)
    if not context.html_include_images:
        image_filenames = None
    write_scene_list_html(
        html_path,
        scene_list,
        cut_list,
        image_filenames=image_filenames,
        image_width=context.image_width,
        image_height=context.image_height)


def _split_video(context: CliContext, scene_list: List[Tuple[FrameTimecode,
                                                             FrameTimecode]]) -> None:
    """Handles the `split-video` command."""
    if not context.split_video:
        return

    output_path_template = context.split_name_format
    # Add proper extension to filename template if required.
    dot_pos = output_path_template.rfind('.')
    extension_length = 0 if dot_pos < 0 else len(output_path_template) - (dot_pos + 1)
    # If using mkvmerge, force extension to .mkv.
    if context.split_mkvmerge and not output_path_template.endswith('.mkv'):
        output_path_template += '.mkv'
    # Otherwise, if using ffmpeg, only add an extension if one doesn't exist.
    elif not 2 <= extension_length <= 4:
        output_path_template += '.mp4'
    output_path_template = get_and_create_path(
        output_path_template, context.split_directory
        if context.split_directory is not None else context.output_directory)
    # Ensure the appropriate tool is available before handling split-video.
    _check_split_video_requirements(context.split_mkvmerge)
    if context.split_mkvmerge:
        split_video_mkvmerge(
            context.video_stream.path,
            scene_list,
            output_path_template,
            show_output=not (context.quiet_mode or context.split_quiet),
        )
    else:
        split_video_ffmpeg(
            context.video_stream.path,
            scene_list,
            output_path_template,
            arg_override=context.split_args,
            show_progress=not context.quiet_mode,
            show_output=not (context.quiet_mode or context.split_quiet),
        )
    if scene_list:
        logger.info('Video splitting completed, individual scenes written to disk.')


def _check_split_video_requirements(use_mkvmerge: bool) -> None:
    # type: (bool) -> None
    """ Validates that the proper tool is available on the system to perform the split-video
    command, which depends on if -m/--mkvmerge is set (if not, defaults to ffmpeg).

    Arguments:
        use_mkvmerge: True if -m/--mkvmerge is set, False otherwise.

    Raises: click.BadParameter if the proper video splitting tool cannot be found.
    """

    if (use_mkvmerge and not is_mkvmerge_available()) or not is_ffmpeg_available():
        error_strs = [
            "{EXTERN_TOOL} is required for split-video{EXTRA_ARGS}.".format(
                EXTERN_TOOL='mkvmerge' if use_mkvmerge else 'ffmpeg',
                EXTRA_ARGS=' -m/--mkvmerge' if use_mkvmerge else '')
        ]
        error_strs += ["Install one of the above tools to enable the split-video command."]
        if not use_mkvmerge and is_mkvmerge_available():
            error_strs += ['You can also specify `-m/--mkvmerge` to use mkvmerge for splitting.']
        elif use_mkvmerge and is_ffmpeg_available():
            error_strs += ['You can also specify `-c/--copy` to use ffmpeg stream copying.']
        error_str = '\n'.join(error_strs)
        raise click.BadParameter(error_str, param_hint='split-video')
