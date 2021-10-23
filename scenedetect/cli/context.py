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

""" ``scenedetect.cli.context`` Module

This file contains the implementation of the PySceneDetect command-line
interface (CLI) context class CliContext, used for the main application
state/context and logic to run the PySceneDetect CLI.
"""

# Standard Library Imports
from __future__ import print_function
import logging
import os
import time
from string import Template
from typing import Optional

# Third-Party Library Imports
import click
import cv2
from scenedetect import video_stream

# PySceneDetect Library Imports
import scenedetect.detectors

from scenedetect.scene_manager import SceneManager
from scenedetect.scene_manager import save_images
from scenedetect.scene_manager import write_scene_list
from scenedetect.scene_manager import write_scene_list_html

from scenedetect.stats_manager import StatsManager
from scenedetect.stats_manager import StatsFileCorrupt

from scenedetect.video_stream import VideoStream

from scenedetect.video_splitter import is_mkvmerge_available
from scenedetect.video_splitter import is_ffmpeg_available
from scenedetect.video_splitter import split_video_mkvmerge
from scenedetect.video_splitter import split_video_ffmpeg

from scenedetect.platform import get_cv2_imwrite_params
from scenedetect.platform import check_opencv_ffmpeg_dll
from scenedetect.platform import get_and_create_path

from scenedetect.frame_timecode import FrameTimecode, MINIMUM_FRAMES_PER_SECOND_FLOAT


from scenedetect.video_stream import VideoOpenFailure
from scenedetect.backends.opencv import VideoStreamCv2


def parse_timecode(cli_ctx, value):
    # type: (CliContext, str) -> Union[FrameTimecode, None]
    """ Parses a user input string expected to be a timecode, given a CLI context.

    Returns:
        (FrameTimecode) Timecode set to value with the CliContext VideoManager framerate.
            If value is None, skips processing and returns None.

    Raises:
        click.BadParameter
     """
    cli_ctx.check_input_open()
    if value is None:
        return value
    try:
        timecode = FrameTimecode(
            timecode=value, fps=cli_ctx.video_stream.frame_rate)
        return timecode
    except (ValueError, TypeError):
        raise click.BadParameter(
            'timecode must be in frames (1234), seconds (123.4s), or HH:MM:SS (00:02:03.400)')


def get_plural(val_list):
    # type: (List[any]) -> str
    """ Get Plural: Helper function to return 's' if a list has more than one (1)
    element, otherwise returns ''.

    Returns:
        str: String of 's' if the length of val_list is greater than 1, otherwise ''.
    """
    return 's' if len(val_list) > 1 else ''


def contains_sequence_or_url(video_path: str) -> bool:
    """Checks if the video path is a URL or image sequence."""
    return '%' in video_path or '://' in video_path


def check_split_video_requirements(use_mkvmerge):
    # type: (bool) -> None
    """ Validates that the proper tool is available on the system to perform the split-video
    command, which depends on if -c/--copy is set (to use mkvmerge) or not (to use ffmpeg).

    Arguments:
        use_mkvmerge: True if -c/--copy is set, False otherwise.

    Raises: click.BadParameter if the proper video splitting tool cannot be found.
    """

    if (use_mkvmerge and not is_mkvmerge_available()) or not is_ffmpeg_available():
        error_strs = [
            "{EXTERN_TOOL} is required for split-video{EXTRA_ARGS}.".format(
                EXTERN_TOOL='mkvmerge' if use_mkvmerge else 'ffmpeg',
                EXTRA_ARGS=' -c/--copy' if use_mkvmerge else '')]
        error_strs += ["Install one of the above tools to enable the split-video command."]
        if not use_mkvmerge and is_mkvmerge_available():
            error_strs += [
                'You can also specify `-c/--copy` to use mkvmerge for splitting.']
        elif use_mkvmerge and is_ffmpeg_available():
            error_strs += [
                'You can also omit `-c/--copy` to use ffmpeg for splitting.']
        error_str = '\n'.join(error_strs)
        raise click.BadParameter(error_str, param_hint='split-video')



class CliContext(object):
    """ Context of the command-line interface passed between the various sub-commands.

    Pools all options, processing the main program options as they come in (e.g. those not passed
    to a command), followed by parsing each sub-command's options.  After preparing the commands,
    their actions are executed by calling the process_input() method.

    The only other module which should directly access/modify the properties of this class is
    `scenedetect.cli.__init__` (file scenedetect/cli/__init__.py).
    """

    def __init__(self):
        # Properties for main scenedetect command options (-i, -s, etc...) and CliContext logic.
        self.options_processed: bool = False    # True when CLI option parsing is complete.
        self.scene_manager: SceneManager = None # detect-* commands and -d/--downscale

        # Input/output options
        self.video_stream: VideoStream = None   # -i/--input and -f/--framerate
        self.base_timecode: FrameTimecode = None
        self.output_directory = None            # -o/--output
        self.quiet_mode = False                 # -q/--quiet or -v/--verbosity quiet

        # Statsfile options
        self.stats_manager: StatsManager = None # -s/--stats
        self.stats_file_path = None

        # Global scene detection parameters
        self.drop_short_scenes = False          # --drop-short-scenes
        self.min_scene_len = None               # -m/--min-scene-len
        self.frame_skip = 0                     # -fs/--frame-skip

        # Properties for save-images command.
        self.save_images = False                # save-images command
        self.image_extension = 'jpg'            # save-images -j/--jpeg, -w/--webp, -p/--png
        self.image_directory = None             # save-images -o/--output

        self.image_param = None                 # save-images -q/--quality if -j/-w,
                                                #   -c/--compression if -p
        self.image_name_format = (              # save-images -f/--name-format
            '$VIDEO_NAME-Scene-$SCENE_NUMBER-$IMAGE_NUMBER')
        self.num_images = 3                     # save-images -n/--num-images
        self.frame_margin = 1                   # save-images -m/--frame-margin
        self.scale = None                       # save-images -s/--scale
        self.height = None                      # save-images -h/--height
        self.width = None                       # save-images -w/--width

        # Properties for split-video command.
        self.split_video = False                # split-video command
        self.split_mkvmerge = False             # split-video -c/--copy
        self.split_args = None                  # split-video -a/--override-args
        self.split_directory = None             # split-video -o/--output
        self.split_name_format = '$VIDEO_NAME-Scene-$SCENE_NUMBER'  # split-video -f/--filename
        self.split_quiet = False                # split-video -q/--quiet

        # Properties for list-scenes command.
        self.list_scenes = False                # list-scenes command
        self.print_scene_list = False           # list-scenes --quiet/-q
        self.scene_list_directory = None        # list-scenes -o/--output
        self.scene_list_name_format = None      # list-scenes -f/--filename
        self.scene_list_output = False          # list-scenes -n/--no-output
        self.skip_cuts = False                  # list-scenes -s/--skip-cuts

        # Properties for export-html command.
        self.export_html = False                # export-html command
        self.html_name_format = None            # export-html -f/--filename
        self.html_include_images = True         # export-html --no-images
        self.image_width = None                 # export-html -w/--image-width
        self.image_height = None                # export-html -h/--image-height

        # Logger for CLI output.
        self.logger = logging.getLogger('pyscenedetect')


    def _open_stats_file(self):
        self.stats_manager = StatsManager()
        if self.stats_file_path is not None:
            if os.path.exists(self.stats_file_path):
                self.logger.info('Loading frame metrics from stats file: %s',
                             os.path.basename(self.stats_file_path))
                try:
                    with open(self.stats_file_path, 'rt') as stats_file:
                        self.stats_manager.load_from_csv(stats_file)
                except StatsFileCorrupt:
                    error_info = (
                        'Could not load frame metrics from stats file - file is either corrupt,'
                        ' or not a valid PySceneDetect stats file. If the file exists, ensure that'
                        ' it is a valid stats file CSV, otherwise delete it and run PySceneDetect'
                        ' again to re-generate the stats file.')
                    error_strs = [
                        'Could not load stats file.', 'Failed to parse stats file:', error_info ]
                    self.logger.error('\n'.join(error_strs))
                    raise click.BadParameter(
                        '\n  Could not load given stats file, see above output for details.',
                        param_hint='input stats file')


    def process_input(self):
        # type: () -> None
        """ Process Input: Processes input video(s) and generates output as per CLI commands.

        Run after all command line options/sub-commands have been parsed.
        """
        self.logger.debug('Processing input...')
        if not self.options_processed:
            self.logger.debug('Skipping processing, CLI options were not parsed successfully.')
            return
        self.check_input_open()
        assert self.scene_manager.get_num_detectors() >= 0
        if self.scene_manager.get_num_detectors() == 0:
            self.logger.error(
                'No scene detectors specified (detect-content, detect-threshold, etc...),\n'
                ' or failed to process all command line arguments.')
            return

        # Display a warning if the video codec type seems unsupported (#86).
        if int(abs(self.video_stream.capture.get(cv2.CAP_PROP_FOURCC))) == 0:
            self.logger.error(
                'Video codec detection failed, output may be incorrect.\nThis could be caused'
                ' by using an outdated version of OpenCV, or using codecs that currently are'
                ' not well supported (e.g. VP9).\n'
                'As a workaround, consider re-encoding the source material before processing.\n'
                'For details, see https://github.com/Breakthrough/PySceneDetect/issues/86')

        # Handle scene detection commands (detect-content, detect-threshold, etc...).

        start_time = time.time()
        self.logger.info('Detecting scenes...')

        num_frames = self.scene_manager.detect_scenes(
            video=self.video_stream, frame_skip=self.frame_skip,
            show_progress=not self.quiet_mode)

        # Handle case where video fails with multiple audio tracks (#179).
        # TODO: Using a different video backend as per #213 may also resolve this issue,
        # as well as numerous other timing related issues.
        if num_frames <= 0:
            self.logger.critical(
                'Failed to read any frames from video file. This could be caused'
                ' by the video having multiple audio tracks. If so, please try'
                ' removing the audio tracks or muxing to mkv via:\n'
                '      ffmpeg -i input.mp4 -c copy -an output.mp4\n'
                'or:\n'
                '      mkvmerge -o output.mkv input.mp4\n'
                'For details, see https://pyscenedetect.readthedocs.io/en/latest/faq/')
            return

        duration = time.time() - start_time
        self.logger.info('Processed %d frames in %.1f seconds (average %.2f FPS).',
                     num_frames, duration, float(num_frames)/duration)

        # Handle -s/--statsfile option.
        if self.stats_file_path is not None:
            if self.stats_manager.is_save_required():
                with open(self.stats_file_path, 'wt') as stats_file:
                    self.logger.info('Saving frame metrics to stats file: %s',
                                 os.path.basename(self.stats_file_path))
                    base_timecode = self.video_stream.base_timecode
                    self.stats_manager.save_to_csv(
                        stats_file, base_timecode)
            else:
                self.logger.debug('No frame metrics updated, skipping update of the stats file.')

        # Get list of detected cuts and scenes from the SceneManager to generate the required output
        # files with based on the given commands (list-scenes, split-video, save-images, etc...).
        cut_list = self.scene_manager.get_cut_list()
        scene_list = self.scene_manager.get_scene_list()

        # Handle --drop-short-scenes.
        if self.drop_short_scenes and self.min_scene_len > 0:
            scene_list = [
                s for s in scene_list
                if (s[1] - s[0]) >= self.min_scene_len
            ]

        if scene_list:  # Ensure we don't divide by zero.
            self.logger.info('Detected %d scenes, average shot length %.1f seconds.',
                         len(scene_list),
                         sum([(end_time - start_time).get_seconds()
                              for start_time, end_time in scene_list]) / float(len(scene_list)))
        else:
            self.logger.info('No scenes detected.')

        # Handle list-scenes command.
        if self.scene_list_output:
            scene_list_filename = Template(self.scene_list_name_format).safe_substitute(
                VIDEO_NAME=self.video_stream.name)
            if not scene_list_filename.lower().endswith('.csv'):
                scene_list_filename += '.csv'
            scene_list_path = get_and_create_path(
                scene_list_filename,
                self.scene_list_directory if self.scene_list_directory is not None
                else self.output_directory)
            self.logger.info('Writing scene list to CSV file:\n  %s', scene_list_path)
            with open(scene_list_path, 'wt') as scene_list_file:
                write_scene_list(
                    output_csv_file=scene_list_file,
                    scene_list=scene_list,
                    include_cut_list=not self.skip_cuts,
                    cut_list=cut_list)

        if self.print_scene_list:
            self.logger.info("""Scene List:
-----------------------------------------------------------------------
 | Scene # | Start Frame |  Start Time  |  End Frame  |   End Time   |
-----------------------------------------------------------------------
%s
-----------------------------------------------------------------------
""", '\n'.join(
    [' |  %5d  | %11d | %s | %11d | %s |' % (
        i+1,
        start_time.get_frames(), start_time.get_timecode(),
        end_time.get_frames(), end_time.get_timecode())
     for i, (start_time, end_time) in enumerate(scene_list)]))

        if cut_list:
            self.logger.info('Comma-separated timecode list:\n  %s',
                         ','.join([cut.get_timecode() for cut in cut_list]))

        # Handle save-images command.
        # TODO(v1.0): Test save-images output matches the frames from v0.5.x.
        if self.save_images:
            image_output_dir = self.output_directory
            if self.image_directory is not None:
                image_output_dir = self.image_directory

            image_filenames = save_images(
                scene_list=scene_list,
                video=self.video_stream,
                num_images=self.num_images,
                frame_margin=self.frame_margin,
                image_extension=self.image_extension,
                encoder_param=self.image_param,
                image_name_template=self.image_name_format,
                output_dir=image_output_dir,
                show_progress=not self.quiet_mode,
                scale=self.scale,
                height=self.height,
                width=self.width)

        # Handle export-html command.
        if self.export_html:
            html_filename = Template(self.html_name_format).safe_substitute(
                VIDEO_NAME=self.video_stream.name)
            if not html_filename.lower().endswith('.html'):
                html_filename += '.html'
            html_path = get_and_create_path(
                html_filename,
                self.image_directory if self.image_directory is not None
                else self.output_directory)
            self.logger.info('Exporting to html file:\n %s:', html_path)
            if not self.html_include_images:
                image_filenames = None
            write_scene_list_html(html_path, scene_list, cut_list,
                                  image_filenames=image_filenames,
                                  image_width=self.image_width,
                                  image_height=self.image_height)

        # Handle split-video command.
        if self.split_video:
            output_path_template = self.split_name_format
            # Add proper extension to filename template if required.
            dot_pos = output_path_template.rfind('.')
            extension_length = 0 if dot_pos < 0 else len(output_path_template) - (dot_pos + 1)
            # If using mkvmerge, force extension to .mkv.
            if self.split_mkvmerge and not output_path_template.endswith('.mkv'):
                output_path_template += '.mkv'
            # Otherwise, if using ffmpeg, only add an extension if one doesn't exist.
            elif not 2 <= extension_length <= 4:
                output_path_template += '.mp4'
            output_path_template = get_and_create_path(
                output_path_template,
                self.split_directory if self.split_directory is not None
                else self.output_directory)
            # Ensure the appropriate tool is available before handling split-video.
            check_split_video_requirements(self.split_mkvmerge)
            if self.split_mkvmerge:
                split_video_mkvmerge([self.video_stream.path], scene_list, output_path_template, self.video_stream.name,
                                     suppress_output=self.quiet_mode or self.split_quiet)
            else:
                split_video_ffmpeg([self.video_stream.path], scene_list, output_path_template,
                                   self.video_stream.name, arg_override=self.split_args,
                                   hide_progress=self.quiet_mode,
                                   suppress_output=self.quiet_mode or self.split_quiet)
            if scene_list:
                self.logger.info('Video splitting completed, individual scenes written to disk.')



    def check_input_open(self):
        # type: () -> None
        """ Check Input Open: Ensures that the CliContext's VideoManager was initialized,
        started, and at *least* one input video was successfully opened - otherwise, an
        exception is raised.

        Raises:
            click.BadParameter
        """
        if self.video_stream is None:
            self.logger.error("-i/--input [VIDEO] must be specified before any commands.")
            raise click.BadParameter('No input specified.', param_hint='input video')


    def add_detector(self, detector):
        """ Add Detector: Adds a detection algorithm to the CliContext's SceneManager. """
        self.check_input_open()
        options_processed_orig = self.options_processed
        self.options_processed = False
        try:
            self.scene_manager.add_detector(detector)
        except scenedetect.stats_manager.FrameMetricRegistered:
            raise click.BadParameter(message='Cannot specify detection algorithm twice.',
                                     param_hint=detector.cli_name)
        self.options_processed = options_processed_orig


    def _init_video_stream(self, input_path: str=None, framerate: float=None):
        self.base_timecode = None
        self.logger.debug('Initializing VideoStreamCv2.')
        try:
            self.video_stream = VideoStreamCv2(path_or_device=input_path, framerate=framerate)
            self.base_timecode = self.video_stream.base_timecode
        except VideoOpenFailure as ex:
            dll_okay, dll_name = check_opencv_ffmpeg_dll()
            if dll_okay:
                self.logger.error('Backend failed to open video: %s', str(ex))
            else:
                self.logger.error(
                    'Error: OpenCV dependency %s not found.'
                    ' Ensure that you installed the Python OpenCV module, and that the'
                    ' %s file can be found to enable video support.',
                    dll_name, dll_name)
                # Add additional output message in red.
                click.echo(click.style(
                    '\nOpenCV dependency missing, video input/decoding not available.\n', fg='red'))
            raise click.BadParameter('Failed to open video!', param_hint='input video')
        except IOError as ex:
            self.logger.error('Input error: %s', str(ex))
            raise click.BadParameter('Input error!', param_hint='input video')

        if self.video_stream.frame_rate < MINIMUM_FRAMES_PER_SECOND_FLOAT:
            self.logger.error(
                'Failed to obtain framerate for input video.'
                ' Manually specify framerate with the -f/--framerate option.')
            raise click.BadParameter('Failed to get framerate!', param_hint='input video')



    def parse_options(self, input_path: str, framerate: float, stats_file: Optional[str], downscale: Optional[int], frame_skip: int,
                      min_scene_len: int, drop_short_scenes: bool):
        """ Parse Options: Parses all global options/arguments passed to the main
        scenedetect command, before other sub-commands (e.g. this function processes
        the [options] when calling scenedetect [options] [commands [command options]].

        This method calls the _init_video_stream(), _open_stats_file(), and
        check_input_open() methods, which may raise a click.BadParameter exception.

        Raises:
            click.BadParameter
        """

        self.logger.debug('Parsing program options.')

        self.frame_skip = frame_skip

        self._init_video_stream(input_path=input_path, framerate=framerate)

        # Open StatsManager if --stats is specified.
        if stats_file:
            self.stats_file_path = get_and_create_path(stats_file, self.output_directory)
            self._open_stats_file()

        # Init SceneManager.
        self.logger.debug('Initializing SceneManager.')
        self.scene_manager = SceneManager(self.stats_manager)
        if downscale is None:
            self.scene_manager.auto_downscale = True
        else:
            try:
                self.scene_manager.downscale = downscale
            except ValueError as ex:
                self.logger.debug(str(ex))
                raise click.BadParameter(str(ex), param_hint='downscale factor')

        self.drop_short_scenes = drop_short_scenes
        self.min_scene_len = parse_timecode(self, min_scene_len)


    # TODO(v1.0): Requires update.
    def time_command(self, start=None, duration=None, end=None):
        # type: (Optional[str], Optional[str], Optional[str]) -> None
        """ Time Command: Parses all options/arguments passed to the time command,
        or with respect to the CLI, this function processes [time options] when calling:
        scenedetect [global options] time [time options] [other commands...].

        Raises:
            click.BadParameter, VideoDecodingInProgress
        """
        self.logger.debug('Setting video time:\n    start: %s, duration: %s, end: %s',
                      start, duration, end)

        self.check_input_open()

        if duration is not None and end is not None:
            raise click.BadParameter(
                'Only one of --duration/-d or --end/-e can be specified, not both.',
                param_hint='time')

        raise click.BadParameter('TODO - time command needs to be implemented for v1.0.', param_hint='time')

        #self.video_manager.set_duration(start_time=start, duration=duration, end_time=end)
        if start is not None:
            self.video_stream.seek(target=start)


    def list_scenes_command(self, output_path, filename_format, no_output_mode,
                            quiet_mode, skip_cuts):
        # type: (str, str, bool, bool) -> None
        """ List Scenes Command: Parses all options/arguments passed to the list-scenes command,
        or with respect to the CLI, this function processes [list-scenes options] when calling:
        scenedetect [global options] list-scenes [list-scenes options] [other commands...].

        Raises:
            click.BadParameter
        """
        self.check_input_open()

        self.print_scene_list = True if quiet_mode is None else not quiet_mode
        self.scene_list_directory = output_path
        self.scene_list_name_format = filename_format
        if self.scene_list_name_format is not None and not no_output_mode:
            self.logger.info('Scene list CSV file name format:\n  %s', self.scene_list_name_format)
        self.scene_list_output = False if no_output_mode else True
        if self.scene_list_directory is not None:
            self.logger.info('Scene list output directory set:\n  %s', self.scene_list_directory)
        self.skip_cuts = skip_cuts


    def export_html_command(self, filename, no_images, image_width, image_height):
        # type: (str, bool) -> None
        """Export HTML command: Parses all options/arguments passed to the export-html command,
        or with respect to the CLI, this function processes [export-html] options when calling:
        scenedetect [global options] export-html [export-html options] [other commands...].

        Raises:
            click.BadParameter
        """
        self.check_input_open()

        self.html_name_format = filename
        if self.html_name_format is not None:
            self.logger.info('Scene list html file name format:\n %s', self.html_name_format)
        self.html_include_images = False if no_images else True
        self.image_width = image_width
        self.image_height = image_height


    def save_images_command(self, num_images, output, name_format, jpeg, webp, quality,
                            png, compression, frame_margin, scale, height, width):
        # type: (int, str, str, bool, bool, int, bool, int, float, int, int) -> None
        """ Save Images Command: Parses all options/arguments passed to the save-images command,
        or with respect to the CLI, this function processes [save-images options] when calling:
        scenedetect [global options] save-images [save-images options] [other commands...].

        Raises:
            click.BadParameter
        """
        self.check_input_open()

        if contains_sequence_or_url(self.video_stream.path):
            self.options_processed = False
            error_str = '\nThe save-images command is incompatible with image sequences/URLs.'
            self.logger.error(error_str)
            raise click.BadParameter(error_str, param_hint='save-images')

        num_flags = sum([1 if flag else 0 for flag in [jpeg, webp, png]])
        if num_flags <= 1:

            # Ensure the format exists.
            extension = 'jpg'   # Default is jpg.
            if png:
                extension = 'png'
            elif webp:
                extension = 'webp'
            valid_params = get_cv2_imwrite_params()
            if not extension in valid_params or valid_params[extension] is None:
                error_strs = [
                    'Image encoder type %s not supported.' % extension.upper(),
                    'The specified encoder type could not be found in the current OpenCV module.',
                    'To enable this output format, please update the installed version of OpenCV.',
                    'If you build OpenCV, ensure the the proper dependencies are enabled. ']
                self.logger.debug('\n'.join(error_strs))
                raise click.BadParameter('\n'.join(error_strs), param_hint='save-images')

            self.save_images = True
            self.image_directory = output
            self.image_extension = extension
            self.image_param = compression if png else quality
            self.image_name_format = name_format
            self.num_images = num_images
            self.frame_margin = frame_margin
            self.scale = scale
            self.height = height
            self.width = width

            image_type = 'JPEG' if self.image_extension == 'jpg' else self.image_extension.upper()
            image_param_type = ''
            if self.image_param:
                image_param_type = 'Compression' if image_type == 'PNG' else 'Quality'
                image_param_type = ' [%s: %d]' % (image_param_type, self.image_param)
            self.logger.info('Image output format set: %s%s', image_type, image_param_type)
            if self.image_directory is not None:
                self.logger.info('Image output directory set:\n  %s',
                             os.path.abspath(self.image_directory))
        else:
            self.options_processed = False
            self.logger.error('Multiple image type flags set for save-images command.')
            raise click.BadParameter(
                'Only one image type (JPG/PNG/WEBP) can be specified.', param_hint='save-images')
