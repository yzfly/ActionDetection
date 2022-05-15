import io
import math
import logging
import pathlib

from iopath.common.file_io import g_pathmgr
from decord import VideoReader
from decord import cpu, gpu

logger = logging.getLogger(__name__)

class EncodedVideo():
    def __init__(self, file_path, gpu_id=None, **other_args):

        with g_pathmgr.open(file_path, "rb") as fh:
            video_file = io.BytesIO(fh.read())

        self.video_name=pathlib.Path(file_path).name
        if gpu_id:
            self.video_reader = VideoReader(video_file, ctx=gpu(gpu_id))
        else:
            self.video_reader = VideoReader(video_file, ctx=cpu(0))

        self.fps = self.video_reader.get_avg_fps()
        self.duration = float(len(self.video_reader)) / float(self.fps)
        self.height, self.width, _ = self.video_reader[0].shape

    def get_clip(self, start_sec: float, end_sec: float):
        """
        Retrieves frames from the encoded video at the specified start and end times
        in seconds (the video always starts at 0 seconds).

        Args:
            start_sec (float): the clip start time in seconds
            end_sec (float): the clip end time in seconds
        Returns:
            clip_data:
                A dictionary mapping the entries at "video" to a numpy matrix.

                "video": A tensor of the clip's RGB frames with shape:
                (time, height, width, channel). The frames are of type numpy.float32 and
                in the range [0 - 255].

            Returns None if no video or audio found within time range.
        """

        if start_sec > end_sec or start_sec > self.duration:
            raise RuntimeError(
                f"Incorrect time window for Decord decoding for video: {self.video_name}."
            )

        start_idx = math.ceil(self.fps * start_sec)
        end_idx = math.ceil(self.fps * end_sec)
        end_idx = min(end_idx, len(self.video_reader))
        frame_idxs = list(range(start_idx, end_idx))


        try:
            video = self.video_reader.get_batch(frame_idxs).asnumpy()
        except Exception as e:
            logger.debug(f"Failed to decode video with Decord: {self.video_name}. {e}")
            raise e

        return video  # thwc ,rgb