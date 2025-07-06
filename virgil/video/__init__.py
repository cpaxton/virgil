# Copyright 2024 Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .base import VideoBackend
from .ltx_video import LTXVideo
from .framepack import Framepack
from .wan_2_1 import Wan21
from .hunyuan_video_2 import HunyuanVideo2

video_backend_list = [
    "ltx-video",
    "framepack",
    "wan-2.1",
    "hunyuan-video-2",
]


def get_video_backend(name: str, **kwargs) -> VideoBackend:
    """
    Get a video backend by name.

    Args:
        name (str): The name of the video backend.

    Returns:
        VideoBackend: The video backend instance.
    """
    name = name.lower()
    if name == "ltx-video":
        return LTXVideo(**kwargs)
    elif name == "framepack":
        return Framepack(**kwargs)
    elif name == "wan-2.1":
        return Wan21(**kwargs)
    elif name == "hunyuan-video-2":
        return HunyuanVideo2(**kwargs)
    else:
        raise ValueError(f"Unknown video backend: {name}")
