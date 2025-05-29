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

# (c) 2024 by Chris Paxton

from .base import ImageGenerator
from .diffuser import DiffuserImageGenerator
from .flux import FluxImageGenerator


def create_image_generator(generator: str, **kwargs) -> ImageGenerator:
    """
    Create an image generator based on the given generator name and parameters.

    Args:
        generator (str): The name of the image generator to create.
        **kwargs: Additional keyword arguments to pass to the image generator constructor.

    Returns:
        ImageGenerator: The created image generator.
    """
    if generator == "diffuser":
        return DiffuserImageGenerator(**kwargs)
    else:
        raise ValueError(f"Unknown image generator: {generator}")


__all__ = [
    ImageGenerator,
    DiffuserImageGenerator,
    FluxImageGenerator,
    create_image_generator,
]
