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

import click
from virgil.video import get_video_backend


@click.command()
@click.option(
    "--prompt", default="A cat yawning.", help="The prompt to generate the video from."
)
@click.option(
    "--image-path",
    default="virgil/video/examples/cat.jpg",
    help="The path to the initial image.",
)
@click.option("--num-frames", default=16, help="The number of frames to generate.")
@click.option(
    "--num-inference-steps", default=50, help="The number of inference steps to use."
)
@click.option("--guidance-scale", default=7.5, help="The guidance scale to use.")
@click.option(
    "--output-path",
    default="framepack_output.mp4",
    help="The path to save the generated video.",
)
def main(
    prompt, image_path, num_frames, num_inference_steps, guidance_scale, output_path
):
    """
    Generate a video using the Framepack model.
    """
    backend = get_video_backend("framepack")
    backend(
        prompt=prompt,
        image_path=image_path,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        output_path=output_path,
    )
    print(f"Video saved to {output_path}")


if __name__ == "__main__":
    main()
