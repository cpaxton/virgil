from virgil.image.siglip import SigLIPAligner
from virgil.image.diffuser import DiffuserImageGenerator
import click


@click.command()
@click.option(
    "--use-aligner",
    is_flag=True,
    help="Use SigLIP aligner for enhanced image generation",
)
def main(use_aligner: bool = False):
    generator = DiffuserImageGenerator(model="playground")
    aligner = SigLIPAligner()

    # Base generation
    prompt = "A futuristic cityscape at sunset, neon reflections on wet streets"
    basic_image = generator.generate(prompt)
    basic_image.save("playground_cityscape_basic.png")

    if use_aligner:
        # Use aligner for enhanced generation
        print("Using SigLIP aligner for enhanced image generation...")
        # Quality-enhanced generation (using alignment)
        enhanced_prompt = (
            prompt + ", masterpiece, 8k resolution, unreal engine, cinematic lighting"
        )
        score, best_image = aligner.search(
            generator,
            enhanced_prompt,
            num_tries=5,
            generation_params={
                "num_inference_steps": 25,
                "guidance_scale": 3.5,  # Slightly higher for detail
            },
        )
        best_image.save("playground_cityscape.png")


if __name__ == "__main__":
    main()
