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
    generator = DiffuserImageGenerator(model="playground", height=1024, width=1024)
    aligner = SigLIPAligner()

    # Base generation
    # prompt = "A futuristic cityscape at sunset, neon reflections on wet streets"
    # prompt = "A futuristic cityscape, baroque and renaissance architecture, vibrant colors, intricate details"
    # prompt = "A futuristic cityscape, dark, inspired by istanbul, inspired by Beksiński"
    # tag = "futuristic_cityscape_dark_istanbul_beksinski"
    prompt = "the sorceress in her workshop, cybernetic implants, baroque, jewelry and designs, intricate details, black and gold, glowing blue highlights"
    tag = "sorceress2"

    enhanced_prompt = (
        # prompt + ", masterpiece, 8k resolution, unreal engine, cinematic lighting"
        prompt + ", masterpiece, 8k resolution, oil painting, dramatic lighting"
    )

    for i in range(10):
        basic_image = generator.generate(
            prompt, num_inference_steps=25, guidance_scale=3.5
        )
        basic_image.save(f"{tag}_basic_{i}.png")

        enhanced_image = generator.generate(
            enhanced_prompt, num_inference_steps=25, guidance_scale=3.5
        )
        enhanced_image.save(f"{tag}_enhanced_{i}.png")

    if use_aligner:
        # Use aligner for enhanced generation
        print("Using SigLIP aligner for enhanced image generation...")
        # Quality-enhanced generation (using alignment)
        score, best_image = aligner.search(
            generator,
            enhanced_prompt,
            num_tries=5,
            generation_params={
                "num_inference_steps": 25,
                "guidance_scale": 3.5,  # Slightly higher for detail
            },
        )
        best_image.save(f"{tag}.png")


if __name__ == "__main__":
    main()
