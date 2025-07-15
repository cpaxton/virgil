#!/usr/bin/env bash


prompt1="Detailed whole-body illustration of a clockwork knight with a flintlock pistol and a flaming sword, wearing a brass helmet shaped like an owl. Flasks, potions at hip. Blank background, for a game sprite. Single character, exactly centered."

# python -m virgil.image.flux --height 800 --width 1200 -n 5 \
python -m virgil.image.flux --height 512 --width 512 -n 5 \
	--output "chartest_flux_v03.png" \
	--prompt "$prompt1"

echo "Done with Flux."

python -m virgil.image.diffuser --height 512 --width 512 -n 5 \
	--model "turbo" \
	--output "chartest_diffusion_v03.png" \
	--prompt "$prompt1"

echo "Done with Diffuser."
