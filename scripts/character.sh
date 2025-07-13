#!/usr/bin/env bash


prompt1="Concept art style whole-body illustration of a clockwork knight with a flintlock pistol and a flaming sword, wearing a helmet shaped like an owl's head. Blank background, for a game sprite."

# python -m virgil.image.flux --height 800 --width 1200 -n 5 \
python -m virgil.image.flux --height 512 --width 512 -n 5 \
	--output "chartest_flux.png" \
	--prompt "$prompt1"

echo "Done with Flux."

python -m virgil.image.diffuser --height 512 --width 512 -n 5 \
	--model "turbo" \
	--output "chartest_diffusion.png" \
	--prompt "$prompt1"

echo "Done with Diffuser."
