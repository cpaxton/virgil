#!/usr/bin/env bash
#
prompt1="A vibrant solarpunk city, growing green on buildings, painterly style, sunny day, robots and humans, robots holding hands with human, flowers, hanging gardens, futuristic, solar panels, wind turbines, harmonious coexistence, bright colors, utopian atmosphere" 
prompt2="A vibrant solarpunk city, growing green on buildings, strange city, painterly style, sunny day, robots and humans, robots holding hands with human, flowers, hanging gardens, futuristic, solar panels, wind turbines, harmonious coexistence, bright colors, utopian atmosphere" 
#
python -m virgil.image.flux --height 800 --width 1200 -n 10 \
	--output "solarpunk_banner_v03.png" \
	--prompt "$prompt2"

echo "Done with Flux."

python -m virgil.image.diffuser --height 800 --width 1200 -n 10 \
	--model "turbo" \
	--output "solarpunk_banner_diffuser_v03.png" \
	--prompt "$prompt2"

echo "Done with Diffuser."
