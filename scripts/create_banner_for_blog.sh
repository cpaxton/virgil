#!/usr/bin/env bash
#
prompt1="A vibrant solarpunk city, growing green on buildings, painterly style, sunny day, robots and humans, robots holding hands with human, flowers, hanging gardens, futuristic, solar panels, wind turbines, harmonious coexistence, bright colors, utopian atmosphere" 
prompt2="A vibrant solarpunk city, strange city, painterly style, sunny day, robots and humans, crowded streets and markets, flowers, hanging gardens, futuristic, solar panels, wind turbines, bright colors, utopian atmosphere" 
prompt3="A vibrant and strange solarpunk city, painterly style, sunny day, robots and humans, crowded streets and markets, flowers, hanging gardens, futuristic, wind turbines and airships, bright colors, utopian atmosphere" 
prompt4="A vibrant and strange solarpunk city, painterly style, robots and humans, crowded streets and markets filled with robotic technology, hanging gardens off of skyscrapers tended by scuttling robot insects, wind turbines and airships in the background, bright colors, utopian atmosphere" 
#
# python -m virgil.image.flux --height 800 --width 1200 -n 5 \
python -m virgil.image.flux --height 512 --width 512 -n 5 \
	--output "solarpunk_banner_v06.png" \
	--prompt "$prompt4"

echo "Done with Flux."

python -m virgil.image.diffuser --height 512 --width 512 -n 5 \
	--model "turbo" \
	--output "solarpunk_banner_diffuser_v06.png" \
	--prompt "$prompt4"

echo "Done with Diffuser."
