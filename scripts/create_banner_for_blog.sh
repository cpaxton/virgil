#!/usr/bin/env bash
#
prompt1="A vibrant solarpunk city, growing green on buildings, painterly style, sunny day, robots and humans, robots holding hands with human, flowers, hanging gardens, futuristic, solar panels, wind turbines, harmonious coexistence, bright colors, utopian atmosphere" 
prompt2="A vibrant solarpunk city, strange city, painterly style, sunny day, robots and humans, crowded streets and markets, flowers, hanging gardens, futuristic, solar panels, wind turbines, bright colors, utopian atmosphere" 
prompt3="A vibrant and strange solarpunk city, painterly style, sunny day, robots and humans, crowded streets and markets, flowers, hanging gardens, futuristic, wind turbines and airships, bright colors, utopian atmosphere" 
prompt4="A vibrant and strange solarpunk city, painterly style, sunny day, robots and humans, crowded streets and markets filled with technology, hanging gardens off of skyscrapers, wind turbines and airships in the background, bright colors, utopian atmosphere" 
#
python -m virgil.image.flux --height 800 --width 1200 -n 5 \
	--output "solarpunk_banner_v04.png" \
	--prompt "$prompt4"

echo "Done with Flux."

python -m virgil.image.diffuser --height 800 --width 1200 -n 5 \
	--model "turbo" \
	--output "solarpunk_banner_diffuser_v04.png" \
	--prompt "$prompt4"

echo "Done with Diffuser."
