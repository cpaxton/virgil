#!/usr/bin/env bash
python -m virgil.image.diffuser --height=512 --width=512 --model=turbo --prompt "A many-armed robot performing a variety of tasks; many limbs with different tools; its head has many different cameras and other sensors. many of its arms end in humanlike hands. bright, colorful, sci fi art" --output "vla_v02.png" --num-images 10
