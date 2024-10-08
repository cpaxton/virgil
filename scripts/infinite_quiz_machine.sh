#!/bin/bash

# Exit immediately if any command exits with a non-zero status
set -e

# Check if an argument is provided
if [ $# -ne 1 ]; then
    echo "Error: Please provide exactly one argument."
    exit 1
fi

# Store the argument in a variable
prompt="$1"

# Run the first Python command
echo "Running quiz maker..."
python -m virgil.quiz.quiz_maker --topic "$prompt"

# Run the second Python command
echo "Creating images..."
python -m virgil.quiz.create_images --quiz_path "$prompt"

echo "Creating HTML..."
python -m virgil.quiz.create_html --quiz_path "$prompt"

echo "Script completed successfully."

