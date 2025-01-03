#!/bin/bash

# Define the list of configurations for each parameter
AGENT1_MODELS=("gpt-4o")
AGENT2_MODELS=("gpt-4o")
AGENT1_PROMPTS=("original_vs_harmonized_debate") #("self_play_debate" "basic" "debate" "self_reflection" "self_consistency")
AGENT2_PROMPTS=("basic")
LOOK_AHEAD_VALUES=(3)
NUM_GAMES_VALUES=(50)

# Output directory for logs
OUTPUT_DIR="/home/jihwan/NashIP/result/FN20"
mkdir -p "$OUTPUT_DIR"

# Iterate over all combinations of parameters
for agent1_model in "${AGENT1_MODELS[@]}"; do
    for agent2_model in "${AGENT2_MODELS[@]}"; do
        for agent1_prompt in "${AGENT1_PROMPTS[@]}"; do
            for agent2_prompt in "${AGENT2_PROMPTS[@]}"; do
                for look_ahead in "${LOOK_AHEAD_VALUES[@]}"; do
                    for num_games in "${NUM_GAMES_VALUES[@]}"; do
                        # Print the current configuration
                        echo "Running experiment with the following configuration:"
                        echo "Agent 1 Model: $agent1_model"
                        echo "Agent 2 Model: $agent2_model"
                        echo "Agent 1 Prompting Method: $agent1_prompt"
                        echo "Agent 2 Prompting Method: $agent2_prompt"
                        echo "Look Ahead Steps: $look_ahead"
                        echo "Number of Games: $num_games"

                        # Construct the output log file path
                        LOG_FILE="$OUTPUT_DIR/${agent1_model}_${agent1_prompt}_${look_ahead}_${agent2_model}_${agent2_prompt}.txt"

                        # Run the Python script with the current configuration
                        python3 /home/jihwan/NashIP/fibonacci_first.py \
                            --agent1_model "$agent1_model" \
                            --agent2_model "$agent2_model" \
                            --agent1_prompt "$agent1_prompt" \
                            --agent2_prompt "$agent2_prompt" \
                            --look_ahead "$look_ahead" \
                            --num_games "$num_games" \
                            >> "$LOG_FILE" 2>&1 &
                        
                        # Optionally, you can limit the number of concurrent experiments
                        # Uncomment the following line to run up to 4 experiments at a time
                        # wait -n
                    done
                done
            done
        done
    done
done

echo "All experiments have been started."