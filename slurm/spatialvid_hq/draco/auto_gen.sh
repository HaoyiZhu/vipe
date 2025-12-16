#!/bin/bash

# Define the range of group IDs you want to generate (0009 to 0072)
START_ID=2
END_ID=72

# Loop through the desired IDs
for i in $(seq $START_ID $END_ID); do
    # Format the ID with leading zeros (e.g., 9 -> 0009, 72 -> 0072)
    # The %04d format ensures four digits, padded with zeros.
    GROUP_ID=$(printf "%04d" $i)

    # Define the new filename
    NEW_FILE="group_${GROUP_ID}.sh"

    # 1. Copy the source file to the new file
    cp group_0001.sh "$NEW_FILE"

    # 2. Use 'sed' to replace all occurrences of 'group_0001' with the new group ID.
    # The -i flag performs the replacement in-place on the new file.
    # We use a non-standard delimiter (e.g., @) to avoid conflict with path slashes.
    sed -i "s@group_0001@group_${GROUP_ID}@g" "$NEW_FILE"

    echo "Created and updated $NEW_FILE"
done

echo "Script generation complete. Total files created: $((END_ID - START_ID + 1))"
