#!/bin/bash

echo "Finding unused dependencies..."
python scripts/clean_dependencies.py > unused_deps.txt

if [ -s unused_deps.txt ]; then
    echo "Removing unused dependencies..."
    while IFS= read -r cmd; do
        echo "Executing: $cmd"
        $cmd
    done < unused_deps.txt
else
    echo "No unused dependencies found!"
fi

rm unused_deps.txt
echo "Cleanup completed!"
