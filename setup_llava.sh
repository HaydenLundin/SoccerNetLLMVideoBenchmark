#!/bin/bash
# Setup script to copy LLaVA pipeline files to soccer_project directory

REPO_DIR="/home/user/SoccerNetLLMVideoBenchmark"
PROJECT_DIR="$HOME/soccer_project"

echo "📦 Setting up LLaVA pipeline files..."
echo ""

# Check if project directory exists
if [ ! -d "$PROJECT_DIR" ]; then
    echo "❌ Error: $PROJECT_DIR does not exist"
    echo "   Create it first: mkdir -p $PROJECT_DIR"
    exit 1
fi

# Copy LLaVA worker script
echo "Copying llava_split_worker.py..."
cp "$REPO_DIR/llava_split_worker.py" "$PROJECT_DIR/"

# Copy LLaVA merge script
echo "Copying merge_results_llava.py..."
cp "$REPO_DIR/merge_results_llava.py" "$PROJECT_DIR/"

# Make executable
chmod +x "$PROJECT_DIR/llava_split_worker.py"
chmod +x "$PROJECT_DIR/merge_results_llava.py"

echo ""
echo "✅ Setup complete!"
echo ""
echo "Files copied to $PROJECT_DIR:"
ls -lh "$PROJECT_DIR/llava_split_worker.py"
ls -lh "$PROJECT_DIR/merge_results_llava.py"
echo ""
echo "Now you can run: sbatch submit_llava.sh 0 39"
