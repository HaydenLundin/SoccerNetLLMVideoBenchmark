#!/bin/bash
# Setup script to copy Phi-3.5-Vision pipeline files to soccer_project directory

REPO_DIR="/home/user/SoccerNetLLMVideoBenchmark"
PROJECT_DIR="$HOME/soccer_project"

echo "📦 Setting up Phi-3.5-Vision pipeline files..."
echo ""

# Check if project directory exists
if [ ! -d "$PROJECT_DIR" ]; then
    echo "❌ Error: $PROJECT_DIR does not exist"
    echo "   Create it first: mkdir -p $PROJECT_DIR"
    exit 1
fi

# Copy Phi-3.5-Vision worker script
echo "Copying phi3_split_worker.py..."
cp "$REPO_DIR/phi3_split_worker.py" "$PROJECT_DIR/"

# Copy Phi-3.5-Vision merge script
echo "Copying merge_results_phi3.py..."
cp "$REPO_DIR/merge_results_phi3.py" "$PROJECT_DIR/"

# Make executable
chmod +x "$PROJECT_DIR/phi3_split_worker.py"
chmod +x "$PROJECT_DIR/merge_results_phi3.py"

echo ""
echo "✅ Setup complete!"
echo ""
echo "Files copied to $PROJECT_DIR:"
ls -lh "$PROJECT_DIR/phi3_split_worker.py"
ls -lh "$PROJECT_DIR/merge_results_phi3.py"
echo ""
echo "Now you can run: sbatch submit_phi3.sh 0 0  # Test on first video"
echo "                 sbatch submit_phi3.sh 0 39  # Run all videos"
