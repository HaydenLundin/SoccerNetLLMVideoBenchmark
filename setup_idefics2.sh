#!/bin/bash
# Setup script to copy Idefics2 pipeline files to soccer_project directory

REPO_DIR="/home/user/SoccerNetLLMVideoBenchmark"
PROJECT_DIR="$HOME/soccer_project"

echo "📦 Setting up Idefics2 pipeline files..."
echo ""

# Check if project directory exists
if [ ! -d "$PROJECT_DIR" ]; then
    echo "❌ Error: $PROJECT_DIR does not exist"
    echo "   Create it first: mkdir -p $PROJECT_DIR"
    exit 1
fi

# Copy Idefics2 worker script
echo "Copying idefics2_split_worker.py..."
cp "$REPO_DIR/idefics2_split_worker.py" "$PROJECT_DIR/"

# Copy Idefics2 merge script
echo "Copying merge_results_idefics2.py..."
cp "$REPO_DIR/merge_results_idefics2.py" "$PROJECT_DIR/"

# Make executable
chmod +x "$PROJECT_DIR/idefics2_split_worker.py"
chmod +x "$PROJECT_DIR/merge_results_idefics2.py"

echo ""
echo "✅ Setup complete!"
echo ""
echo "Files copied to $PROJECT_DIR:"
ls -lh "$PROJECT_DIR/idefics2_split_worker.py"
ls -lh "$PROJECT_DIR/merge_results_idefics2.py"
echo ""
echo "Now you can run: sbatch submit_idefics2.sh 0 39"
