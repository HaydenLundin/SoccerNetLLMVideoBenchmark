# Check Google Drive
drive_dir = "/content/drive/MyDrive/SoccerNet_EPL/batch_1"

print("\nChecking Google Drive:")
print(f"Directory exists: {os.path.exists(drive_dir)}")

if os.path.exists(drive_dir):
    # List contents
    for root, dirs, files in os.walk(drive_dir):
        level = root.replace(drive_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')

        subindent = ' ' * 2 * (level + 1)
        for file in files[:10]:  # Show first 10 files
            size_mb = os.path.getsize(os.path.join(root, file)) / (1024**2)
            print(f'{subindent}{file} - {size_mb:.1f} MB')
else:
    print("  Directory doesn't exist in Drive!")
