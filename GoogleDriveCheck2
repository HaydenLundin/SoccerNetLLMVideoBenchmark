drive_dir = "/content/drive/MyDrive/SoccerNet_EPL/batch_1"

if os.path.exists(drive_dir):
    print("Contents of Drive folder:")
    for root, dirs, files in os.walk(drive_dir):
        print(f"\n{root}:")
        for d in dirs:
            print(f"  📁 {d}/")
        for f in files:
            size_mb = os.path.getsize(os.path.join(root, f)) / (1024**2)
            print(f"  📄 {f} - {size_mb:.1f} MB")
else:
    print("Drive folder doesn't exist")
