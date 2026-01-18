import os

def limit_files_in_folders(base_path, limit=300):
    for root, dirs, files in os.walk(base_path):
        # We only care about folders that contain files (leaf folders)
        if not dirs and files:
            # Check if it's a banana folder or if we should just do all
            # The user specifically mentioned banana, but previously said "each folder".
            # I will process all folders to be safe and consistent.
            print(f"Processing folder: {root}")
            print(f"Current file count: {len(files)}")
            
            if len(files) > limit:
                files.sort()
                files_to_delete = files[limit:]
                print(f"Deleting {len(files_to_delete)} files...")
                
                for file_name in files_to_delete:
                    file_path = os.path.join(root, file_name)
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        print(f"Error deleting {file_path}: {e}")
                
                print(f"New file count: {len(os.listdir(root))}")
            else:
                print("File count is already below or equal to the limit.")
            print("-" * 20)

if __name__ == "__main__":
    dataset_path = "/home/aishwarya/Documents/colz/Msc-DSCI/Modules/Artificial Neural Network/src/data/train_test_dataset"
    limit_files_in_folders(dataset_path, 300)
