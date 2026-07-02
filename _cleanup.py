import os
import shutil


def cleanup_checkpoints(root_directory, dry_run=True):
    for root, dirs, files in os.walk(root_directory, topdown=False):
        for name in dirs:
            if name == ".ipynb_checkpoints":
                full_path = os.path.join(root, name)
                if dry_run:
                    print(f"[DRY RUN] Would delete: {full_path}")
                else:
                    try:
                        shutil.rmtree(full_path)
                        print(f"Deleted: {full_path}")
                    except Exception as e:
                        print(f"Error deleting {full_path}: {e}")

target_path = "./"
cleanup_checkpoints(target_path, dry_run=False)
