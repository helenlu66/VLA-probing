# import os
# import zipfile

# def zip_episodes(root_dir, output_zip):
#     """
#     Zips all files matching the pattern episode_N.pt in the directory tree starting at root_dir.

#     Args:
#         root_dir (str): The root directory to search for episode_N.pt files.
#         output_zip (str): The path to the output ZIP file.
#     """
#     with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
#         for foldername, subfolders, filenames in os.walk(root_dir):
#             for filename in filenames:
#                 if filename.startswith("episode_") and filename.endswith(".pt"):
#                     file_path = os.path.join(foldername, filename)
#                     arcname = os.path.relpath(file_path, start=root_dir)  # Keep relative paths in ZIP
#                     zipf.write(file_path, arcname)
#                     print(f"Added {file_path} as {arcname} to the zip.")
#     print(f"All episode_N.pt files zipped into {output_zip}")

# # Example usage
# if __name__ == "__main__":
#     root_directory = "/root/openvla/experiments/logs"  # Change to your root directory
#     output_zip_file = "episodes.zip"  # Change to your desired output ZIP file name

#     zip_episodes(root_directory, output_zip_file)


import os
import zipfile

def zip_episodes(root_dir, output_zip):
    """
    Zips all files matching the pattern episode_N.pt (where 1 <= N <= 100) 
    in the directory tree starting at root_dir.

    Args:
        root_dir (str): The root directory to search for episode_N.pt files.
        output_zip (str): The path to the output ZIP file.
    """
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for foldername, subfolders, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.startswith("episode_") and filename.endswith(".pt"):
                    try:
                        # Extract the episode number N from the filename
                        episode_number = int(filename[8:-3])  # Strip "episode_" and ".pt"
                        if 1 <= episode_number <= 100:  # Check if N is within the desired range
                            file_path = os.path.join(foldername, filename)
                            arcname = os.path.relpath(file_path, start=root_dir)  # Keep relative paths in ZIP
                            zipf.write(file_path, arcname)
                            print(f"Added {file_path} as {arcname} to the zip.")
                    except ValueError:
                        # Skip files where the number extraction fails
                        print(f"Skipped invalid file: {filename}")
    print(f"All episode_N.pt files (1 <= N <= 100) zipped into {output_zip}")

# Example usage
if __name__ == "__main__":
    root_directory = "/root/autodl-tmp/logs_60_episodes_of_libero10"  # Change to your root directory
    output_zip_file = "episodes1_60_libero10.zip"  # Change to your desired output ZIP file name

    zip_episodes(root_directory, output_zip_file)
