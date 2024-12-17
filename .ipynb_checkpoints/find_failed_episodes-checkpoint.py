import os

def find_failed_episodes(root_dir, prefix):
    """
    Loops through files in the given directory and identifies episodes marked as failed,
    limited to those starting with the given prefix.

    Args:
        root_dir (str): The root directory to search for files.
        prefix (str): The prefix to filter files (e.g., "2024_12_10-05_00_47").

    Returns:
        list: A list of failed episode filenames in the format ["episode_N.pt", ...].
    """
    failed_episodes = []

    # Walk through the directory
    for foldername, subfolders, filenames in os.walk(root_dir):
        for filename in filenames:
            # Check if the filename starts with the specified prefix and has a failure marker
            if filename.startswith(prefix) and "--success=False" in filename:
                # Extract the episode number from the filename
                try:
                    episode_start = filename.find("--episode=") + len("--episode=")
                    episode_end = filename.find("--", episode_start)
                    episode_number = filename[episode_start:episode_end]
                    failed_episodes.append(f"episode_{episode_number}.pt")
                except ValueError:
                    print(f"Error parsing episode number from: {filename}")
    
    return failed_episodes

# Example usage
if __name__ == "__main__":
    root_directory = "/root/openvla/rollouts/2024_12_10"  # Change to your root directory
    prefix = "2024_12_10-07_18_36"  # Change to your desired prefix

    # Find failed episodes
    failed_files = find_failed_episodes(root_directory, prefix)

    # Output the results
    print(f"\nFailed episodes: {failed_files}")
