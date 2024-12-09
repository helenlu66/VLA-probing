import os

def find_failed_episodes(root_dir):
    """
    Loops through files in the given directory and identifies episodes marked as failed.

    Args:
        root_dir (str): The root directory to search for files.

    Returns:
        list: A list of failed episode filenames in the format ["episode_N.pt", ...].
    """
    failed_episodes = []

    # Walk through the directory
    for foldername, subfolders, filenames in os.walk(root_dir):
        for filename in filenames:
            # Check if the file has a failure marker
            if "--success=False" in filename:
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

    # Find failed episodes
    failed_files = find_failed_episodes(root_directory)

    # Output the results
    print(f"\nFailed episodes: {failed_files}")
