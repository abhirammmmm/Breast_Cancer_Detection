import os

def print_directory_structure(path, indent=0, exclude_dirs=None):
    if exclude_dirs is None:
        exclude_dirs = []

    # List all files and directories in the current directory
    for item in os.listdir(path):
        # Skip excluded directories
        if item in exclude_dirs:
            continue

        # Create the full path
        full_path = os.path.join(path, item)
        # Print the item with indentation
        print(' ' * indent + '|-- ' + item)
        # If it's a directory, call the function recursively
        if os.path.isdir(full_path):
            print_directory_structure(full_path, indent + 4, exclude_dirs)

if __name__ == '__main__':
    # Specify the root directory of your project
    root_directory = r'/Users/santhosh/Documents/GitHub/Breast_Cancer_Detection'  # Using raw string
    exclude_directories = ['venv', '.git']  # Add any other directories you want to exclude here
    print(f"Project Structure of: {root_directory}\n")
    print_directory_structure(root_directory, exclude_dirs=exclude_directories)
