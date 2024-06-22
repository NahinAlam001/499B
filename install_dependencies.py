import subprocess

def install(package):
    subprocess.check_call(["pip", "install", package])

def main():
    # Install standard packages
    packages = [
        "monai",
        "datasets",
        "scikit-learn",
        "fvcore",
    ]

    for package in packages:
        install(package)

    # Install packages from GitHub
    git_packages = [
        "git+https://github.com/facebookresearch/segment-anything.git",
        "git+https://github.com/huggingface/transformers.git"
    ]

    for git_package in git_packages:
        install(git_package)

    print("All dependencies have been installed successfully!")

if __name__ == "__main__":
    main()