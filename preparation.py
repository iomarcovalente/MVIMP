import shutil
from mvimp_utils.location import *
import argparse

# version 0907
external_links = {
    # DAIN
    "dain-best-model": "https://www.dropbox.com/s/yw7qw5ygrvixinc/0907-best.pth",
}

def downloader(link: str, name: str) -> None:
    os.system(f"wget {link} -O {name}")

def config():
    parser = argparse.ArgumentParser(description="MVIMP configuration.")
    parser.add_argument(
        "--function",
        "-f",
        type=str,
        help="Function or functions your wanna prepare.",
    )
    return parser.parse_args()

def dain_preparation():
    os.chdir(DAIN_PREFIX)
    print("Downloading DAIN model weights...")
    my_package_dir = os.path.join(DAIN_PREFIX, "my_package")
    nvidia_pwcnet_dir = os.path.join(
        DAIN_PREFIX, "PWCNet/correlation_package_pytorch1_0"
    )
    model_weights_dir = os.path.join(DAIN_PREFIX, "model_weights")

    os.chdir(my_package_dir)
    os.system(f"sh {os.path.join(my_package_dir, 'build.sh')}")
    os.chdir(nvidia_pwcnet_dir)
    os.system(f"sh {os.path.join(nvidia_pwcnet_dir, 'build.sh')}")

    os.makedirs(model_weights_dir)
    os.chdir(model_weights_dir)
    downloader(external_links["dain-best-model"], "best.pth")
    print("Done.")
