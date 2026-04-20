import importlib
import sys


REQUIRED_MODULES = [
    "jupyter",
    "numpy",
    "pandas",
    "matplotlib",
    "sklearn",
]


def main() -> None:
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")

    if version < (3, 9):
        raise SystemExit("Python >= 3.9 is required")

    for module_name in REQUIRED_MODULES:
        importlib.import_module(module_name)
        print(f"{module_name}: OK")


if __name__ == "__main__":
    main()
