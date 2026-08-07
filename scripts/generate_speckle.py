from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from specklegenerator.generator import DatasetGenerator


def main() -> None:

    generator = DatasetGenerator()
    generator.generate()


if __name__ == "__main__":
    main()