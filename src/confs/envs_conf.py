from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EnvsConf:
    # Data Directory
    data_dir = Path(__file__).parent.parent.parent / "data"

    dev_data_dir = data_dir / "dev"

    train_data_dir = data_dir / "train"

    test_data_dir = data_dir / "test"


impl = EnvsConf()
