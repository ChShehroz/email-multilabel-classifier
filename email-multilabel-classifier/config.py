from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

DATA_FILES = [
    DATA_DIR / "emails.csv",
    DATA_DIR / "AppGallery.csv",
    DATA_DIR / "Purchasing.csv",
]

TEXT_CANDIDATES = ["EmailText", "Interaction content", "Ticket Summary"]
TYPE1_COLUMN = "Type 1"
TYPE2_COLUMN = "Type 2"
TYPE3_COLUMN = "Type 3"
TYPE4_COLUMN = "Type 4"

TEXT_COLUMN = "EmailText"

TEST_SIZE = 0.2
RANDOM_STATE = 42
MIN_CLASS_COUNT = 2
MODEL_RANDOM_STATE = 42
N_ESTIMATORS = 200
