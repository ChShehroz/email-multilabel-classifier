from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import config


def _first_available_text(row):
    for col in config.TEXT_CANDIDATES:
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            return str(row[col]).strip()
    return ""


def _normalize_columns(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # rename old column names to new common names
    rename_map = {
        "Type2": "Type 2",
        "Type3": "Type 3",
        "Type4": "Type 4",
        "EmailText": "EmailText",
        "Interaction content": "Interaction content",
        "Ticket Summary": "Ticket Summary",
    }
    df = df.rename(columns=rename_map)

    required = [config.TYPE2_COLUMN, config.TYPE3_COLUMN, config.TYPE4_COLUMN]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}. Found columns: {list(df.columns)}")

    df[config.TEXT_COLUMN] = df.apply(_first_available_text, axis=1)

    keep_cols = [config.TEXT_COLUMN]
    if config.TYPE1_COLUMN in df.columns:
        keep_cols.append(config.TYPE1_COLUMN)
    keep_cols.extend(required)

    df = df[keep_cols].copy()
    df[config.TEXT_COLUMN] = df[config.TEXT_COLUMN].fillna("").astype(str).str.strip()

    for col in [config.TYPE1_COLUMN, *required]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str).str.strip()

    return df


def load_data():
    frames = []
    loaded_files = []
    for path in config.DATA_FILES:
        path = Path(path)
        if path.exists():
            df = pd.read_csv(path)
            df = _normalize_columns(df)
            df["source_file"] = path.name
            frames.append(df)
            loaded_files.append(path.name)

    if not frames:
        raise FileNotFoundError("No dataset file found in data folder.")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined[config.TEXT_COLUMN].str.len() > 0].copy()
    return combined, loaded_files


def remove_rare_classes(df):
    df = df.copy()
    removed = {}
    for col in [config.TYPE2_COLUMN, config.TYPE3_COLUMN, config.TYPE4_COLUMN]:
        counts = df[col].value_counts()
        valid_classes = counts[counts >= config.MIN_CLASS_COUNT].index
        removed[col] = sorted(counts[counts < config.MIN_CLASS_COUNT].index.tolist())
        df = df[df[col].isin(valid_classes)].copy()
    df = df.reset_index(drop=True)
    return df, removed


def preprocess_data(df):
    df = df.dropna(subset=[config.TEXT_COLUMN, config.TYPE2_COLUMN, config.TYPE3_COLUMN, config.TYPE4_COLUMN]).copy()
    df, removed = remove_rare_classes(df)

    if config.TYPE1_COLUMN in df.columns:
        df[config.TYPE1_COLUMN] = df[config.TYPE1_COLUMN].fillna("Unknown").astype(str).str.strip()

    X = df[config.TEXT_COLUMN]
    y2 = df[config.TYPE2_COLUMN]
    y3 = df[config.TYPE3_COLUMN]
    y4 = df[config.TYPE4_COLUMN]

    type1_values = []
    if config.TYPE1_COLUMN in df.columns:
        type1_values = sorted(df[config.TYPE1_COLUMN].unique().tolist())

    metadata = {
        "row_count": len(df),
        "removed_rare_classes": removed,
        "type1_unique": type1_values
    }

    return X, y2, y3, y4, metadata


def split_data(X, y2, y3, y4):
    return train_test_split(
        X, y2, y3, y4,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y2
    )