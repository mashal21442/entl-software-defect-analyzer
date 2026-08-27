from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    family: str
    name: str
    relative_path: str
    expected_entries: int
    expected_bugs: int
    expected_metrics: int
    paper_bug_percent: float
    label_candidates: tuple[str, ...]
    drop_if_present: tuple[str, ...] = ()


DATASETS = {
    # AEEM
    "AEEM/EQ": DatasetSpec(
        "AEEM/EQ", "AEEM", "EQ", "data/raw/AEEM/EQ.csv",
        324, 129, 61, 39.8,
        ("bug", "bugs", "defect", "defects", "class", "label", "buggy")
    ),
    "AEEM/JDT": DatasetSpec(
        "AEEM/JDT", "AEEM", "JDT", "data/raw/AEEM/JDT.csv",
        997, 206, 61, 20.7,
        ("bug", "bugs", "defect", "defects", "class", "label", "buggy")
    ),
    "AEEM/LC": DatasetSpec(
        "AEEM/LC", "AEEM", "LC", "data/raw/AEEM/LC.csv",
        691, 64, 61, 9.3,
        ("bug", "bugs", "defect", "defects", "class", "label", "buggy")
    ),
    "AEEM/ML": DatasetSpec(
        "AEEM/ML", "AEEM", "ML", "data/raw/AEEM/ML.csv",
        1862, 245, 61, 13.2,
        ("bug", "bugs", "defect", "defects", "class", "label", "buggy")
    ),

    # JIRA
    "JIRA/activemq5.0.0": DatasetSpec(
        "JIRA/activemq5.0.0", "JIRA", "activemq5.0.0",
        "data/raw/JIRA/activemq5.0.0.csv",
        1884, 293, 65, 15.6,
        ("RealBug", "realbug", "bug", "bugs", "defect", "defects", "label"),
        ("File", "RealBugCount", "HeuBug", "HeuBugCount")
    ),
    "JIRA/Derby10.5.1.1": DatasetSpec(
        "JIRA/Derby10.5.1.1", "JIRA", "Derby10.5.1.1",
        "data/raw/JIRA/Derby10.5.1.1.csv",
        2705, 383, 65, 14.2,
        ("RealBug", "realbug", "bug", "bugs", "defect", "defects", "label"),
        ("File", "RealBugCount", "HeuBug", "HeuBugCount")
    ),
    "JIRA/Hbase0.94.0": DatasetSpec(
        "JIRA/Hbase0.94.0", "JIRA", "Hbase0.94.0",
        "data/raw/JIRA/Hbase0.94.0.csv",
        1059, 218, 65, 2.6,
        ("RealBug", "realbug", "bug", "bugs", "defect", "defects", "label"),
        ("File", "RealBugCount", "HeuBug", "HeuBugCount")
    ),
    "JIRA/Hive0.9.0": DatasetSpec(
        "JIRA/Hive0.9.0", "JIRA", "Hive0.9.0",
        "data/raw/JIRA/Hive0.9.0.csv",
        1416, 283, 65, 20.0,
        ("RealBug", "realbug", "bug", "bugs", "defect", "defects", "label"),
        ("File", "RealBugCount", "HeuBug", "HeuBugCount")
    ),

    # NASA
    "NASA/KC1": DatasetSpec(
        "NASA/KC1", "NASA", "KC1", "data/raw/NASA/KC1.csv",
        2095, 325, 21, 15.5,
        ("defects", "defect", "bug", "bugs", "class", "label")
    ),
    "NASA/PC1": DatasetSpec(
        "NASA/PC1", "NASA", "PC1", "data/raw/NASA/PC1.csv",
        735, 61, 37, 8.3,
        ("defects", "defect", "bug", "bugs", "class", "label")
    ),
    "NASA/PC3": DatasetSpec(
        "NASA/PC3", "NASA", "PC3", "data/raw/NASA/PC3.csv",
        1099, 138, 37, 12.6,
        ("defects", "defect", "bug", "bugs", "class", "label")
    ),
    "NASA/PC4": DatasetSpec(
        "NASA/PC4", "NASA", "PC4", "data/raw/NASA/PC4.csv",
        1379, 178, 37, 12.9,
        ("defects", "defect", "bug", "bugs", "class", "label")
    ),

    # PROMISE
        # PROMISE
    "PROMISE/Lucene2.4": DatasetSpec(
        "PROMISE/Lucene2.4", "PROMISE", "Lucene2.4",
        "data/raw/PROMISE/Lucene2.4.csv",
        340, 203, 20, 59.7,
        ("bug", "bugs", "defect", "defects", "class", "label"),
        ("name", "version", "name.1")
    ),
    "PROMISE/Poi3.0": DatasetSpec(
        "PROMISE/Poi3.0", "PROMISE", "Poi3.0",
        "data/raw/PROMISE/Poi3.0.csv",
        442, 281, 20, 63.6,
        ("bug", "bugs", "defect", "defects", "class", "label"),
        ("name", "version", "name.1")
    ),
    "PROMISE/Synapse1.2": DatasetSpec(
        "PROMISE/Synapse1.2", "PROMISE", "Synapse1.2",
        "data/raw/PROMISE/Synapse1.2.csv",
        256, 86, 20, 33.6,
        ("bug", "bugs", "defect", "defects", "class", "label"),
        ("name", "version", "name.1")
    ),
    "PROMISE/Velocity1.6": DatasetSpec(
        "PROMISE/Velocity1.6", "PROMISE", "Velocity1.6",
        "data/raw/PROMISE/Velocity1.6.csv",
        229, 78, 20, 34.1,
        ("bug", "bugs", "defect", "defects", "class", "label"),
        ("name", "version", "name.1")
    ),}
def get_spec(key: str) -> DatasetSpec:
    try:
        return DATASETS[key]
    except KeyError as exc:
        raise KeyError(
            f"Unknown dataset key {key!r}. Valid keys: {sorted(DATASETS)}"
        ) from exc


def raw_path(project_root: Path, key: str) -> Path:
    return project_root / get_spec(key).relative_path