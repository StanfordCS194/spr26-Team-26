from __future__ import annotations

import json
from pathlib import Path

from src.data_generator.mode_a import detect_format, load_raw_data


def test_load_raw_data_csv(tmp_path: Path):
    csv_path = tmp_path / "examples.csv"
    csv_path.write_text("input,output\nhello,world\nfoo,bar\n", encoding="utf-8")

    raw = load_raw_data(str(csv_path))

    assert raw["format_meta"]["file_type"] == "csv"
    assert raw["format_meta"]["modality"] == "tabular"
    assert "Mode A Local Data Acquisition Report" in raw["human_readable"]
    assert raw["records"] == [
        {"input": "hello", "output": "world", "source_path": str(csv_path), "row_index": 1},
        {"input": "foo", "output": "bar", "source_path": str(csv_path), "row_index": 2},
    ]


def test_load_raw_data_tsv(tmp_path: Path):
    tsv_path = tmp_path / "examples.tsv"
    tsv_path.write_text("input\tlabel\nhello\tpositive\n", encoding="utf-8")

    raw = load_raw_data(str(tsv_path))

    assert raw["format_meta"]["file_type"] == "tsv"
    assert raw["records"] == [{"input": "hello", "label": "positive", "source_path": str(tsv_path), "row_index": 1}]


def test_load_raw_data_jsonl(tmp_path: Path):
    jsonl_path = tmp_path / "examples.jsonl"
    jsonl_path.write_text('{"input":"a","output":"b"}\n{"input":"c","output":"d"}\n', encoding="utf-8")

    raw = load_raw_data(str(jsonl_path))

    assert raw["format_meta"]["file_type"] == "jsonl"
    assert raw["records"] == [
        {"input": "a", "output": "b", "source_path": str(jsonl_path), "row_index": 1},
        {"input": "c", "output": "d", "source_path": str(jsonl_path), "row_index": 2},
    ]


def test_load_raw_data_plain_text(tmp_path: Path):
    text_path = tmp_path / "notes.txt"
    text_path.write_text("first line\n\nsecond line\n", encoding="utf-8")

    raw = load_raw_data(str(text_path))

    assert raw["format_meta"]["modality"] == "text"
    assert raw["records"] == [
        {"input": "first line", "output": "unknown", "source_path": str(text_path), "row_index": 1},
        {"input": "second line", "output": "unknown", "source_path": str(text_path), "row_index": 2},
    ]


def test_load_raw_data_image_file(tmp_path: Path):
    image_path = tmp_path / "cat.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n")

    raw = load_raw_data(str(image_path))

    assert raw["format_meta"] == {"modality": "image", "file_type": "png", "encoding": "binary"}
    assert raw["records"] == [
        {"source": "local_image", "path": str(image_path), "filename": "cat.png"}
    ]


def test_detect_format_image_directory(tmp_path: Path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "a.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (image_dir / "b.jpg").write_bytes(b"\xff\xd8\xff")

    fmt = detect_format(str(image_dir))
    raw = load_raw_data(str(image_dir))

    assert fmt == {"modality": "image", "file_type": "image_dir", "encoding": "binary"}
    assert [record["filename"] for record in raw["records"]] == ["a.png", "b.jpg"]


def test_load_raw_data_generic_directory(tmp_path: Path):
    data_dir = tmp_path / "mixed"
    data_dir.mkdir()
    (data_dir / "part1.json").write_text(json.dumps({"input": "hi", "output": "there"}), encoding="utf-8")
    (data_dir / "part2.txt").write_text("line one\nline two\n", encoding="utf-8")

    raw = load_raw_data(str(data_dir))

    assert raw["format_meta"]["file_type"] == "directory"
    assert raw["format_meta"]["modality"] == "tabular"
    assert len(raw["records"]) == 2
    assert raw["records"][0]["source_path"].endswith("part1.json")
    assert raw["records"][1]["source_path"].endswith("part2.txt")


def test_load_raw_data_parquet(tmp_path: Path):
    pytest = __import__("pytest")
    pytest.importorskip("pyarrow", reason="pyarrow is required for parquet Mode A test")

    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet_path = tmp_path / "examples.parquet"
    table = pa.table({"input": ["hello"], "label": ["greeting"]})
    pq.write_table(table, parquet_path)

    raw = load_raw_data(str(parquet_path))

    assert raw["format_meta"]["file_type"] == "parquet"
    assert raw["records"] == [{"input": "hello", "label": "greeting", "source_path": str(parquet_path), "row_index": 1}]
