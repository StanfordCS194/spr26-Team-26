from src.data_sources import looks_like_local_data_path, normalize_hf_dataset_source


def test_normalize_hf_dataset_source_accepts_supported_forms():
    assert normalize_hf_dataset_source("hf://SetFit/sst2") == "hf://SetFit/sst2"
    assert normalize_hf_dataset_source("SetFit/sst2") == "hf://SetFit/sst2"
    assert (
        normalize_hf_dataset_source("https://huggingface.co/datasets/SetFit/sst2")
        == "hf://SetFit/sst2"
    )


def test_normalize_hf_dataset_source_rejects_unsupported_sources():
    assert normalize_hf_dataset_source(None) is None
    assert normalize_hf_dataset_source("") is None
    assert normalize_hf_dataset_source("https://example.com/datasets/SetFit/sst2") is None
    assert normalize_hf_dataset_source("s3://bucket/train.jsonl") is None
    assert normalize_hf_dataset_source("not-a-dataset") is None


def test_looks_like_local_data_path_detects_common_paths():
    assert looks_like_local_data_path("/tmp/train.jsonl") is True
    assert looks_like_local_data_path("./train.csv") is True
    assert looks_like_local_data_path("../data/train.parquet") is True
    assert looks_like_local_data_path("data/train.jsonl") is True
    assert looks_like_local_data_path("SetFit/sst2") is False
