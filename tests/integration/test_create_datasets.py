# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from collections.abc import Generator
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from bson import ObjectId

from src.api.models import DatasetType, FlywheelRun, TaskResult, WorkloadClassification  # E402
from src.config import settings
from src.lib.flywheel.util import DataSplitConfig  # E402
from src.tasks.tasks import create_datasets  # E402


@pytest.fixture
def mock_external_services() -> Generator[dict[str, MagicMock], None, None]:
    """Mock external service responses"""
    with (
        patch("src.lib.nemo.data_uploader.DataUploader.upload_data") as mock_upload_data,
        patch("src.lib.nemo.data_uploader.DataUploader.get_file_uri") as mock_get_file_uri,
    ):
        # Configure mock_get_file_uri to return a string URI
        mock_get_file_uri.return_value = "test_uri"
        # Configure mock_upload_data to return the file path
        mock_upload_data.side_effect = lambda data, file_path: file_path

        yield {
            "upload_data": mock_upload_data,
            "get_file_uri": mock_get_file_uri,
        }


@pytest.fixture
def create_flywheel_run(test_workload_id: str, client_id: str):
    """Helper fixture to create a flywheel run document."""
    flywheel_run = FlywheelRun(
        workload_id=test_workload_id,
        client_id=client_id,
        started_at=datetime.utcnow(),
        num_records=0,
        nims=[],
    )
    from src.api.db import get_db, init_db

    init_db()
    mongo_db = get_db()
    result = mongo_db.flywheel_runs.insert_one(flywheel_run.to_mongo())
    return str(result.inserted_id), mongo_db


@pytest.mark.integration
def test_create_datasets(
    test_workload_id: str,
    client_id: str,
    create_flywheel_run,
    load_test_data_fixture,
    mock_external_services,
) -> None:
    """Test the create_datasets task with default split configuration."""
    # Create default split configuration
    settings.data_split_config = DataSplitConfig(
        eval_size=1,  # Small eval set for testing
        val_ratio=0.5,  # 50% validation split
    )

    # Select the appropriate workload ID and load the corresponding test data
    current_workload_id = test_workload_id

    flywheel_run_id, mongo_db = create_flywheel_run

    # Run the task
    result = create_datasets(
        workload_id=current_workload_id,
        flywheel_run_id=flywheel_run_id,
        client_id=client_id,
    )

    # Convert result to TaskResult model
    task_result = TaskResult.model_validate(result)

    # Verify result structure
    assert task_result is not None
    assert task_result.workload_id == current_workload_id
    assert task_result.flywheel_run_id == flywheel_run_id

    # Verify datasets were created with correct types
    assert task_result.datasets is not None
    assert DatasetType.BASE in task_result.datasets
    assert DatasetType.ICL in task_result.datasets
    assert DatasetType.TRAIN in task_result.datasets

    # Verify dataset names follow expected format
    for dataset_name in task_result.datasets.values():
        assert dataset_name.startswith("flywheel-")
        assert current_workload_id in dataset_name

    # Verify MongoDB document structure
    db_doc = mongo_db.flywheel_runs.find_one({"_id": ObjectId(flywheel_run_id)})
    assert db_doc is not None
    assert "datasets" in db_doc
    assert len(db_doc["datasets"]) == 3  # BASE, ICL, and TRAIN datasets

    # Verify each dataset document structure
    for dataset in db_doc["datasets"]:
        assert "name" in dataset
        assert "num_records" in dataset
        assert "nmp_uri" in dataset
        assert dataset["nmp_uri"] == "test_uri"  # From our mock
        assert dataset["name"].startswith("flywheel-")
        assert current_workload_id in dataset["name"]

    # Verify mock calls
    mock_upload_data = mock_external_services["upload_data"]
    mock_get_file_uri = mock_external_services["get_file_uri"]

    # Should be called 4 times: eval data, icl data, train data, and val data
    assert mock_upload_data.call_count == 4
    assert mock_get_file_uri.call_count == 3  # Called once for each dataset type


@pytest.mark.integration
def test_create_datasets_workload_classification(
    test_workload_id: str,
    client_id: str,
    create_flywheel_run,
    load_test_data_fixture,
    mock_external_services,
) -> None:
    """Test that workload type is correctly identified and stored."""
    # Select the appropriate workload ID and load the corresponding test data
    current_workload_id = test_workload_id
    flywheel_run_id, _ = create_flywheel_run

    result = create_datasets(
        workload_id=current_workload_id,
        flywheel_run_id=flywheel_run_id,
        client_id=client_id,
    )
    task_result = TaskResult.model_validate(result)

    # Verify workload type is set
    assert task_result.workload_type is not None
    assert task_result.workload_type in [
        WorkloadClassification.GENERIC,
        WorkloadClassification.TOOL_CALLING,
    ]


@pytest.mark.integration
def test_create_datasets_different_split_configs(
    test_workload_id: str,
    client_id: str,
    create_flywheel_run,
    load_test_data_fixture,
    mock_external_services,
) -> None:
    """Test dataset creation with different split configurations."""
    # Select the appropriate workload ID and load the corresponding test data
    current_workload_id = test_workload_id
    flywheel_run_id, mongo_db = create_flywheel_run

    # Test with different split configurations
    test_configs = [
        DataSplitConfig(eval_size=2, val_ratio=0.3),
        DataSplitConfig(eval_size=3, val_ratio=0.4),
    ]

    for config in test_configs:
        settings.data_split_config = config
        result = create_datasets(
            workload_id=current_workload_id,
            flywheel_run_id=flywheel_run_id,
            client_id=client_id,
        )
        TaskResult.model_validate(result)

        # Verify datasets reflect the split configuration
        db_doc = mongo_db.flywheel_runs.find_one({"_id": ObjectId(flywheel_run_id)})
        datasets = db_doc["datasets"]
        eval_dataset = next(d for d in datasets if "eval" in d["name"])
        assert eval_dataset["num_records"] == config.eval_size


@pytest.mark.integration
def test_create_datasets_upload_validation(
    test_workload_id: str,
    client_id: str,
    create_flywheel_run,
    load_test_data_fixture,
    mock_external_services,
) -> None:
    """Test that training data is properly formatted."""
    # Select the appropriate workload ID and load the corresponding test data
    current_workload_id = test_workload_id

    flywheel_run_id, mongo_db = create_flywheel_run

    create_datasets(
        workload_id=current_workload_id,
        flywheel_run_id=flywheel_run_id,
        client_id=client_id,
    )

    # Verify the upload_data calls contain properly formatted data
    mock_upload_data = mock_external_services["upload_data"]
    calls = mock_upload_data.call_args_list

    eval_call, icl_call, train_call, val_call = calls
    eval_data = eval_call[0][0]
    icl_data = icl_call[0][0]
    train_data = train_call[0][0]
    val_data = val_call[0][0]

    # Verify data is not None
    assert eval_data is not None
    assert icl_data is not None
    assert train_data is not None
    assert val_data is not None


@pytest.mark.integration
def test_create_datasets_with_prefix(
    test_workload_id: str, flywheel_run_id: str, client_id: str, mock_external_services
) -> None:
    """Test the create_datasets task with a custom dataset prefix."""
    # Create default split configuration
    settings.data_split_config = DataSplitConfig(eval_size=1, val_ratio=0.5)

    # Run the task with prefix
    prefix = "test-prefix"
    result = create_datasets(
        workload_id=test_workload_id,
        flywheel_run_id=flywheel_run_id,
        client_id=client_id,
        output_dataset_prefix=prefix,
    )

    # Convert result to TaskResult model
    task_result = TaskResult.model_validate(result)

    # Verify dataset names include the prefix
    assert task_result.datasets is not None
    for dataset_name in task_result.datasets.values():
        assert prefix in dataset_name
