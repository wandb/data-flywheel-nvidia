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
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from bson import ObjectId

from src.api.models import EvalType, NIMEvaluation, ToolEvalType, WorkloadClassification
from src.config import settings
from src.lib.nemo.evaluator import Evaluator


@pytest.fixture
def evaluator() -> Evaluator:
    with patch.dict("os.environ", {"NEMO_URL": "http://test-nemo-url"}):
        return Evaluator(llm_judge_config=settings.llm_judge_config)


@pytest.fixture
def mock_evaluation() -> NIMEvaluation:
    return NIMEvaluation(
        nim_id=ObjectId(),  # Generate a new ObjectId
        eval_type=EvalType.BASE,
        scores={"base": 0.0},
        started_at=datetime.utcnow(),
        finished_at=None,
        runtime_seconds=0.0,
        progress=0.0,
    )


def test_wait_for_evaluation_created_state(
    evaluator: Evaluator, mock_evaluation: NIMEvaluation
) -> None:
    """Test handling of 'created' state in wait_for_evaluation"""
    job_id = "test-job-id"
    progress_updates: list[dict[str, Any]] = []

    def progress_callback(update_data: dict[str, Any]) -> None:
        progress_updates.append(update_data)

    # Mock the job status response for 'created' state
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "status": "created",
        "status_details": {"message": None, "task_status": {}, "progress": None},
    }

    with patch("requests.get", return_value=mock_response):
        with pytest.raises(TimeoutError) as exc_info:
            evaluator.wait_for_evaluation(
                job_id=job_id,
                evaluation=mock_evaluation,
                polling_interval=1,
                timeout=1,
                progress_callback=progress_callback,
            )
        assert "stalled for more than 1 seconds" in str(exc_info.value)

        # Verify progress callback was called with 0 progress
        assert len(progress_updates) > 0
        assert progress_updates[0]["progress"] == 0.0


def test_wait_for_evaluation_running_state(
    evaluator: Evaluator, mock_evaluation: NIMEvaluation
) -> None:
    """Test handling of 'running' state in wait_for_evaluation"""
    job_id = "test-job-id"
    progress_updates: list[dict[str, Any]] = []

    def progress_callback(update_data: dict[str, Any]) -> None:
        progress_updates.append(update_data)

    # Mock the job status response for 'running' state with increasing progress
    responses = [
        {"status": "running", "status_details": {"progress": 50}},
        {"status": "running", "status_details": {"progress": 75}},
        {"status": "completed", "status_details": {"progress": 100}},
    ]
    mock_response = MagicMock()
    mock_response.json = MagicMock(side_effect=responses)

    with patch("requests.get", return_value=mock_response):
        result = evaluator.wait_for_evaluation(
            job_id=job_id,
            evaluation=mock_evaluation,
            polling_interval=0.1,  # Reduce polling interval for test
            timeout=1,
            progress_callback=progress_callback,
        )

        # Verify progress callback was called with correct progress
        assert len(progress_updates) > 0
        assert progress_updates[0]["progress"] == 50.0
        assert result["status"] == "completed"


def test_wait_for_evaluation_completed_state(
    evaluator: Evaluator, mock_evaluation: NIMEvaluation
) -> None:
    """Test handling of 'completed' state in wait_for_evaluation"""
    job_id = "test-job-id"
    progress_updates: list[dict[str, Any]] = []

    def progress_callback(update_data: dict[str, Any]) -> None:
        progress_updates.append(update_data)

    # Mock the job status response for 'completed' state
    mock_response = MagicMock()
    mock_response.json.return_value = {"status": "completed", "status_details": {"progress": 100}}

    with patch("requests.get", return_value=mock_response):
        result = evaluator.wait_for_evaluation(
            job_id=job_id,
            evaluation=mock_evaluation,
            polling_interval=1,
            timeout=1,
            progress_callback=progress_callback,
        )

        # Verify progress callback was called with 100% progress
        assert len(progress_updates) > 0
        assert progress_updates[0]["progress"] == 100.0
        # Verify the job data was returned
        assert result["status"] == "completed"


def test_wait_for_evaluation_error_state(
    evaluator: Evaluator, mock_evaluation: NIMEvaluation
) -> None:
    """Test handling of error state in wait_for_evaluation"""
    job_id = "test-job-id"

    # Mock the job status response for error state
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "status": "failed",
        "status_details": {"error": "Test error"},
    }

    with patch("requests.get", return_value=mock_response):
        with pytest.raises(Exception) as exc_info:
            evaluator.wait_for_evaluation(
                job_id=job_id, evaluation=mock_evaluation, polling_interval=1, timeout=1
            )

        assert "Job status: failed" in str(exc_info.value)


def test_wait_for_evaluation_timeout(evaluator: Evaluator, mock_evaluation: NIMEvaluation) -> None:
    """Test timeout handling in wait_for_evaluation"""
    job_id = "test-job-id"
    progress_updates: list[dict[str, Any]] = []

    def progress_callback(update_data: dict[str, Any]) -> None:
        progress_updates.append(update_data)

    # Mock the job status response to always return same progress
    mock_response = MagicMock()
    mock_response.json.return_value = {"status": "running", "status_details": {"progress": 50}}

    with patch("requests.get", return_value=mock_response):
        with pytest.raises(TimeoutError) as exc_info:
            evaluator.wait_for_evaluation(
                job_id=job_id,
                evaluation=mock_evaluation,
                polling_interval=1,
                timeout=1,
                progress_callback=progress_callback,
            )
        assert "stalled for more than 1 seconds" in str(exc_info.value)

        # Verify progress callback was called with the stalled progress
        assert len(progress_updates) > 0
        assert progress_updates[-1]["progress"] == 0.0
        assert "error" in progress_updates[-1]


def test_wait_for_evaluation_none_progress(
    evaluator: Evaluator, mock_evaluation: NIMEvaluation
) -> None:
    """Test handling of None progress value in wait_for_evaluation"""
    job_id = "test-job-id"
    progress_updates: list[dict[str, Any]] = []

    def progress_callback(update_data: dict[str, Any]) -> None:
        progress_updates.append(update_data)

    # Mock the job status response with None progress
    mock_response = MagicMock()
    mock_response.json.return_value = {"status": "running", "status_details": {"progress": None}}

    with patch("requests.get", return_value=mock_response):
        with pytest.raises(TimeoutError) as exc_info:
            evaluator.wait_for_evaluation(
                job_id=job_id,
                evaluation=mock_evaluation,
                polling_interval=1,
                timeout=1,
                progress_callback=progress_callback,
            )
        assert "stalled for more than 1 seconds" in str(exc_info.value)

        # Verify progress callback was called with 0 progress
        assert len(progress_updates) > 0
        assert progress_updates[0]["progress"] == 0.0


def make_remote_judge_config():
    from src.config import LLMJudgeConfig

    return LLMJudgeConfig(
        type="remote",
        url="http://test-remote-url/v1/chat/completions",
        model_id="remote-model-id",
        api_key_env="TEST_API_KEY_ENV",
        api_key="test-api-key",
    )


def make_local_judge_config():
    from src.config import LLMJudgeConfig

    return LLMJudgeConfig(
        type="local",
        model_name="local-model-name",
        tag="test-tag",
        context_length=1234,
        gpus=1,
        pvc_size="10Gi",
        registry_base="test-registry",
        customization_enabled=True,
    )


def test_evaluator_uses_remote_judge_config(monkeypatch):
    from src.lib.nemo.evaluator import Evaluator

    remote_cfg = make_remote_judge_config()
    monkeypatch.setattr("src.config.settings.llm_judge_config", remote_cfg)
    evaluator = Evaluator()
    # Should use the remote config dict
    assert isinstance(evaluator.judge_model_config, dict)
    assert evaluator.judge_model_config["api_endpoint"]["url"] == remote_cfg.url
    assert evaluator.judge_model_config["api_endpoint"]["model_id"] == remote_cfg.model_id
    assert evaluator.judge_model_config["api_endpoint"]["api_key"] == remote_cfg.api_key


def test_evaluator_uses_local_judge_config(monkeypatch):
    from src.lib.nemo.evaluator import Evaluator

    local_cfg = make_local_judge_config()
    monkeypatch.setattr("src.config.settings.llm_judge_config", local_cfg)
    evaluator = Evaluator()
    # Should use the local model name
    assert evaluator.judge_model_config == local_cfg.model_name


def test_evaluator_prefers_explicit_llm_judge_config(monkeypatch):
    from src.lib.nemo.evaluator import Evaluator

    remote_cfg = make_remote_judge_config()
    monkeypatch.setattr("src.config.settings.llm_judge_config", remote_cfg)

    # If you pass an explicit NIMConfig, it should use that model_name
    class DummyNIMConfig:
        model_name = "explicit-model"

    evaluator = Evaluator(llm_judge_config=DummyNIMConfig())
    assert evaluator.judge_model_config == "explicit-model"


@pytest.fixture
def mock_nemo_url_val():
    return "http://mock-nemo-api.com"


@pytest.fixture
def mock_judge_model_name_val():
    return "mock/judge-model-name"


@pytest.fixture
def evaluator_instance(mock_nemo_url_val, mock_judge_model_name_val):
    """
    Provides an instance of the Evaluator with mocked settings
    for nemo_url and judge_model_config.
    """
    with patch("src.lib.nemo.evaluator.settings") as mock_settings:
        mock_settings.nmp_config.nemo_base_url = mock_nemo_url_val

        mock_judge_cfg = MagicMock()
        mock_judge_cfg.is_remote.return_value = False

        mock_local_nim_cfg = MagicMock()
        mock_local_nim_cfg.model_name = mock_judge_model_name_val
        mock_judge_cfg.get_local_nim_config.return_value = mock_local_nim_cfg

        mock_settings.llm_judge_config = mock_judge_cfg

        yield Evaluator()


@pytest.fixture
def mock_response():
    """Provides a mock successful response for requests.post."""
    response = MagicMock()
    response.status_code = 201
    response.json.return_value = {"id": "mock-job-id"}
    return response


class TestRunEvaluation:
    @pytest.mark.parametrize(
        "test_params",
        [
            # Test case 1: GENERIC workload
            {
                "workload_type": WorkloadClassification.GENERIC,
                "tool_eval_type": None,
                "namespace": "test-namespace",
                "dataset_name": "test-dataset",
                "target_model": "meta/llama-3.3-70b-instruct",
                "test_file": "test.jsonl",
                "limit": 75,
                "expected_config_method": "get_llm_as_judge_config",
                "should_raise_error": False,
            },
            # Test case 2: TOOL_CALLING workload with missing tool_eval_type
            {
                "workload_type": WorkloadClassification.TOOL_CALLING,
                "tool_eval_type": None,
                "namespace": "test-namespace",
                "dataset_name": "test-dataset",
                "target_model": "meta/llama-3.3-70b-instruct",
                "test_file": "test-tools.jsonl",
                "limit": 50,
                "expected_error": ValueError,
                "expected_error_msg": "tool_eval_type must be provided for tool calling workload",
                "should_raise_error": True,
            },
            # Test case 3: TOOL_CALLING workload with TOOL_CALLING_METRIC
            {
                "workload_type": WorkloadClassification.TOOL_CALLING,
                "tool_eval_type": ToolEvalType.TOOL_CALLING_METRIC,
                "namespace": "test-tool-namespace",
                "dataset_name": "test-tool-dataset",
                "target_model": "custom/tool-model",
                "test_file": "tools-data.jsonl",
                "limit": 99,
                "expected_config_method": "get_tool_calling_config",
                "should_raise_error": False,
            },
        ],
    )
    def test_run_evaluation_scenarios(self, evaluator_instance, test_params):
        # Mock the specific evaluation jobs endpoint
        evaluation_jobs_endpoint = f"{evaluator_instance.nemo_url}/v1/evaluation/jobs"

        with patch("src.lib.nemo.evaluator.requests.post") as mock_post:
            # Configure mock response for the specific endpoint
            mock_response = MagicMock()
            mock_response.status_code = 201
            mock_response.json.return_value = {
                "id": "mock-job-id",
                "status": "created",
                "created_at": "2024-03-20T10:00:00Z",
            }
            mock_post.return_value = mock_response

            if test_params["should_raise_error"]:
                with pytest.raises(
                    test_params["expected_error"], match=test_params["expected_error_msg"]
                ):
                    evaluator_instance.run_evaluation(
                        namespace=test_params["namespace"],
                        dataset_name=test_params["dataset_name"],
                        workload_type=test_params["workload_type"],
                        target_model=test_params["target_model"],
                        test_file=test_params["test_file"],
                        tool_eval_type=test_params["tool_eval_type"],
                        limit=test_params["limit"],
                    )
                return

            config_method = getattr(evaluator_instance, test_params["expected_config_method"])
            expected_config_payload = config_method(
                namespace=test_params["namespace"],
                dataset_name=test_params["dataset_name"],
                test_file=test_params["test_file"],
                limit=test_params["limit"],
            )

            job_id = evaluator_instance.run_evaluation(
                namespace=test_params["namespace"],
                dataset_name=test_params["dataset_name"],
                workload_type=test_params["workload_type"],
                tool_eval_type=test_params["tool_eval_type"],
                target_model=test_params["target_model"],
                test_file=test_params["test_file"],
                limit=test_params["limit"],
            )

            assert job_id == "mock-job-id"

            # Verify the POST request was made to the correct endpoint
            mock_post.assert_called_once_with(
                evaluation_jobs_endpoint,
                json={
                    "config": expected_config_payload,
                    "target": {"type": "model", "model": test_params["target_model"]},
                },
            )


@pytest.mark.parametrize(
    "limit",
    [
        75,  # Normal positive limit
        0,  # Zero limit
        None,  # No limit
    ],
)
def test_run_evaluation_limit_propagation(evaluator_instance, mock_response, limit):
    """Test that run_evaluation correctly passes limit to config methods."""
    # Mock the network request
    with patch("src.lib.nemo.evaluator.requests.post", return_value=mock_response):
        # Mock the config method
        with patch.object(evaluator_instance, "get_llm_as_judge_config") as mock_config_method:
            # Set up mock return value
            mock_config_method.return_value = {"type": "mock-config"}

            # Call run_evaluation with limit
            evaluator_instance.run_evaluation(
                namespace="test-namespace",
                dataset_name="test-dataset",
                workload_type=WorkloadClassification.GENERIC,
                target_model="test-model",
                test_file="test.jsonl",
                limit=limit,
            )

            # Verify config method was called with correct limit
            mock_config_method.assert_called_once_with(
                namespace="test-namespace",
                dataset_name="test-dataset",
                test_file="test.jsonl",
                limit=limit,
            )
