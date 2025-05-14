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

from bson import ObjectId
from bson.errors import InvalidId
from fastapi import HTTPException

from src.api.db import get_db
from src.api.models import DeploymentStatus, FlywheelRun
from src.api.schemas import (
    Customization,
    Evaluation,
    JobDetailResponse,
    LLMJudgeResponse,
    NIMResponse,
)


def validate_object_id(id_str: str, param_name: str = "id") -> ObjectId:
    """
    Validate and convert a string ID to ObjectId.

    Args:
        id_str: The string ID to validate
        param_name: Name of the parameter for error messages

    Returns:
        ObjectId: The validated MongoDB ObjectId

    Raises:
        HTTPException: If the ID is invalid
    """
    try:
        if not ObjectId.is_valid(id_str):
            raise ValueError(f"Invalid {param_name} format")
        return ObjectId(id_str)
    except (InvalidId, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


def validate_object_id_list(ids: list[ObjectId]) -> list[ObjectId]:
    """
    Validate a list of ObjectIds to ensure they are safe for querying.

    Args:
        ids: List of ObjectIds to validate

    Returns:
        List[ObjectId]: The validated list of ObjectIds

    Raises:
        HTTPException: If any ID is invalid
    """
    if not isinstance(ids, list):
        raise HTTPException(status_code=400, detail="Expected a list of IDs")

    # Ensure all IDs are valid ObjectIds
    for id_obj in ids:
        if not isinstance(id_obj, ObjectId):
            raise HTTPException(status_code=400, detail=f"Invalid ID in list: {id_obj}")

    return ids


def get_job_details(job_id: str) -> JobDetailResponse:
    """
    Get the status and result of a job, including detailed information about all tasks in the workflow.
    """
    db = get_db()

    # Validate job_id and convert to ObjectId
    job_object_id = validate_object_id(job_id, "job_id")

    # Get the flywheel run document
    doc = db.flywheel_runs.find_one({"_id": job_object_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Job not found")

    flywheel_run = FlywheelRun.from_mongo(doc)

    # Get all NIMs for this flywheel run using the validated job_id
    nims = list(db.nims.find({"flywheel_run_id": job_object_id}))

    # Extract and validate NIM IDs
    nim_ids = validate_object_id_list([nim["_id"] for nim in nims])

    # Get all evaluations for these NIMs using validated IDs
    evaluations = list(db.evaluations.find({"nim_id": {"$in": nim_ids}}))

    # Group evaluations by NIM
    nim_evaluations: dict[ObjectId, list[Evaluation]] = {}
    for eval in evaluations:
        if eval["nim_id"] not in nim_evaluations:
            nim_evaluations[eval["nim_id"]] = []
        nim_evaluations[eval["nim_id"]].append(
            Evaluation(
                eval_type=eval["eval_type"],
                scores=eval["scores"],
                started_at=eval["started_at"],
                finished_at=eval["finished_at"],
                runtime_seconds=eval["runtime_seconds"],
                progress=eval["progress"],
                nmp_uri=eval["nmp_uri"],
                error=eval.get("error", None),
            )
        )

    # Get customizations using validated NIM IDs
    customizations = list(db.customizations.find({"nim_id": {"$in": nim_ids}}))
    nim_customizations: dict[ObjectId, list[Customization]] = {}
    for custom in customizations:
        if custom["nim_id"] not in nim_customizations:
            nim_customizations[custom["nim_id"]] = []
        nim_customizations[custom["nim_id"]].append(
            Customization(
                started_at=custom["started_at"],
                finished_at=custom["finished_at"],
                runtime_seconds=custom["runtime_seconds"],
                progress=custom["progress"],
                epochs_completed=custom["epochs_completed"],
                steps_completed=custom["steps_completed"],
                nmp_uri=custom["nmp_uri"],
                error=custom.get("error", None),
            )
        )

    # Get LLM judge data using validated job_id
    llm_judge = db.llm_judge_runs.find_one({"flywheel_run_id": job_object_id})
    if llm_judge:
        llm_judge_response = LLMJudgeResponse(
            model_name=llm_judge["model_name"],
            deployment_status=DeploymentStatus(
                llm_judge["deployment_status"] or DeploymentStatus.PENDING
            ),
            error=llm_judge.get("error", None),
        )
    else:
        llm_judge_response = None

    return JobDetailResponse(
        id=str(flywheel_run.id),
        workload_id=flywheel_run.workload_id,
        client_id=flywheel_run.client_id,
        status="failed"
        if flywheel_run.error
        else "completed"
        if flywheel_run.finished_at
        else "running",
        started_at=flywheel_run.started_at,
        finished_at=flywheel_run.finished_at,
        num_records=flywheel_run.num_records or 0,
        llm_judge=llm_judge_response,
        nims=[
            NIMResponse(
                model_name=nim["model_name"],
                deployment_status=DeploymentStatus(
                    nim["deployment_status"] or DeploymentStatus.PENDING
                ),
                evaluations=nim_evaluations.get(nim["_id"], []),
                customizations=nim_customizations.get(nim["_id"], []),
                error=nim.get("error", None),
            )
            for nim in nims
        ],
        datasets=flywheel_run.datasets,
        error=flywheel_run.error,
    )
