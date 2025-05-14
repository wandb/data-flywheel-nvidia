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
from fastapi import FastAPI

from src.api.db import init_db
from src.api.endpoints import router as api_router
from src.lib.nemo.evaluator import Evaluator
from src.log_utils import setup_logging

logger = setup_logging("data_flywheel.app")

app = FastAPI()

# Mount the API router
app.include_router(api_router, prefix="/api")

checked_evaluator_llm_judge_availability = False


def check_evaluator_llm_judge_availability():
    global checked_evaluator_llm_judge_availability
    if not checked_evaluator_llm_judge_availability:
        logger.info("Checking evaluator LLM judge availability...")
        Evaluator().validate_llm_judge_availability()
        logger.info("Evaluator LLM judge is available!")
        checked_evaluator_llm_judge_availability = True


@app.on_event("startup")
def startup_db_client():
    init_db()
    check_evaluator_llm_judge_availability()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
