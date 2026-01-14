# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Dict, List

import ray
from ray.types import ObjectRef

class RayDPDataOwner:
    """Registry mixin for single-owner Spark->Ray Dataset conversion.

    JVM side behavior:
    - Spark executor (JVM) Ray actors generate Arrow IPC bytes for each batch and buffer them
      inside the executor process keyed by `batch_key` (see `RayDPExecutor.putArrowIPC`).
    - JVM returns (executor_actor_name, batch_key) pairs back to Python.

    Python side behavior (this class):
    - The owner/registry actor calls `executor_actor.popArrowIPC(batch_key)` to retrieve bytes.
      Since the owner actor is the *caller*, Ray assigns ownership of the returned object to
      this owner actor.
    - We return these Arrow IPC bytes refs directly. Ray Data's `from_arrow_refs()` accepts
      refs to Arrow IPC streaming bytes and will decode them internally as needed.
    """

    def fetch_block_refs(self, executor_actor_names: List[str], batch_keys: List[str]) -> List[ObjectRef]:
        # Ray cross-language calls (Python -> JVM actors) require the driver/worker to allow
        # loading code from local. In some environments/tests we don't start Ray with
        # `--load-code-from-local`, so we enable it programmatically here.
        try:
            from ray._private.worker import global_worker
            global_worker.set_load_code_from_local(True)
        except Exception:
            pass

        if len(executor_actor_names) != len(batch_keys):
            raise ValueError(
                f"executor_actor_names and batch_keys must have the same length, got "
                f"{len(executor_actor_names)} and {len(batch_keys)}")

        # Pop IPC bytes refs from executors. These refs are owned by this actor (caller).
        ipc_refs: List[ObjectRef] = []
        handles: Dict[str, ray.actor.ActorHandle] = {}
        for actor_name, key in zip(executor_actor_names, batch_keys):
            h = handles.get(actor_name)
            if h is None:
                h = ray.get_actor(actor_name)
                handles[actor_name] = h
            ipc_refs.append(h.popArrowIPC.remote(key))

        return ipc_refs
