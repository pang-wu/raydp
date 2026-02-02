#
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
#

import os
import pytest
import pyarrow as pa
from pyspark.storagelevel import StorageLevel
import ray
from ray.cluster_utils import Cluster
from ray.data import from_arrow_refs
import ray.util.client as ray_client
import raydp

try:
    # Ray cross-language calls require enabling load_code_from_local.
    # This is an internal Ray API; keep it isolated and optional.
    from ray._private.worker import global_worker as _ray_global_worker  # type: ignore
except Exception:  # pragma: no cover
    _ray_global_worker = None

@ray.remote(max_retries=-1)
def _fetch_arrow_table_from_executor(
    executor_actor_name: str,
    rdd_id: int,
    partition_id: int,
    schema_json: str,
    driver_agent_url: str,
) -> pa.Table:
    """Fetch Arrow table bytes from a JVM executor actor and decode to `pyarrow.Table`.

    This is a test-local version of RayDP's recoverable fetch task. Keeping it in this test
    avoids Ray remote function registration issues when driver/workers import different `raydp`
    versions.
    """
    if _ray_global_worker is not None:
        _ray_global_worker.set_load_code_from_local(True)

    executor_actor = ray.get_actor(executor_actor_name)
    ipc_bytes = ray.get(
        executor_actor.getRDDPartition.remote(
            rdd_id, partition_id, schema_json, driver_agent_url
        )
    )
    reader = pa.ipc.open_stream(pa.BufferReader(ipc_bytes))
    table = reader.read_all()
    # Match RayDP behavior: strip schema metadata for stability.
    table = table.replace_schema_metadata()
    return table



def test_recoverable_forwarding_via_fetch_task(jdk17_extra_spark_configs):
    """Verify JVM-side forwarding in recoverable Spark->Ray conversion.

    This test intentionally calls the recoverable fetch task on the *wrong* Spark executor actor.
    It should still succeed because `RayDPExecutor.getRDDPartition` refreshes the block owner and
    forwards the fetch one hop.
    """
    if ray_client.ray.is_connected():
        pytest.skip("Skip forwarding test in Ray client mode")

    stop_after = os.environ.get("RAYDP_TRACE_STOP_AFTER", "").strip().lower()
    fetch_mode = os.environ.get("RAYDP_FETCH_MODE", "task").strip().lower()
    cluster = Cluster(
        initialize_head=True,
        head_node_args={
            "num_cpus": 2,
            "resources": {"master": 10},
            "include_dashboard": True,
            "dashboard_port": 0,
        },
    )
    cluster.add_node(num_cpus=4, resources={"spark_executor": 10})
    
    def phase(name: str) -> None:
        # Prints are the most reliable breadcrumb if the raylet crashes.
        print(f"\n=== PHASE: {name} ===", flush=True)

    def should_stop(name: str) -> bool:
        return bool(stop_after) and stop_after == name.lower()

    spark = None
    try:
        # Single-node Ray is sufficient to reproduce / bisect the crash.
        phase("ray.init")
        ray.shutdown()
        ray.init(address=cluster.address, include_dashboard=False)
        if should_stop("ray.init"):
            return

        phase("raydp.init_spark")
        node_ip = ray.util.get_node_ip_address()
        spark = raydp.init_spark(
            app_name="test_recoverable_forwarding_via_fetch_task",
            num_executors=2,
            executor_cores=1,
            executor_memory="500M",
            configs={
                "spark.driver.host": node_ip,
                "spark.driver.bindAddress": node_ip,
                **jdk17_extra_spark_configs,
            },
        )
        if should_stop("raydp.init_spark"):
            return

        phase("spark.range.count")
        df = spark.range(0, 10000, numPartitions=8)
        _ = df.count()
        if should_stop("spark.range.count"):
            return

        phase("prepareRecoverableRDD")
        sc = spark.sparkContext
        storage_level = sc._getJavaStorageLevel(StorageLevel.MEMORY_AND_DISK)
        object_store_writer = sc._jvm.org.apache.spark.sql.raydp.ObjectStoreWriter
        info = object_store_writer.prepareRecoverableRDD(df._jdf, storage_level)
        rdd_id = info.rddId()
        schema_json = info.schemaJson()
        driver_agent_url = info.driverAgentUrl()
        locations = list(info.locations())
        if should_stop("preparerecoverablerdd"):
            return

        assert locations
        unique_execs = sorted(set(locations))
        assert len(unique_execs) >= 2, f"Need >=2 executors, got {unique_execs}"

        partition_id = 0
        owner_executor_id = locations[partition_id]
        wrong_executor_id = next(e for e in unique_execs if e != owner_executor_id)
        wrong_executor_actor_name = f"raydp-executor-{wrong_executor_id}"

        phase("fetch_wrong_executor")

        phase("get_wrong_executor_actor")
        wrong_executor_actor = ray.get_actor(wrong_executor_actor_name)
        if should_stop("get_wrong_executor_actor"):
            return

        phase("call_fetch_task")
        if fetch_mode == "driver":
            phase("driver_call_java_actor")
            if _ray_global_worker is not None:
                _ray_global_worker.set_load_code_from_local(True)
            ipc_bytes = ray.get(
                wrong_executor_actor.getRDDPartition.remote(
                    rdd_id, partition_id, schema_json, driver_agent_url
                )
            )
            reader = pa.ipc.open_stream(pa.BufferReader(ipc_bytes))
            table = reader.read_all()
            table = table.replace_schema_metadata()
        else:
            phase("task_call_java_actor")
            refs: list[ray.ObjectRef] = []
            refs.append(
                _fetch_arrow_table_from_executor.remote(
                    wrong_executor_actor_name,
                    rdd_id,
                    partition_id,
                    schema_json,
                    driver_agent_url,
                )
            )
            table = from_arrow_refs(refs)
        assert table.count() > 0
    finally:
        phase("teardown")
        
        spark.stop()
        raydp.stop_spark()
        ray.shutdown()
        cluster.shutdown()

