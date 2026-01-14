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

import time

import pytest
import ray
from ray.cluster_utils import Cluster
from ray.util.state import list_actors

import raydp
from raydp.spark.dataset import _save_spark_df_to_object_store
from raydp.spark.ray_cluster_master import RAYDP_SPARK_MASTER_SUFFIX
from raydp.spark.dataset import get_raydp_master_owner


def _node_ip_by_node_id(node_id: str) -> str:
    for n in ray.nodes():
        if n.get("NodeID") == node_id:
            ip = n.get("NodeManagerAddress")
            if ip:
                return ip
    raise AssertionError(f"Cannot resolve node ip for node_id={node_id}. ray.nodes()={ray.nodes()}")


def _actor_node_id_by_name(actor_name: str) -> str:
    actors = list_actors(filters=[("name", "=", actor_name)], detail=True)
    assert len(actors) == 1, f"{actor_name} actor not found or multiple found: {actors}"
    node_id = getattr(actors[0], "node_id", None) or getattr(actors[0], "nodeId", None)
    assert node_id, f"Missing node_id on actor state for {actor_name}: {actors[0]}"
    return node_id


def _single_executor_actor_name() -> str:
    actors = list_actors(detail=True)
    names = [
        a.name for a in actors
        if getattr(a, "name", None)
        and a.name.startswith("raydp-executor-")
        and getattr(a, "state", None) == "ALIVE"
    ]
    assert len(names) == 1, f"Expected exactly one executor actor, got: {names}"
    return names[0]


def test_dataset_blocks_local_to_executor_node(jdk17_extra_spark_configs):
    """Ensure Spark->Ray Dataset blocks remain physically located on executor node.

    New design expectation:
    - JVM executor actor generates Arrow IPC bytes and returns them as Ray object(s).
    - A single owner actor owns the returned refs, but the bytes should be located on
      the executor node (not the owner node) unless explicitly fetched elsewhere.
    """
    ray.shutdown()

    cluster = Cluster(
        initialize_head=True,
        head_node_args={
            "num_cpus": 1,
            "resources": {"spark_master": 10},
            "include_dashboard": True,
            "dashboard_port": 0,
        },
    )
    cluster.add_node(num_cpus=2, resources={"spark_executor": 2})

    ray.init(address=cluster.address)

    app_name = "test_object_locality"
    spark = raydp.init_spark(
        app_name=app_name,
        num_executors=1,
        executor_cores=1,
        executor_memory="500M",
        configs={
            **jdk17_extra_spark_configs,
            # Pin owner (spark master) to head node.
            "spark.ray.raydp_spark_master.actor.resource.spark_master": "1",
            "spark.ray.raydp_spark_master.actor.resource.CPU": "0",
            # Pin executor actor to worker node(s).
            "spark.ray.raydp_spark_executor.actor.resource.spark_executor": "1",
        },
    )

    # Make a sufficiently large DF to avoid any "tiny object" special-casing.
    df = spark.range(0, 200_000).repartition(1)

    owner = get_raydp_master_owner(spark)
    block_refs, _ = _save_spark_df_to_object_store(df, use_batch=False, owner=owner)
    assert len(block_refs) >= 1

    owner_actor_name = app_name + RAYDP_SPARK_MASTER_SUFFIX
    executor_actor_name = _single_executor_actor_name()

    owner_node_id = _actor_node_id_by_name(owner_actor_name)
    executor_node_id = _actor_node_id_by_name(executor_actor_name)

    # If they end up on the same node (unexpected in this multi-node cluster), skip.
    if owner_node_id == executor_node_id:
        pytest.skip("Owner actor and executor actor are colocated; locality assertion is not meaningful.")

    # Check physical locations of the returned IPC-bytes objects.
    locs = ray.experimental.get_object_locations(block_refs)
    for ref in block_refs:
        meta = locs.get(ref)
        assert meta is not None, f"Missing location metadata for {ref}: {locs}"
        node_ids = meta.get("node_ids") or []
        assert executor_node_id in node_ids, (
            f"Expected block {ref} to be located on executor node.\n"
            f"owner_node_id={owner_node_id} ({_node_ip_by_node_id(owner_node_id)}), "
            f"executor_node_id={executor_node_id} ({_node_ip_by_node_id(executor_node_id)}), "
            f"node_ids={node_ids}, meta={meta}"
        )
        assert owner_node_id not in node_ids, (
            f"Expected block {ref} not to be located on owner node.\n"
            f"owner_node_id={owner_node_id} ({_node_ip_by_node_id(owner_node_id)}), "
            f"executor_node_id={executor_node_id} ({_node_ip_by_node_id(executor_node_id)}), "
            f"node_ids={node_ids}, meta={meta}"
        )

    raydp.stop_spark()
    ray.shutdown()
    cluster.shutdown()

