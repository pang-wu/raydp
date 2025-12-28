
import sys
import time
from typing import Any

import pytest
import ray
from ray._private.client_mode_hook import client_mode_wrap
from ray.cluster_utils import Cluster
from ray.exceptions import RayTaskError, OwnerDiedError
import ray.util.client as ray_client
from ray.util.state import list_actors
import raydp
from raydp.spark import PartitionObjectsOwner
from pyspark.sql import SparkSession
from raydp.spark import get_raydp_master_owner
from raydp.spark.object_owner import RayDPBlockStoreActorRegistry


def gen_test_data(spark_session: SparkSession):
  data = []
  tmp = [("ming", 20, 15552211521),
          ("hong", 19, 13287994007),
          ("dave", 21, 15552211523),
          ("john", 40, 15322211523),
          ("wong", 50, 15122211523)]

  for _ in range(10):
    data += tmp

  rdd = spark_session.sparkContext.parallelize(data)
  out = spark_session.createDataFrame(rdd, ["Name", "Age", "Phone"])
  return out

@client_mode_wrap
def ray_gc():
  ray._private.internal_api.global_gc()

def test_fail_without_data_ownership_transfer(ray_cluster, jdk17_extra_spark_configs):
  """
  Test shutting down Spark worker after data been put
  into Ray object store without data ownership transfer.
  This test should be throw error of data inaccessible after
  its owner (e.g. Spark JVM process) has terminated, which is expected.
  """

  # skipping this to be compatible with ray 2.4.0
  # see issue #343
  if ray_client.ray.is_connected():
        pytest.skip("Skip this test if using ray client")

  from raydp.spark.dataset import spark_dataframe_to_ray_dataset

  num_executor = 1
  spark = raydp.init_spark(
    app_name = "example",
    num_executors = num_executor,
    executor_cores = 1,
    executor_memory = "500M",
    configs=jdk17_extra_spark_configs
    )

  df_train = gen_test_data(spark)
  # df_train = df_train.sample(False, 0.001, 42)

  resource_stats = ray.available_resources()
  cpu_cnt = resource_stats['CPU']

  # convert data from spark dataframe to ray dataset without data ownership transfer
  ds = spark_dataframe_to_ray_dataset(df_train, parallelism=4)

  # display data
  ds.show(5)

  # release resource by shutting down spark
  raydp.stop_spark()
  ray_gc() # ensure GC kicked in
  time.sleep(3)

  # confirm that resources has been recycled
  resource_stats = ray.available_resources()
  assert resource_stats['CPU'] == cpu_cnt + num_executor

  # confirm that data get lost (error thrown)
  try:
    ds.mean('Age')
  except RayTaskError as e:
    assert isinstance(e.cause, OwnerDiedError)

def test_data_ownership_transfer(ray_cluster, jdk17_extra_spark_configs):
  """
  Test shutting down Spark worker after data been put
  into Ray object store with data ownership transfer.
  This test should be able to execute till the end without crash as expected.
  """

  if ray_client.ray.is_connected():
        pytest.skip("Skip this test if using ray client")

  from raydp.spark.dataset import spark_dataframe_to_ray_dataset
  import numpy as np

  num_executor = 1

  spark = raydp.init_spark(
    app_name = "example",
    num_executors = num_executor,
    executor_cores = 1,
    executor_memory = "500M",
    configs=jdk17_extra_spark_configs
    )

  df_train = gen_test_data(spark)

  resource_stats = ray.available_resources()
  cpu_cnt = resource_stats['CPU']

  # convert data from spark dataframe to ray dataset,
  # and transfer data ownership to dedicated Object Holder (Singleton)
  ds = spark_dataframe_to_ray_dataset(df_train, parallelism=4,
                                      owner=get_raydp_master_owner(spark))

  # display data
  ds.show(5)

  # release resource by shutting down spark Java process
  raydp.stop_spark(cleanup_data=False)
  ray_gc() # ensure GC kicked in
  time.sleep(3)

  # confirm that resources has been recycled
  resource_stats = ray.available_resources()
  assert resource_stats['CPU'] == cpu_cnt + num_executor

  # confirm that data is still available from object store!
  # sanity check the dataset is as functional as normal
  assert np.isnan(ds.mean('Age')) is not True

  # final clean up
  raydp.stop_spark()

def _print_actors():
    # Debug: print all actors, their state, and required resources after stopping Spark.
    actor_states = list_actors(detail=True)
    actor_rows = [
        (a.name, a.state, getattr(a, "required_resources", None))
        for a in actor_states
    ]
    actor_rows.sort(key=lambda x: (str(x[1] or ""), str(x[0] or "")))
    print("Actors (name, state, required_resources):")
    for row in actor_rows:
        print(row)

def _get_ray_node_ip_by_node_id(node_id: str) -> str:
    for n in ray.nodes():
        if n.get("NodeID") == node_id:
            ip = n.get("NodeManagerAddress")
            if ip:
                return ip
    raise AssertionError(f"Cannot resolve node ip for node_id={node_id}. ray.nodes()={ray.nodes()}")


def _get_executor_node_ip(executor_actor_name: str) -> str:
    """Resolve executor node ip via Ray APIs (actor->node_id, node_id->ray.nodes())."""
    actors = list_actors(filters=[("name", "=", executor_actor_name)], detail=True)
    assert len(actors) == 1, f"{executor_actor_name} actor not found or multiple found"
    node_id = getattr(actors[0], "node_id", None) or getattr(actors[0], "nodeId", None)
    assert node_id, f"Missing node_id on actor state for {executor_actor_name}: {actors[0]}"
    return _get_ray_node_ip_by_node_id(node_id)


def _get_single_executor_actor_name() -> str:
    """Find the single RayDP executor actor name in this test."""
    actors = list_actors(detail=True)
    names = [
        a.name for a in actors
        if getattr(a, "name", None)
        and a.name.startswith("raydp-executor-")
        and getattr(a, "state", None) == "ALIVE"
    ]
    assert len(names) == 1, f"Expected exactly one executor actor, got: {names}"
    return names[0]


def test_data_ownership_transfer_with_custom_actor_resources(jdk17_extra_spark_configs):
  """
  Test shutting down Spark worker after data been put
  into Ray object store with data ownership transfer.
  This test should be able to execute till the end without crash as expected.
  """
  
  if ray_client.ray.is_connected():
    pytest.skip("Skip this test if using ray client")

  total_cpu = 5
  cluster = Cluster(
        initialize_head=True,
        head_node_args={
            "num_cpus": 1,
            "resources": {"spark_master": 10},
            "include_dashboard": True,
            "dashboard_port": 8271,
        },
    )
  cluster.add_node(num_cpus=2, resources={"spark_executor": 2})
  cluster.add_node(num_cpus=2, resources={"spark_executor": 2})

  ray.init(address=cluster.address)
            
  from raydp.spark.dataset import spark_dataframe_to_ray_dataset
  import numpy as np

  num_executor = 1
  blockstore_actor_resource_cpu = 1
  app_name = "example"
  blockstore_actor_name = f"{app_name}_BLOCKSTORE_0"

  spark = raydp.init_spark(
    app_name = app_name,
    num_executors = num_executor,
    executor_cores = 1,
    executor_memory = "500M",
    configs={
      **jdk17_extra_spark_configs,
      "spark.ray.raydp_spark_master.actor.resource.spark_master": "1",
      "spark.ray.raydp_spark_master.actor.resource.CPU": "0",
      "spark.ray.raydp_spark_executor.actor.resource.spark_executor": "1",
      "spark.ray.raydp_blockstore.actor.resource.CPU": blockstore_actor_resource_cpu,
      "spark.ray.raydp_blockstore.actor.resource.memory": "100M",
    })

  df_train = gen_test_data(spark)

  # convert data from spark dataframe to ray dataset,
  # and transfer data ownership to dedicated Object Holder (Singleton)
  ds = spark_dataframe_to_ray_dataset(df_train, parallelism=4,
                                      owner=get_raydp_master_owner(spark))

  # display data
  ds.show(5)

  _print_actors()
  # confirm that blockstore actors have been created
  resource_stats = ray.available_resources()
  assert resource_stats['CPU'] == total_cpu \
    - num_executor \
    - blockstore_actor_resource_cpu * num_executor

  # Derive the expected node IP from Ray's node table, via the Spark executor actor placement.
  # This avoids reading the blockstore actor's own state to "discover" its node IP.
  executor_actor_name = _get_single_executor_actor_name()
  expected_node_ip = _get_executor_node_ip(executor_actor_name)

  # release resource by shutting down spark Java process
  raydp.stop_spark(cleanup_data=False)
  ray_gc() # ensure GC kicked in
  time.sleep(3)

  _print_actors()
  
  blockstore_actors = list_actors(filters=[("name", "=", blockstore_actor_name)], detail=True)
  assert len(blockstore_actors) == 1, f"{blockstore_actor_name} actor not found or multiple found"
  actor_state = blockstore_actors[0]
  resources = actor_state.required_resources

  assert resources["memory"] == 100 * 1024 * 1024
  assert resources["CPU"] == blockstore_actor_resource_cpu
  assert resources[f"node:{expected_node_ip}"] == 0.001

  # confirm that resources has been recycled
  resource_stats = ray.available_resources()
  assert resource_stats['CPU'] == total_cpu \
    - blockstore_actor_resource_cpu * num_executor

  # confirm that data is still available from object store!
  # sanity check the dataset is as functional as normal
  assert np.isnan(ds.mean('Age')) is not True

  # final clean up
  raydp.stop_spark()
  time.sleep(3)
  ray.shutdown()
  cluster.shutdown()


def test_custom_ownership_transfer_custom_actor(ray_cluster, jdk17_extra_spark_configs):
  """
  Test shutting down Spark worker after data been put
  into Ray object store with data ownership transfer to custom user actor.
  This test should be able to execute till the end without crash as expected.
  """

  @ray.remote
  class CustomActor(RayDPBlockStoreActorRegistry):
      objects: Any

      def wake(self):
          pass

      def set_objects(self, objects):
          self.objects = objects

  if ray_client.ray.is_connected():
      pytest.skip("Skip this test if using ray client")

  from raydp.spark.dataset import spark_dataframe_to_ray_dataset
  import numpy as np

  num_executor = 1

  spark = raydp.init_spark(
      app_name="example",
      num_executors=num_executor,
      executor_cores=1,
      executor_memory="500M",
      configs=jdk17_extra_spark_configs
  )

  df_train = gen_test_data(spark)

  resource_stats = ray.available_resources()
  cpu_cnt = resource_stats['CPU']

  # create owner
  owner_actor_name = 'owner_actor_name'
  actor = CustomActor.options(name=owner_actor_name).remote()
  # waiting for the actor to be created
  ray.get(actor.wake.remote())

  # convert data from spark dataframe to ray dataset,
  # and transfer data ownership to dedicated Object Holder (Singleton)
  ds = spark_dataframe_to_ray_dataset(df_train, parallelism=4, owner=PartitionObjectsOwner(
      owner_actor_name,
      lambda actor, objects: actor.set_objects.remote(objects)))

  # display data
  ds.show(5)

  # release resource by shutting down spark Java process
  raydp.stop_spark()
  ray_gc()  # ensure GC kicked in
  time.sleep(3)

  # confirm that resources has been recycled
  resource_stats = ray.available_resources()
  assert resource_stats['CPU'] == cpu_cnt + num_executor

  # confirm that data is still available from object store!
  # sanity check the dataset is as functional as normal
  assert np.isnan(ds.mean('Age')) is not True


def test_api_compatibility(ray_cluster, jdk17_extra_spark_configs):
  """
  Test the changes been made are not to break public APIs.
  """

  num_executor = 1

  spark = raydp.init_spark(
    app_name = "test_api_compatibility",
    num_executors = num_executor,
    executor_cores = 1,
    executor_memory = "500M",
    configs=jdk17_extra_spark_configs
    )

  df_train = gen_test_data(spark)

  resource_stats = ray.available_resources()
  cpu_cnt = resource_stats['CPU']

  # check compatibility of ray 1.9.0 API: no data onwership transfer
  ds = ray.data.from_spark(df_train)
  if not ray_client.ray.is_connected():
    ds.show(1)
  ray_gc() # ensure GC kicked in
  time.sleep(3)

  # confirm that resources is still being occupied
  resource_stats = ray.available_resources()
  assert resource_stats['CPU'] == cpu_cnt

  # final clean up
  raydp.stop_spark()

if __name__ == '__main__':
  sys.exit(pytest.main(["-v", __file__]))

  # test_api_compatibility()
  # test_data_ownership_transfer()
  # test_fail_without_data_ownership_transfer()

