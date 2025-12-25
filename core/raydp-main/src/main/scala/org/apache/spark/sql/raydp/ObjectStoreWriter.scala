/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.sql.raydp

import com.intel.raydp.shims.SparkShimLoader
import io.ray.api.{ActorHandle, ObjectRef, Ray}
import io.ray.api.PyActorHandle
import io.ray.api.function.PyActorMethod
import io.ray.runtime.AbstractRayRuntime
import io.ray.runtime.config.RayConfig
import java.io.ByteArrayOutputStream
import java.util.{List, Optional, UUID}
import java.util.concurrent.{ConcurrentHashMap, ConcurrentLinkedQueue}
import java.util.function.{Function => JFunction}
import org.apache.arrow.vector.VectorSchemaRoot
import org.apache.arrow.vector.ipc.ArrowStreamWriter
import org.apache.arrow.vector.types.pojo.Schema
import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

import org.apache.spark.{RayDPException, SparkContext, SparkEnv}
import org.apache.spark.deploy.raydp._
import org.apache.spark.executor.RayDPExecutor
import org.apache.spark.network.util.JavaUtils
import org.apache.spark.raydp.{RayDPUtils, RayExecutorUtils}
import org.apache.spark.raydp.SparkOnRayConfigs
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.execution.arrow.ArrowWriter
import org.apache.spark.sql.execution.python.BatchIterator
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.util.ArrowUtils
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.Utils

/**
 * A batch of record that has been wrote into Ray object store.
 * @param ownerAddress the owner address of the ray worker
 * @param objectId the ObjectId for the stored data
 * @param numRecords the number of records for the stored data
 */
case class RecordBatch(
    ownerAddress: Array[Byte],
    objectId: Array[Byte],
    numRecords: Int)

class ObjectStoreWriter(@transient val df: DataFrame) extends Serializable {

  val uuid: UUID = ObjectStoreWriter.dfToId.getOrElseUpdate(df, UUID.randomUUID())

  def writeToRay(
      data: Array[Byte],
      numRecords: Int,
      queue: ObjectRefHolder.Queue,
      ownerName: String): RecordBatch = {

    // Owner-transfer only implementation:
    // - ownerName must always be provided (non-empty) and refer to a Python actor.
    // - JVM never creates/handles Ray ObjectRefs for the dataset blocks.
    // - JVM returns only a per-batch key encoded in RecordBatch.objectId (bytes),
    //   and Python will fetch the real ObjectRefs from the owner actor by key.

    if (ownerName == null || ownerName.isEmpty) {
      throw new RayDPException("ownerName must be set for Spark->Ray conversion.")
    }

    val registryActorOptional = Ray.getActor(ownerName).asInstanceOf[Optional[AnyRef]]
    if (!registryActorOptional.isPresent) {
      throw new RayDPException(s"Blobstore registry actor $ownerName not found.")
    }
    val registryActorHandle: AnyRef = registryActorOptional.get()
    if (!registryActorHandle.isInstanceOf[PyActorHandle]) {
      throw new RayDPException(
        s"Blobstore registry actor $ownerName is not a Python actor.")
    }

    val appName = SparkEnv.get.conf.get("spark.app.name", "raydp")
    val blockStoreActorName =
      ObjectStoreWriter.getBlockStoreActorName(appName, SparkEnv.get.executorId)
    val pyHandle = registryActorHandle.asInstanceOf[PyActorHandle]
    val getActorMethod = PyActorMethod.of(
      "get_or_create_blockstore_actor", classOf[java.lang.Boolean])

    // Get config inside to retain backward compatibility since this is a public API.
    val nodeIp = RayConfig.create().nodeIp
    val cpuOpt =
      SparkEnv.get.conf.getOption(SparkOnRayConfigs.BLOCKSTORE_ACTOR_RESOURCE_CPU)
    val memOpt =
      SparkEnv.get.conf.getOption(SparkOnRayConfigs.BLOCKSTORE_ACTOR_RESOURCE_MEMORY)
    val nodeAffinityOpt =
      SparkEnv.get.conf.getOption(SparkOnRayConfigs.BLOCKSTORE_ACTOR_NODE_AFFINITY_RESOURCE)
    val numCpus = cpuOpt.map(_.toDouble).getOrElse(0.0)
    val memory = memOpt.map(ObjectStoreWriter.parseMemoryBytes).getOrElse(0.0)
    val nodeAffinity = nodeAffinityOpt.map(_.toDouble).getOrElse(0.001)

    pyHandle
      .task(
        getActorMethod,
        blockStoreActorName,
        nodeIp,
        Double.box(numCpus),
        Double.box(memory),
        Double.box(nodeAffinity))
      .remote()
      .get()
    val blockStorageActorHandleOpt =
      Ray.getActor(blockStoreActorName).asInstanceOf[Optional[PyActorHandle]]
    if (!blockStorageActorHandleOpt.isPresent) {
      throw new RayDPException(s"Actor $blockStoreActorName not found when putting dataset block.")
    }
    val blockStorageActorHandle = blockStorageActorHandleOpt.get()

    val batchKey = UUID.randomUUID().toString

    // put_arrow_ipc(batchKey, arrowBytes) -> boolean ack
    val putArrowIPCMethod = PyActorMethod.of("put_arrow_ipc", classOf[java.lang.Boolean])
    blockStorageActorHandle.task(putArrowIPCMethod, batchKey, data).remote().get()

    // RecordBatch payload is an application-level locator (not Ray object metadata):
    // - ownerAddress encodes the BlockStore actor name (UTF-8)
    // - objectId encodes the batch key (UTF-8)
    RecordBatch(blockStoreActorName.getBytes("UTF-8"), batchKey.getBytes("UTF-8"), numRecords)
  }

  /**
   * Save the DataFrame to Ray object store with Apache Arrow format.
   */
  def save(useBatch: Boolean, ownerName: String): List[RecordBatch] = {
    val conf = df.queryExecution.sparkSession.sessionState.conf
    val timeZoneId = conf.getConf(SQLConf.SESSION_LOCAL_TIMEZONE)
    var batchSize = conf.getConf(SQLConf.ARROW_EXECUTION_MAX_RECORDS_PER_BATCH)
    if (!useBatch) {
      batchSize = 0
    }
    val schema = df.schema

    val objectIds = df.queryExecution.toRdd.mapPartitions{ iter =>
      val queue = ObjectRefHolder.getQueue(uuid)

      // DO NOT use iter.grouped(). See BatchIterator.
      val batchIter = if (batchSize > 0) {
        new BatchIterator(iter, batchSize)
      } else {
        Iterator(iter)
      }

      val arrowSchema = SparkShimLoader.getSparkShims.toArrowSchema(schema, timeZoneId)
      val allocator = ArrowUtils.rootAllocator.newChildAllocator(
        s"ray object store writer", 0, Long.MaxValue)
      val root = VectorSchemaRoot.create(arrowSchema, allocator)
      val results = new ArrayBuffer[RecordBatch]()

      val byteOut = new ByteArrayOutputStream()
      val arrowWriter = ArrowWriter.create(root)
      var numRecords: Int = 0

      Utils.tryWithSafeFinally {
        while (batchIter.hasNext) {
          // reset the state
          numRecords = 0
          byteOut.reset()
          arrowWriter.reset()

          // write out the schema meta data
          val writer = new ArrowStreamWriter(root, null, byteOut)
          writer.start()

          // get the next record batch
          val nextBatch = batchIter.next()

          while (nextBatch.hasNext) {
            numRecords += 1
            arrowWriter.write(nextBatch.next())
          }

          // set the write record count
          arrowWriter.finish()
          // write out the record batch to the underlying out
          writer.writeBatch()

          // get the wrote ByteArray and save to Ray ObjectStore
          val byteArray = byteOut.toByteArray
          results += writeToRay(byteArray, numRecords, queue, ownerName)
          // end writes footer to the output stream and doesn't clean any resources.
          // It could throw exception if the output stream is closed, so it should be
          // in the try block.
          writer.end()
        }
        arrowWriter.reset()
        byteOut.close()
      } {
        // If we close root and allocator in TaskCompletionListener, there could be a race
        // condition where the writer thread keeps writing to the VectorSchemaRoot while
        // it's being closed by the TaskCompletion listener.
        // Closing root and allocator here is cleaner because root and allocator is owned
        // by the writer thread and is only visible to the writer thread.
        //
        // If the writer thread is interrupted by TaskCompletionListener, it should either
        // (1) in the try block, in which case it will get an InterruptedException when
        // performing io, and goes into the finally block or (2) in the finally block,
        // in which case it will ignore the interruption and close the resources.

        root.close()
        allocator.close()
      }

      results.toIterator
    }.collect()
    objectIds.toSeq.asJava
  }

  /**
   * For test.
   */
  def getRandomRef(): List[_] = {

    df.queryExecution.toRdd.mapPartitions { _ =>
      Iterator(ObjectRefHolder.getRandom(uuid))
    }.collect().toSeq.asJava
  }

  def clean(): Unit = {
    ObjectStoreWriter.dfToId.remove(df)
    ObjectRefHolder.removeQueue(uuid)
  }

}

object ObjectStoreWriter {
  val dfToId = new mutable.HashMap[DataFrame, UUID]()
  var driverAgent: RayDPDriverAgent = _
  var driverAgentUrl: String = _
  var address: Array[Byte] = null

  def connectToRay(): Unit = {
    if (!Ray.isInitialized) {
      Ray.init()
      // restore log level to WARN since it's inside Spark driver
      SparkContext.getOrCreate().setLogLevel("WARN")
      driverAgent = new RayDPDriverAgent()
      driverAgentUrl = driverAgent.getDriverAgentEndpointUrl
    }
  }

  private def parseMemoryBytes(value: String): Double = {
    if (value == null || value.isEmpty) {
      0.0
    } else {
      // Spark parser supports both plain numbers (bytes) and strings like "100M", "2g".
      JavaUtils.byteStringAsBytes(value).toDouble
    }
  }

  private def sanitizeActorName(name: String): String = {
    if (name == null || name.isEmpty) {
      "raydp"
    } else {
      // Ray named actor names should be reasonably simple; normalize to [A-Za-z0-9_].
      name.replaceAll("[^A-Za-z0-9_]", "_")
    }
  }

  private[spark] def getBlockStoreActorName(appName: String, executorId: String): String = {
    val safeAppName = sanitizeActorName(appName)
    s"${safeAppName}_BLOCKSTORE_${executorId}"
  }

  def getAddress(): Array[Byte] = {
    if (address == null) {
      val objectRef = Ray.put(1)
      val objectRefImpl = RayDPUtils.convert(objectRef)
      val objectId = objectRefImpl.getId
      val runtime = Ray.internal.asInstanceOf[AbstractRayRuntime]
      address = runtime.getObjectStore.getOwnershipInfo(objectId)
    }
    address
  }

  def toArrowSchema(df: DataFrame): Schema = {
    val conf = df.queryExecution.sparkSession.sessionState.conf
    val timeZoneId = conf.getConf(SQLConf.SESSION_LOCAL_TIMEZONE)
    SparkShimLoader.getSparkShims.toArrowSchema(df.schema, timeZoneId)
  }

  @deprecated
  def fromSparkRDD(df: DataFrame, storageLevel: StorageLevel): Array[Array[Byte]] = {
    if (!Ray.isInitialized) {
      throw new RayDPException(
        "Not yet connected to Ray! Please set fault_tolerant_mode=True when starting RayDP.")
    }
    val uuid = dfToId.getOrElseUpdate(df, UUID.randomUUID())
    val queue = ObjectRefHolder.getQueue(uuid)
    val rdd = df.toArrowBatchRdd
    rdd.persist(storageLevel)
    rdd.count()
    var executorIds = df.sqlContext.sparkContext.getExecutorIds.toArray
    val numExecutors = executorIds.length
    val appMasterHandle = Ray.getActor(RayAppMaster.ACTOR_NAME)
                             .get.asInstanceOf[ActorHandle[RayAppMaster]]
    val restartedExecutors = RayAppMasterUtils.getRestartedExecutors(appMasterHandle)
    // Check if there is any restarted executors
    if (!restartedExecutors.isEmpty) {
      // If present, need to use the old id to find ray actors
      for (i <- 0 until numExecutors) {
        if (restartedExecutors.containsKey(executorIds(i))) {
          val oldId = restartedExecutors.get(executorIds(i))
          executorIds(i) = oldId
        }
      }
    }
    val schema = ObjectStoreWriter.toArrowSchema(df).toJson
    val numPartitions = rdd.getNumPartitions
    val results = new Array[Array[Byte]](numPartitions)
    val refs = new Array[ObjectRef[Array[Byte]]](numPartitions)
    val handles = executorIds.map {id =>
      Ray.getActor("raydp-executor-" + id)
         .get
         .asInstanceOf[ActorHandle[RayDPExecutor]]
    }
    val handlesMap = (executorIds zip handles).toMap
    val locations = RayExecutorUtils.getBlockLocations(
        handles(0), rdd.id, numPartitions)
    for (i <- 0 until numPartitions) {
      // TODO use getPreferredLocs, but we don't have a host ip to actor table now
      refs(i) = RayExecutorUtils.getRDDPartition(
          handlesMap(locations(i)), rdd.id, i, schema, driverAgentUrl)
      queue.add(refs(i))
    }
    for (i <- 0 until numPartitions) {
      results(i) = RayDPUtils.convert(refs(i)).getId.getBytes
    }
    results
  }

  /**
   * Recoverable Spark->Ray Dataset conversion without Ray private APIs:
   * persist Arrow batches in Spark, then push cached partitions into Python BlockStore actors
   * (owned by Python) via the given registry actor.
   *
   * This returns application-level locators in RecordBatch:
   * - ownerAddress: UTF-8 BlockStore actor name
   * - objectId: UTF-8 batch key
   */
  def fromSparkRDDToBlockStore(
      df: DataFrame,
      storageLevel: StorageLevel,
      registryActorName: String): List[RecordBatch] = {
    if (!Ray.isInitialized) {
      throw new RayDPException(
        "Not yet connected to Ray! Please set fault_tolerant_mode=True when starting RayDP.")
    }
    if (registryActorName == null || registryActorName.isEmpty) {
      throw new RayDPException("registryActorName must be set for recoverable conversion.")
    }

    val rdd = df.toArrowBatchRdd
    rdd.persist(storageLevel)
    rdd.count()

    val executorIds = df.sqlContext.sparkContext.getExecutorIds.toArray
    val numExecutors = executorIds.length
    val appMasterHandle = Ray.getActor(RayAppMaster.ACTOR_NAME)
                             .get.asInstanceOf[ActorHandle[RayAppMaster]]
    val restartedExecutors = RayAppMasterUtils.getRestartedExecutors(appMasterHandle)
    if (!restartedExecutors.isEmpty) {
      for (i <- 0 until numExecutors) {
        if (restartedExecutors.containsKey(executorIds(i))) {
          val oldId = restartedExecutors.get(executorIds(i))
          executorIds(i) = oldId
        }
      }
    }

    val sparkConf = df.sqlContext.sparkContext.getConf
    val appName = sparkConf.get("spark.app.name", "raydp")

    val cpuOpt = sparkConf.getOption(SparkOnRayConfigs.BLOCKSTORE_ACTOR_RESOURCE_CPU)
    val memOpt = sparkConf.getOption(SparkOnRayConfigs.BLOCKSTORE_ACTOR_RESOURCE_MEMORY)
    val nodeAffinityOpt =
      sparkConf.getOption(SparkOnRayConfigs.BLOCKSTORE_ACTOR_NODE_AFFINITY_RESOURCE)
    val numCpus = cpuOpt.map(_.toDouble).getOrElse(0.0)
    val memory = memOpt.map(parseMemoryBytes).getOrElse(0.0)
    val nodeAffinity = nodeAffinityOpt.map(_.toDouble).getOrElse(0.001)

    val schema = ObjectStoreWriter.toArrowSchema(df).toJson
    val numPartitions = rdd.getNumPartitions
    val handles = executorIds.map { id =>
      Ray.getActor("raydp-executor-" + id)
         .get
         .asInstanceOf[ActorHandle[RayDPExecutor]]
    }
    val handlesMap = (executorIds zip handles).toMap
    val locations = RayExecutorUtils.getBlockLocations(handles(0), rdd.id, numPartitions)

    val acks = new Array[ObjectRef[java.lang.Boolean]](numPartitions)
    val owners = new Array[String](numPartitions)
    val keys = new Array[String](numPartitions)
    for (i <- 0 until numPartitions) {
      val executorId = locations(i)
      val blockStoreActorName = getBlockStoreActorName(appName, executorId)
      val batchKey = s"rdd-${rdd.id}-$i-${UUID.randomUUID().toString}"
      owners(i) = blockStoreActorName
      keys(i) = batchKey
      acks(i) = RayExecutorUtils.putRDDPartitionToBlockStoreViaRegistry(
        handlesMap(executorId),
        rdd.id,
        i,
        schema,
        driverAgentUrl,
        registryActorName,
        blockStoreActorName,
        batchKey,
        numCpus,
        memory,
        nodeAffinity
      )
    }
    val results = new Array[RecordBatch](numPartitions)
    for (i <- 0 until numPartitions) {
      acks(i).get()
      results(i) = RecordBatch(owners(i).getBytes("UTF-8"), keys(i).getBytes("UTF-8"), 0)
    }
    results.toSeq.asJava
  }

}

object ObjectRefHolder {
  type Queue = ConcurrentLinkedQueue[ObjectRef[_]]
  private val dfToQueue = new ConcurrentHashMap[UUID, Queue]()

  def getQueue(df: UUID): Queue = {
    dfToQueue.computeIfAbsent(df, new JFunction[UUID, Queue] {
      override def apply(v1: UUID): Queue = {
        new Queue()
      }
    })
  }

  @inline
  def checkQueueExists(df: UUID): Queue = {
    val queue = dfToQueue.get(df)
    if (queue == null) {
      throw new RuntimeException("The DataFrame does not exist")
    }
    queue
  }

  def getQueueSize(df: UUID): Int = {
    val queue = checkQueueExists(df)
    queue.size()
  }

  def getRandom(df: UUID): Any = {
    val queue = checkQueueExists(df)
    val ref = RayDPUtils.convert(queue.peek())
    ref.get()
  }

  def removeQueue(df: UUID): Unit = {
    dfToQueue.remove(df)
  }

  def clean(): Unit = {
    dfToQueue.clear()
  }
}
