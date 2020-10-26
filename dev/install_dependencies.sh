#!/usr/bin/env bash

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

set -ex

CURRENT_DIR="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
DIST_PATH=${CURRENT_DIR}/../dist/

pip install torch==1.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install tensorflow==2.0.0

# build and install pyspark
${CURRENT_DIR}/build_pyspark_with_patch.sh
pip install ${DIST_PATH}/pyspark-*
export SPARK_HOME=${DIST_PATH}/spark

# build and install ray
${CURRENT_DIR}/build_ray_with_patch.sh
pip install ${DIST_PATH}/ray-0.8.7-*

set +ex