.. _quick_start:

Quick Start
===========

Infero can be easily installed and run in few simple steps:

.. code-block:: bash

   # Download Infero
   cd $HOME
   git clone https://github.com/ecmwf-projects/infero.git

   # Install Infero and all its dependencies
   cd infero/dev && ./1_install_deps.sh && ./2_install_infero.sh

   # Run a ONNX model for inference from a C++ example application
   $HOME/builds/infero/bin/2_example_mimo_cpp $HOME/infero/tests/data/mimo_model/mimo_model.onnx onnx input_1 input_2 dense_6

On successful run, Infero reports an output similar to the one shown here:

.. code-block:: none

   Checking key path
   Checking key type

   ONNX model has: 2 inputs
   Layer [0] input_1 has shape: -1, 32,
   Layer [1] input_2 has shape: -1, 128,
   ONNX model has: 1 outputs
   Layer [0] dense_6 has shape: -1, 1,

   doing inference..
   Tensor(right=0,shape=[3,1,],array=[168172,336345,504516,])

   ========== Infero Model Statistics ==========
   INFERO-STATS: Time to copy/reorder Input  : 2e-06 second (3e-06 second CPU). Updates: 1
   INFERO-STATS: Time to execute inference   : 0.000171 second (0.000172 second CPU). Updates: 1
   INFERO-STATS: Time to copy/reorder Output : 1e-06 second (2e-06 second CPU). Updates: 1
   INFERO-STATS: Total Time                  : 0.000174 second (0.000177 second CPU). Updates: 3


For more information on this and other available examples, see Section :ref:`examples`.

TensorFlow C session options
----------------------------

The ``tf_c`` backend accepts session settings in ``model_config``:

.. code-block:: yaml

   type: tf_c
   path: /path/to/saved_model
   model_config:
     numIntraopThreads: 4
     numInteropThreads: 2
     device: 0

``numIntraopThreads`` controls parallelism within an operation, while
``numInteropThreads`` controls parallelism between operations. Both default to
1; 0 lets TensorFlow choose. Counts of 128 and above are supported, subject to
TensorFlow and system resource limits. A negative ``numInteropThreads`` runs
operations in the calling thread.

``device`` selects a GPU visible to the process and supports multi-digit IDs.
Its default, ``rank``, uses the rank in the default MPI communicator, not the
node-local rank. The selected GPU must exist among the devices visible to that
process; serializing a larger ID does not make additional GPUs available.

