.. _examples:

Examples
========


Overview
--------

Infero comes with examples that demonstrate how to use Infero API's for different languages.
Examples can be found in:

 * <infero-source-path>/examples

And when compiled, the corresponding executables are found in:

 * <infero-build-path>/bin

The examples show how to use Infero for a multi-input single-output Machine Learning model from C, C++ and Fortran 
(same model and input data are used for all the cases, so also same output is expected).

 * *1_example_mimo_c.c*
 * *2_example_mimo_cpp.cc*
 * *3_example_mimo_fortran.F90*
 * *4_example_mimo_thread.cc*


Run the examples
----------------

The examples can be run as follows (in this specific case shown below, the onnx backend is used - therefore to run the
example as below, make sure that ONNX backend is installed - see :ref:`build_and_install`).

Note that here below <path/to/mimo/model> is: <path/to/infero/sources>/tests/data/mimo_model

C example:

.. code-block:: console

   cd <path/to/infero/build>
   ./bin/1_example_mimo_c <path/to/mimo/model>/mimo_model.onnx onnx input_1 input_2 dense_6

C++ example:

.. code-block:: console

   cd <path/to/infero/build>
   ./bin/2_example_mimo_cpp <path/to/mimo/model>/mimo_model.onnx onnx input_1 input_2 dense_6

Fortran example:

.. code-block:: console

   cd <path/to/infero/build>
   ./bin/3_example_mimo_fortran <path/to/mimo/model>/mimo_model.onnx onnx input_1 input_2 dense_6

C++ threaded example:

.. code-block:: console

   cd <path/to/infero/build>
   ./bin/4_example_mimo_thread <path/to/mimo/model>/mimo_model.onnx onnx input_1 input_2 dense_6


Code Explained
----------------

The examples are extensively commented to describe the usage of the API's step-by-step. A brief description
of the main sections from the Fortran example is also reported here below (for the full example, refer
to *3_example_mimo_fortran.F90*).

This section below contains the declaration of the necessary input variables. t1 and t2 are the fortran arrays 
containing input data and t1_name and t2_name are the names of the input layers to which the tensors will be assigned.

.. code-block:: fortran

   ! input tensors
   real(c_float) :: t1(n_batch,32) = 0
   real(c_float) :: t2(n_batch,128) = 0

   ! names of input layers
   character(len=128) :: t1_name
   character(len=128) :: t2_name

The association between tensors and names of the corresponding input layers is then made through a 
key/value container of type *fckit_map* (here below the necessary declarations):

.. code-block:: fortran

   ! auxiliary fckit tensor wrappers
   type(fckit_tensor_real32) :: tensor1
   type(fckit_tensor_real32) :: tensor2

   ! key/value map for name->tensor
   type(fckit_map) :: imap

Output tensor(s) are declared and arranged into an *fckit_map* in the same way.

.. code-block:: fortran

   ! output tensor
   real(c_float) :: t3(n_batch,1) = 0

   ! name of output layer
   character(len=128) :: t3_name

   ! auxiliary fckit tensor wrappers
   type(fckit_tensor_real32) :: tensor3

   ! key/value map for name->tensor
   type(fckit_map) :: omap

The type for the machine learning model is called *infero_model*:

.. code-block:: fortran

   ! the infero model
   type(infero_model) :: model

Input tensors are filled row-wise with dummy values for this example and the *fckit_map* is filled in:

.. code-block:: fortran

   ! fill-in the input tensors
   ! Note: dummy values for this example!
   t1(1,:) = 0.1
   t1(2,:) = 0.2
   t1(3,:) = 0.3

   t2(1,:) = 33.0
   t2(2,:) = 66.0
   t2(3,:) = 99.0

   ! init infero library
   call infero_check(infero_initialise())

   ! wrap input tensors into fckit_tensors
   tensor1 = fckit_tensor_real32(t1)
   tensor2 = fckit_tensor_real32(t2)

   ! construct the fckit input map
   imap = fckit_map()

   ! insert entries name+tensor into the input map
   call imap%insert(TRIM(t1_name), tensor1%c_ptr())
   call imap%insert(TRIM(t2_name), tensor2%c_ptr())


Same thing is done for the output tensor

.. code-block:: fortran

   ! wrap output tensor into fckit_tensor
   tensor3 = fckit_tensor_real32(t3)

   ! construct the fckit output map
   omap = fckit_map()

   ! insert entry name+tensor into the output map
   call omap%insert(TRIM(t3_name), tensor3%c_ptr())


Configure and call infero inference method

.. code-block:: fortran

   ! YAML configuration string string
   yaml_config = "---"//NEW_LINE('A') &
     //"  path: "//TRIM(model_path)//NEW_LINE('A') &
     //"  type: "//TRIM(model_type)//c_null_char

   ! get a inference model model
   call infero_check(model%initialise_from_yaml_string(yaml_config))

   ! run inference
   call infero_check(model%infer(imap, omap))


Print inference statistics, configuration and output values

.. code-block:: fortran

   ! explicitely request to print stats and config
   call infero_check(model%print_statistics())
   call infero_check(model%print_config())

   ! print output
   call infero_check(oset%print())


Finally free the allocated memory for the input and output tensor sets and, free the model and
finalise the library itself

.. code-block:: fortran

   ! free the model
   call infero_check(model%free())

   ! finalise fckit objects
   call tensor1%final()
   call tensor2%final()
   call tensor3%final()
   call imap%final()
   call omap%final()

   ! finalise library
   call infero_check(infero_finalise())
