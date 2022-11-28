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

The examples show how to use Infero for a multi-input single-output Machine Learning model. The examples show how to
call Infero API's from C, C++ and Fortran (same model and input data are used for all the cases). The same
output values are expected for the all the examples.

 * *1_example_mimo_c.c*
 * *2_example_mimo_cpp.cc*
 * *3_example_mimo_fortran.F90*


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



Code Explained
----------------

The examples are extensively commented to describe the usage of the API's step-by-step. A brief description
of the main sections from the Fortran example is also reported here below (for the full example, refer
to *3_example_mimo_fortran.F90*).

This section below contains the declarations of the necessary input variables. t1 and t2 are the fortran arrays containing
the data and t1_name and t2_name are the corresponding names. Each tensor must in fact be associated to the name of the
input layer of the machine learning model that receives the tensor data as input.

.. code-block:: fortran

   ! input tensors
   real(c_float) :: t1(n_batch,32) = 0
   real(c_float) :: t2(n_batch,128) = 0

   ! names of input layers
   character(len=128) :: t1_name
   character(len=128) :: t2_name

Infero provides a fortran type called *infero_tensor_set* that is used as a dictionary of type {name: tensor} and can be
used to define the full input interface to the machine learning model.

.. code-block:: fortran

   ! infero_tensor_set: map {name: tensor}
   type(infero_tensor_set) :: iset

Output tensor(s) are declared and arranged similarly to input tensors

.. code-block:: fortran

   ! output tensor
   real(c_float) :: t3(n_batch,1) = 0

   ! names of output layers
   character(len=128) :: t3_name

   ! infero_tensor_set: map {name: tensor}
   type(infero_tensor_set) :: oset

The type for the machine learning model model is called *infero_model*, shown below:

.. code-block:: fortran

   ! the infero model
   type(infero_model) :: model

Input tensors are filled row-wise with dummy values for this example and the *infero_tensor_set* is filled in:

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

   ! prepare input tensors for named layers
   call infero_check(iset%initialise())
   call infero_check(iset%push_tensor(t1, TRIM(t1_name)))
   call infero_check(iset%push_tensor(t2, TRIM(t2_name)))

   ! print the input tensor set
   call infero_check(iset%print())

Same thing is done for the output tensor

.. code-block:: fortran

   ! prepare output tensors for named layers
   call infero_check(oset%initialise())
   call infero_check(oset%push_tensor(t3, TRIM(t3_name)))
   call infero_check(oset%print())


Configure and call infero inference method

.. code-block:: fortran

   ! YAML configuration string string
   yaml_config = "---"//NEW_LINE('A') &
     //"  path: "//TRIM(model_path)//NEW_LINE('A') &
     //"  type: "//TRIM(model_type)//c_null_char

   ! get a inference model model
   call infero_check(model%initialise_from_yaml_string(yaml_config))

   ! run inference
   call infero_check(model%infer(iset, oset))


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

   ! free tensor sets
   call infero_check(iset%free())
   call infero_check(oset%free())

   ! free the model
   call infero_check(model%free())

   ! finalise library
   call infero_check(infero_finalise())
