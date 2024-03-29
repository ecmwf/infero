!
! (C) Copyright 1996- ECMWF.
!
! This software is licensed under the terms of the Apache Licence Version 2.0
! which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
! In applying this licence, ECMWF does not waive the privileges and immunities
! granted to it by virtue of its status as an intergovernmental organisation
! nor does it submit to any jurisdiction.
!

program my_program

use inferof
use fckit_map_module, only : fckit_map
use fckit_tensor_module, only : fckit_tensor_real32
use iso_c_binding, only : c_double, c_int, c_float, c_char, c_null_char, c_ptr

implicit none

! Command line arguments
integer :: arg_count
character(1024) :: model_path
character(1024) :: model_type
character(1024) :: yaml_config

! n batches
integer, parameter :: n_batch = 3

! input tensors
real(c_float) :: t1(n_batch,32) = 0
real(c_float) :: t2(n_batch,128) = 0

! names of input layers
character(len=128) :: t1_name
character(len=128) :: t2_name

! output tensor
real(c_float) :: t3(n_batch,1) = 0

! name of output layer
character(len=128) :: t3_name

! the infero model
type(infero_model) :: model

! auxiliary fckit tensor wrappers
type(fckit_tensor_real32) :: tensor1
type(fckit_tensor_real32) :: tensor2
type(fckit_tensor_real32) :: tensor3

! key/value map for name->tensor
type(fckit_map) :: imap
type(fckit_map) :: omap

integer :: i, j, cc

! Get Command line arguments
arg_count = command_argument_count()
if (arg_count<5) then
   write(*,*) "Error: This example must be invoked as:"
   write(*,*) "<infero-build-path>/bin/3_example_mimo_fortran & 
& <infero-sources-path>/tests/data/mimo_model/mimo_model.onnx &
& onnx  &
& input_1 &
& input_2 &
& dense_6"
   call exit(1)
endif

CALL get_command_argument(1, model_path)
CALL get_command_argument(2, model_type)
CALL get_command_argument(3, t1_name)
CALL get_command_argument(4, t2_name)
CALL get_command_argument(5, t3_name)

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

! wrap output tensor into fckit_tensor
tensor3 = fckit_tensor_real32(t3)

! construct the fckit output map
omap = fckit_map()

! insert entry name+tensor into the output map
call omap%insert(TRIM(t3_name), tensor3%c_ptr())

! YAML configuration string string
yaml_config = "---"//NEW_LINE('A') &
  //"  path: "//TRIM(model_path)//NEW_LINE('A') &
  //"  type: "//TRIM(model_type)//c_null_char

! get a inference model model
call infero_check(model%initialise_from_yaml_string(yaml_config))

! run inference
call infero_check(model%infer(imap, omap))

! explicitely request to print stats and config
call infero_check(model%print_statistics())
call infero_check(model%print_config())

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

end program