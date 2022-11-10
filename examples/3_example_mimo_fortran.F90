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
use iso_c_binding, only : c_double, c_int, c_float, c_char, c_null_char, c_ptr

implicit none

! Command line arguments
character(1024) :: model_path
character(1024) :: model_type
character(1024) :: yaml_config

! Fortran array containing input data
character(len=128) :: t1_name
character(len=128) :: t2_name
character(len=128) :: t3_name

! n batches
integer, parameter :: n_batch = 3

! infero_tensor_set: key-value store of input tensors
type(infero_tensor_set) :: iset

! input tensors
real(c_float) :: t1(n_batch,32) = 0
real(c_float) :: t2(n_batch,128) = 0

! output tensor
real(c_float) :: t3(n_batch,1) = 0

! infero_tensor_set: key-value store of output tensors
type(infero_tensor_set) :: oset

! the infero model
type(infero_model) :: model

integer :: i, j, cc


! Get Command line arguments
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

! prepare input tensors for named layers
call infero_check(iset%initialise())
call infero_check(iset%push_tensor(t1, TRIM(t1_name)))
call infero_check(iset%push_tensor(t2, TRIM(t2_name)))

! print the input tensor set
call infero_check(iset%print())

! prepare output tensors for named layers
call infero_check(oset%initialise())

call infero_check(oset%push_tensor(t3, TRIM(t3_name)))
call infero_check(oset%print())

! YAML configuration string string
yaml_config = "---"//NEW_LINE('A') &
  //"  path: "//TRIM(model_path)//NEW_LINE('A') &
  //"  type: "//TRIM(model_type)//c_null_char

! get a inference model model
call infero_check(model%initialise_from_yaml_string(yaml_config))

! run inference
call infero_check(model%infer(iset, oset))

! explicitely request to print stats and config
call infero_check(model%print_statistics())
call infero_check(model%print_config())

! print output
call infero_check(oset%print())

! free tensor sets
call infero_check(iset%free())
call infero_check(oset%free())

! free the model
call infero_check(model%free())

! finalise library
call infero_check(infero_finalise())

end program

