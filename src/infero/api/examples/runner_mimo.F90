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

type(infero_tensor_set) :: iset
type(infero_tensor_set) :: oset

! infero model
type(infero_model) :: model
integer, parameter :: n_batch = 10

character(len=128) :: t1_name
real(c_float) :: t1(n_batch,32) = 0

character(len=128):: t2_name
real(c_float) :: t2(n_batch,128) = 0

character(len=128) :: t3_name
real(c_float) :: t3(n_batch,1) = 0

integer :: i, j, cc


! Get CL arguments
CALL get_command_argument(1, model_path)
CALL get_command_argument(2, model_type)

if ((model_type .eq. "onnx").or.(model_type .eq. "tflite")) then
    t1_name = "input_1"
    t2_name = "input_2"
    t3_name = "dense_6"
else
    t1_name = "serving_default_input_1"
    t2_name = "serving_default_input_2"
    t3_name = "StatefulPartitionedCall"
end if

! init the input tensors
cc=0
do i=1,n_batch
    do j=1,32
        t1(i,j) = cc
        cc = cc + 1
    end do
end do
t1 = t1 / (n_batch*32)

cc=0
do i=1,n_batch
    do j=1,128
        t2(i,j) = cc
        cc = cc + 1
    end do
end do
t2 = t2 / (n_batch*128)


! init infero library
call infero_check(infero_initialise())

! prepare input tensors for named layers
call infero_check(iset%initialise())
call infero_check(iset%push_tensor(t1, TRIM(t1_name)))
call infero_check(iset%push_tensor(t2, TRIM(t2_name)))
call infero_check(iset%print())

! prepare output tensors for named layers
call infero_check(oset%initialise())

call infero_check(oset%push_tensor(t3, TRIM(t3_name)))
call infero_check(oset%print())

! YAML config string
yaml_config = "---"//NEW_LINE('A') &
  //"  path: "//TRIM(model_path)//NEW_LINE('A') &
  //"  type: "//TRIM(model_type)//c_null_char

! get a inference model model
call infero_check(model%initialise_from_yaml_string(yaml_config))

! run inference
do i = 1,20
  call infero_check(model%infer(iset, oset))
end do

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

