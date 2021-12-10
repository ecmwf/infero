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
integer :: err

! infero model
type(infero_model) :: model

character(len=*), parameter :: t1_name = "input_1"
real(c_float) :: t1(1,32) = 1

character(len=*), parameter :: t2_name = "input_2"
real(c_float) :: t2(1,128) = 1

character(len=*), parameter :: t3_name = "dense_6"
real(c_float) :: t3(1,1) = 0


! Get CL arguments
CALL get_command_argument(1, model_path)
CALL get_command_argument(2, model_type)

! init infero library
call infero_check(infero_initialise())

! prepare input tensors for named layers
call infero_check(iset%initialise())
call infero_check(iset%push_tensor(t1, t1_name))
call infero_check(iset%push_tensor(t2, t2_name))
call infero_check(iset%print())

! prepare output tensors for named layers
call infero_check(oset%initialise())
call infero_check(oset%push_tensor(t3, t3_name))
call infero_check(oset%print())

! YAML config string
yaml_config = "---"//NEW_LINE('A') &
  //"  path: "//TRIM(model_path)//NEW_LINE('A') &
  //"  type: "//TRIM(model_type)//c_null_char

! get a inference model model
call infero_check(model%initialise_from_yaml_string(yaml_config))

! run inference
call infero_check(model%infer(iset, oset))

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

