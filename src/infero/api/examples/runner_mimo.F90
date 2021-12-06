/*
 * (C) Copyright 1996- ECMWF.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation
 * nor does it submit to any jurisdiction.
 */

program my_program
use inferof
use iso_c_binding, only : c_double, c_int, c_float, c_char, c_null_char, c_ptr
implicit none

! Command line arguments
character(1024) :: model_path
character(1024) :: model_type
character(1024) :: yaml_config

type(infero_tensor_set) :: it_set
type(infero_tensor_set) :: ot_set
integer :: err

! handle of infero model
type(infero_handle) :: handle

! indexes and Tensor dimensions
integer :: ss, i, j, ch

real*8 input_sum
real*8 tmp_input
real output_sum

character(len=*), parameter :: t1_name = "input_1"
real(c_float) :: t1(1,32) = 1

character(len=*), parameter :: t2_name = "input_2"
real(c_float) :: t2(1,128) = 1

character(len=*), parameter :: t3_name = "dense_6"
real(c_float) :: t3(1,1) = 0


! Get CL arguments
CALL get_command_argument(1, model_path)
CALL get_command_argument(2, model_type)

! prapare input tensors for named layers
err = it_set%initialise()
err = it_set%push_tensor_rank2(t1, t1_name)
err = it_set%push_tensor_rank2(t2, t2_name)
err = it_set%print()

! prapare output tensors for named layers
err = ot_set%initialise()
err = ot_set%push_tensor_rank2(t3, t3_name)
err = ot_set%print()

! YAML config string
yaml_config = "---"//NEW_LINE('A') &
  //"  path: "//TRIM(model_path)//NEW_LINE('A') &
  //"  type: "//TRIM(model_type)//c_null_char

! 0) init infero
err = infero_initialise()

! 1) get a inference model handle
err = handle%from_yaml_string(yaml_config)

! 2) open the handle
err = handle%open()

! 3) run inference
err = handle%infer_mimo(it_set, ot_set)

! 4) close and delete the handle
err = handle%close()

! 5) delete the handle
err = handle%delete()

! print output 
err = ot_set%print()


! delete tensor sets
err = it_set%delete()
err = ot_set%delete()

! 6) finalise library
err = infero_finalise()

end program

