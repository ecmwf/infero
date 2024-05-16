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
  use iso_c_binding, only : c_float, c_null_char, c_ptr
  
  implicit none
  
  real(c_float) :: tol = 1e-2;
  integer, parameter :: n_inference_reps = 2
  
  ! Command line arguments
  character(1024) :: model_path
  character(1024) :: model_type
  character(1024) :: yaml_config
  character(1024) :: tol_str
  
  character(len=128) :: t1_name
  character(len=128) :: t2_name
  character(len=128) :: t3_name
  character(len=128) :: t4_name
  
  ! input tensors
  type(fckit_tensor_real32) :: tensor1
  type(fckit_tensor_real32) :: tensor2

  ! output tensors 
  type(fckit_tensor_real32) :: tensor3
  type(fckit_tensor_real32) :: tensor4
  
  integer :: i, j, cc
  integer :: argc

  integer, parameter :: tensor1_size = 10
  integer, parameter :: tensor2_size = 5
  
  real(c_float) :: t1(1,10) = 0
  real(c_float) :: t2(1,5) = 0
  real(c_float) :: t3(1,1) = 0
  real(c_float) :: t4(1,1) = 0
  real(c_float) :: expected_output(2) = (/ 0.6535, 0.5611 /)
  
  type(fckit_map) :: imap
  type(fckit_map) :: omap
  
  type(infero_model) :: model
  
  
  ! Get CL arguments
  CALL get_command_argument(1, model_path)
  CALL get_command_argument(2, model_type)

  t1_name = "input_1"
  t2_name = "input_2"
  t3_name = "output_1"
  t4_name = "output_2"
  
  argc = command_argument_count()
  if (argc>5) then
     call get_command_argument(6, tol_str)
     read(tol_str,*) tol
     write(*,*) "Tolerance set to ", tol
  endif
  
  ! init infero library
  call infero_check(infero_initialise())


  ! reshape input tensors
  t1 = reshape( (/0,1,2,3,4,5,6,7,8,9/), shape(t1))
  t2 = reshape( (/0,1,2,3,4/), shape(t2))
  
  ! prepare input tensors for named layers
  tensor1 = fckit_tensor_real32(t1)
  tensor2 = fckit_tensor_real32(t2)
  
  imap = fckit_map()
  call imap%insert(TRIM(t1_name), tensor1%c_ptr())
  call imap%insert(TRIM(t2_name), tensor2%c_ptr())
  
  ! prepare output tensors for named layers
  tensor3 = fckit_tensor_real32(t3)
  tensor4 = fckit_tensor_real32(t4)
  omap = fckit_map()
  call omap%insert(TRIM(t3_name), tensor3%c_ptr())
  call omap%insert(TRIM(t4_name), tensor4%c_ptr())
  
  ! YAML config string
  yaml_config = "---"//NEW_LINE('A') &
    //"  path: "//TRIM(model_path)//NEW_LINE('A') &
    //"  type: "//TRIM(model_type)//c_null_char
  
  ! get a inference model model
  call infero_check(model%initialise_from_yaml_string(yaml_config))
  
  ! run inference
  do i = 1, n_inference_reps
    call infero_check(model%infer(imap, omap))
  end do
  
  ! explicitely request to print stats and config
  call infero_check(model%print_statistics())
  call infero_check(model%print_config())
  
  ! check all elements of the output
  write(*,*) "output tensor 1 ", t3(1,1)
  write(*,*) "output tensor 2 ", t4(1,1)

  if (abs(t3(1,1) - expected_output(1)) .gt. tol) then
     write(*,*) "ERROR: output element ", t3(1,1), "is different from expected value ", expected_output(1)
     stop 1
  end if

  if (abs(t4(1,1) - expected_output(2)) .gt. tol) then
     write(*,*) "ERROR: output element ", t4(1,1), "is different from expected value ", expected_output(2)
     stop 1
  end if
  
  ! free the model
  call infero_check(model%free())
  
  ! finalise library
  call infero_check(infero_finalise())
  
  end program
  
  