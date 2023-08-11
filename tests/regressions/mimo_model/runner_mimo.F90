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

real(c_float) :: tol = 1e-2;
integer, parameter :: n_inference_reps = 10

! Command line arguments
character(1024) :: model_path
character(1024) :: model_type
character(1024) :: yaml_config
character(1024) :: tol_str

character(len=128) :: t1_name
character(len=128) :: t2_name
character(len=128) :: t3_name

type(fckit_tensor_real32) :: tensor1
type(fckit_tensor_real32) :: tensor2
type(fckit_tensor_real32) :: tensor3

integer :: i, j, cc
integer :: argc

integer, parameter :: n_batch_i = 10
integer, parameter :: n_batch = 256
real(c_float) :: t1(n_batch,32) = 0
real(c_float) :: t2(n_batch,128) = 0
real(c_float) :: t3(n_batch,1) = 0
real(c_float) :: expected_output(n_batch_i) = (/ &
                                             253.61697,&
                                             764.88446,&
                                             1276.1512,&
                                             1787.4171,&
                                             2298.686,&
                                             2809.9534,&
                                             3321.216,&
                                             3832.4849,&
                                             4343.7505,&
                                             4855.0225 /)

type(fckit_map) :: imap
type(fckit_map) :: omap

type(infero_model) :: model


! Get CL arguments
CALL get_command_argument(1, model_path)
CALL get_command_argument(2, model_type)
CALL get_command_argument(3, t1_name)
CALL get_command_argument(4, t2_name)
CALL get_command_argument(5, t3_name)

argc = command_argument_count()
if (argc>5) then
   call get_command_argument(6, tol_str)
   read(tol_str,*) tol
   write(*,*) "Tolerance set to ", tol
endif

! init the input tensors
cc=0
do i=1,n_batch
   if (mod(i-1,n_batch_i) == 0) then
      cc = 0
   end if
   do j=1,32
      t1(i,j) = cc
      cc = cc + 1
   end do
end do
t1 = t1 / (n_batch_i*32)

cc=0
do i=1,n_batch
   if (mod(i-1,n_batch_i) == 0) then
      cc = 0
   end if
    do j=1,128
        t2(i,j) = cc
        cc = cc + 1
    end do
end do
t2 = t2 / (n_batch_i*128)


! init infero library
call infero_check(infero_initialise())

! prepare input tensors for named layers
tensor1 = fckit_tensor_real32(t1)
tensor2 = fckit_tensor_real32(t2)

imap = fckit_map()
call imap%insert(TRIM(t1_name), tensor1%c_ptr())
call imap%insert(TRIM(t2_name), tensor2%c_ptr())

! prepare output tensors for named layers
tensor3 = fckit_tensor_real32(t3)
omap = fckit_map()
call omap%insert(TRIM(t3_name), tensor3%c_ptr())

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
do j = 1,n_batch
   i = mod(j-1,n_batch_i)+1
   if (abs(t3(j,1) - expected_output(i)) .gt. tol) then
      write(*,*) "ERROR: output element ",j, " (", t3(j,1) ,") ", &
           "is different from expected value ", expected_output(i)
      stop 1
  end if

end do

! free the model
call infero_check(model%free())

! finalise library
call infero_check(infero_finalise())

end program

