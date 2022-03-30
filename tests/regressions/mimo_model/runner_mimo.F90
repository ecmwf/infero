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

real(c_float), parameter :: tol = 1e-6;

! Command line arguments
character(1024) :: model_path
character(1024) :: model_type
character(1024) :: yaml_config
character(len=128) :: t1_name
character(len=128) :: t2_name
character(len=128) :: t3_name

integer :: i, j, cc

integer, parameter :: n_batch = 10
real(c_float) :: t1(n_batch,32) = 0
real(c_float) :: t2(n_batch,128) = 0
real(c_float) :: t3(n_batch,1) = 0
real(c_float) :: expected_output(n_batch) = (/ &
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

type(infero_tensor_set) :: iset
type(infero_tensor_set) :: oset

type(infero_model) :: model


! Get CL arguments
CALL get_command_argument(1, model_path)
CALL get_command_argument(2, model_type)
CALL get_command_argument(3, t1_name)
CALL get_command_argument(4, t2_name)
CALL get_command_argument(5, t3_name)

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

! check all elements of the output
do i = 1,n_batch

  if (abs(t3(i,1) - expected_output(i)) .gt. tol) then
    write(*,*) "ERROR: output element ",i, " (", t3(i,1) ,") ", &
    "is different from expected value ", expected_output(i)
    stop 1
  end if

end do

! free tensor sets
call infero_check(iset%free())
call infero_check(oset%free())

! free the model
call infero_check(model%free())

! finalise library
call infero_check(infero_finalise())

end program

