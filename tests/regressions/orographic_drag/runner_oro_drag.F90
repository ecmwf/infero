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

real(c_float), parameter :: tol = 1e-4;

! Command line arguments
character(1024) :: model_path
character(1024) :: model_type
character(1024) :: input_path
character(1024) :: ref_output_path
character(1024) :: yaml_config

! handle of infero model
type(infero_model) :: model

! indexes and Tensor dimensions
integer :: i,j
integer :: input_imax, input_jmax
integer :: output_imax, output_jmax

! file IO
integer :: ios, fu
integer, parameter :: read_unit = 99

! input and output tensors
real(c_float), allocatable :: it2f(:,:)
real(c_float), allocatable :: ot2f(:,:)
real(c_float), allocatable :: ot2f_ref(:,:)

! orographic drag model input size [ 8 , 191 ]
input_imax = 8
input_jmax = 191

! orographic drag model output size [ 8 , 126 ]
output_imax = 8
output_jmax = 126

! Get CL arguments
CALL get_command_argument(1, model_path)
CALL get_command_argument(2, model_type)
CALL get_command_argument(3, input_path)
CALL get_command_argument(4, ref_output_path)

call infero_check(infero_initialise())

! Allocate tensors
allocate( it2f(input_imax,input_jmax) )
allocate( ot2f(output_imax,output_jmax) )
allocate( ot2f_ref(output_imax,output_jmax) )

! Read input data from CSV
open (action='read', file=TRIM(input_path), iostat=ios, newunit=fu)
if (ios /= 0) stop
do i = 1,input_imax
  read(fu, *) (it2f(i, j), j = 1, input_jmax)
end do

! YAML config string
yaml_config = "---"//NEW_LINE('A') &
  //"  path: "//TRIM(model_path)//NEW_LINE('A') &
  //"  type: "//TRIM(model_type)//c_null_char

! get a infero model
call infero_check(model%initialise_from_yaml_string(yaml_config))

! un inference
call infero_check(model%infer(it2f, ot2f ))

! free the model
call infero_check(model%free())

! finalise infero library
call infero_check(infero_finalise())

! Read output reference data from CSV
open (action='read', file=TRIM(ref_output_path), iostat=ios, newunit=fu)
if (ios /= 0) stop
do i = 1,output_imax
  read(fu, *) (ot2f_ref(i, j), j = 1, output_jmax)
end do

! check all elements of the output
do j = 1,output_jmax
    do i = 1,output_imax
      if (abs(ot2f(i,j) - ot2f_ref(i,j)) .gt. tol) then
        write(*,*) "ERROR: output element ",i,j, " (", ot2f(i,j) ,") ", &
        "is different from expected value ", ot2f_ref(i,j)
        stop 1
      end if
    end do
end do


end program

