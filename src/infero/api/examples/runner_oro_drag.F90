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
character(1024) :: input_path
character(1024) :: yaml_config

! handle of infero model
type(c_ptr) :: handle

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

! Allocate tensors
allocate( it2f(input_imax,input_jmax) )
allocate( ot2f(output_imax,output_jmax) )

! Read data from CSV
open (action='read', file=TRIM(input_path), iostat=ios, newunit=fu)
if (ios /= 0) stop
do i = 1,input_imax
  read(fu, *) (it2f(i, j), j = 1, input_jmax)
end do

! YAML config string
yaml_config = "---"//NEW_LINE('A') &
  //"  path: "//TRIM(model_path)//NEW_LINE('A') &
  //"  type: "//TRIM(model_type)//c_null_char

! 1) get a inference model handle
handle = infero_create_handle_from_yaml_str(yaml_config)

! 2) open the handle
call infero_open_handle(handle)

! 3) run inference
call infero_inference( handle, it2f, ot2f )

! 4) close and delete the handle
call infero_close_handle( handle )
call infero_delete_handle( handle )


! print output
do j = 1, output_jmax
 do i = 1, output_imax
   print*, "[",i,",",j,"]", ot2f(i, j)
 end do
end do

end program

