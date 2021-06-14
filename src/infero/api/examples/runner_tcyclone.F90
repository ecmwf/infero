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

character(1024) :: model_path
character(1024) :: model_type
character(1024) :: input_path
character(1024) :: yaml_config

type(c_ptr) :: handle

integer :: ss, i, j, ch

integer :: input_dim_0
integer :: input_dim_1
integer :: input_dim_2
integer :: input_dim_3

integer :: output_dim_0
integer :: output_dim_1
integer :: output_dim_2
integer :: output_dim_3

integer :: ios, fu
integer, parameter :: read_unit = 99

real(c_float), allocatable :: it2f(:,:,:,:)
real(c_float), allocatable :: ot2f(:,:,:,:)

! input size [ 1, 200, 200, 17 ]
input_dim_0 = 1
input_dim_1 = 200
input_dim_2 = 200
input_dim_3 = 17

!output size [ 1, 200, 200, 1 ]
output_dim_0 = 1
output_dim_1 = 200
output_dim_2 = 200
output_dim_3 = 1

! CL arguments
CALL get_command_argument(1, model_path)
CALL get_command_argument(2, model_type)
CALL get_command_argument(3, input_path)

! float
allocate( it2f(input_dim_0,input_dim_1,input_dim_2,input_dim_3) )
allocate( ot2f(output_dim_0,output_dim_1,output_dim_2,output_dim_3) )

! read 4D data from sequential CSV (in Fortran order)
open (action='read', file=TRIM(input_path), iostat=ios, newunit=fu)
if (ios /= 0) stop
do ch = 1,input_dim_3
    do j = 1,input_dim_2
        do i = 1,input_dim_1
            do ss = 1,input_dim_0
              read(fu, *) it2f(ss, i, j, ch)
            end do
        end do
    end do
end do

! ML engine config
yaml_config = "---"//NEW_LINE('A') &
  //"  path: "//TRIM(model_path)//NEW_LINE('A') &
  //"  type: "//TRIM(model_type)//c_null_char

! get a inference model handle
handle = infero_create_handle_from_yaml_str(yaml_config)

! open the handle
call infero_open_handle(handle)

! run inference
call infero_inference( handle, it2f, ot2f )

! close and delete the handle
call infero_close_handle( handle )
call infero_delete_handle( handle )


! print output
do ch = 1,output_dim_3
    do j = 1,output_dim_2
        do i = 1,output_dim_1
            do ss = 1,output_dim_0
              print*, "[",ss, ",",i, ",",j, ",",ch, "]", ot2f(ss, i, j, ch)
            end do
        end do
    end do
end do


end program

