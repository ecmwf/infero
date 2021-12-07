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

! model of infero model
type(infero_model) :: model

! indexes and Tensor dimensions
integer :: ss, i, j, ch

integer :: input_dim_0
integer :: input_dim_1
integer :: input_dim_2
integer :: input_dim_3

integer :: output_dim_0
integer :: output_dim_1
integer :: output_dim_2
integer :: output_dim_3

integer :: err

real*8 input_sum
real*8 tmp_input
real output_sum

! file IO
integer :: ios, fu
integer, parameter :: read_unit = 99

! input and output tensors
real(c_float), allocatable :: it2f(:,:,:,:)
real(c_float), allocatable :: ot2f(:,:,:,:)

! tcyclone model input size [ 1, 200, 200, 17 ]
input_dim_0 = 1
input_dim_1 = 200
input_dim_2 = 200
input_dim_3 = 17

! tcyclone model output size [ 1, 200, 200, 1 ]
output_dim_0 = 1
output_dim_1 = 200
output_dim_2 = 200
output_dim_3 = 1

! Get CL arguments
CALL get_command_argument(1, model_path)
CALL get_command_argument(2, model_type)
CALL get_command_argument(3, input_path)

! 0) init infero
err = infero_initialise()


! Allocate tensors
allocate( it2f(input_dim_0,input_dim_1,input_dim_2,input_dim_3) )
allocate( ot2f(output_dim_0,output_dim_1,output_dim_2,output_dim_3) )

! Read 4D data from sequential CSV (CSV values are in Fortran order)
input_sum = 0
open (action='read', file=TRIM(input_path), iostat=ios, newunit=fu)
if (ios /= 0) stop
do ch = 1,input_dim_3
    do j = 1,input_dim_2
        do i = 1,input_dim_1
            do ss = 1,input_dim_0
              read(fu, *) tmp_input
              it2f(ss, i, j, ch) = tmp_input
              input_sum = input_sum + tmp_input
            end do
        end do
    end do
end do

! YAML config string
yaml_config = "---"//NEW_LINE('A') &
  //"  path: "//TRIM(model_path)//NEW_LINE('A') &
  //"  type: "//TRIM(model_type)//c_null_char

! get a infero model
err = model%initialise_from_yaml_string(yaml_config)

! un inference
err = infero_inference( model, it2f, ot2f )

! free the model
err = model%free()

! finalise infero library
err = infero_finalise()

! print output tensor (Prediction of Infero model)
output_sum = 0
do ch = 1,output_dim_3
    do j = 1,output_dim_2
        do i = 1,output_dim_1
            do ss = 1,output_dim_0
              print*, "[",ss, ",",i, ",",j, ",",ch, "]", ot2f(ss, i, j, ch)
              output_sum = output_sum + ot2f(ss, i, j, ch)
            end do
        end do
    end do
end do

! print out sum
print* , "input_sum: ", input_sum
print* , "output_sum: ", output_sum

end program

