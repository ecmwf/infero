program my_program
use inferof
use iso_c_binding, only : c_double, c_float, c_char, c_null_char
implicit none

character(1024) :: model_path
character(1024) :: model_type
character(1024) :: input_path

integer :: i,j
integer :: input_imax, input_jmax
integer :: output_imax, output_jmax
integer :: ios, fu
integer, parameter :: read_unit = 99

real(c_float), allocatable :: it2f(:,:)
real(c_float), allocatable :: ot2f(:,:)

! input size [ 8 , 191 ]
input_imax = 8
input_jmax = 191

!output size [ 8 , 126 ]
output_imax = 8
output_jmax = 126

! CL arguments
CALL get_command_argument(1, model_path)
CALL get_command_argument(2, model_type)
CALL get_command_argument(3, input_path)

! float
allocate( it2f(input_imax,input_jmax) )
allocate( ot2f(output_imax,output_jmax) )

! read data from CSV
open (action='read', file=TRIM(input_path), iostat=ios, newunit=fu)
if (ios /= 0) stop
do i = 1,input_imax
  read(fu, *) (it2f(i, j), j = 1, input_jmax)
end do

! call inference
call infero_inference(model_path//c_null_char, &
                      model_type//c_null_char, &
                      it2f, ot2f )

! print output
do j = 1, output_jmax
 do i = 1, output_imax
   print*, "[",i,",",j,"]", ot2f(i, j)
 end do
end do

end program

