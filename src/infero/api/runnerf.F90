program my_program
use inferof
use iso_c_binding, only : c_double
implicit none
real(c_double), allocatable :: array_3d(:,:,:)
real(c_double), allocatable :: array_2d(:,:)

allocate( array_3d(4,3,2) )
allocate( array_2d(4,3) )

array_3d(:,:,:) = 2._c_double
array_2d(:,:)   = 1._c_double

call infero_inference( array_3d, array_2d )

end program

