program my_program
use inferof
use iso_c_binding, only : c_double, c_float
implicit none

real(c_float), allocatable :: tensor3f(:,:,:)
real(c_float), allocatable :: tensor2f(:,:)
real(c_double), allocatable :: tensor3d(:,:,:)
real(c_double), allocatable :: tensor2d(:,:)

allocate( tensor3f(4,3,2) )
allocate( tensor2f(4,3) )

tensor3f(:,:,:) = 2._c_float
tensor2f(:,:)   = 1._c_float

call infero_inference( tensor3f, tensor2f )

allocate( tensor3d(4,3,2) )
allocate( tensor2d(4,3) )

tensor3d(:,:,:) = 2._c_double
tensor2d(:,:)   = 1._c_double

call infero_inference( tensor3d, tensor2d )

end program

