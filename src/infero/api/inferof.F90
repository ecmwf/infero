module inferof
implicit none
private

public :: infero_inference

interface

  !-----------------------------------------------------------------------------------------
  ! void infero_infer_real64( double data1[], int rank1, int shape1[],  double data2[], int rank2, int shape2[]  );
  ! ( must be defined within `extern "C" { ... }` scope )
  !-----------------------------------------------------------------------------------------
  subroutine infero_infer_real64( data1, rank1, shape1, data2, rank2, shape2 ) &
    & bind(C,name="infero_infer_real64")
    use iso_c_binding, only: c_int, c_ptr, c_double
    real(c_double), dimension(*) :: data1
    integer(c_int), value :: rank1
    integer(c_int), dimension(*) :: shape1
    real(c_double), dimension(*) :: data2
    integer(c_int), value :: rank2
    integer(c_int), dimension(*) :: shape2
  end subroutine

  ! subroutine infero_infer_real32( data1, rank1, shape1, data2, rank2, shape2 ) &
  !   & bind(C,name="infero_infer_real32")
  !   use iso_c_binding, only: c_int, c_ptr, c_float
  !   real(c_float), dimension(*) :: data1
  !   integer(c_int), value :: rank1
  !   integer(c_int), dimension(*) :: shape1
  !   real(c_float), dimension(*) :: data2
  !   integer(c_int), value :: rank2
  !   integer(c_int), dimension(*) :: shape2
  ! end subroutine

end interface

! array views

interface array_view1d
  module procedure array_view1d_real64_r2
  module procedure array_view1d_real64_r3
end interface

! inference

interface infero_inference ! Function overloading
  module procedure infero_inference_real64_rank3_rank2
  ! module procedure call_inference_real32
end interface

contains

function c_loc_real64(x)
  use, intrinsic :: iso_c_binding
  real(c_double), target :: x
  type(c_ptr) :: c_loc_real64
  c_loc_real64 = c_loc(x)
end function


function array_view1d_real64_r2(array) result( view )
  use, intrinsic :: iso_c_binding
  real(c_double), intent(in), target :: array(:,:)
  type(c_ptr) :: array_c_ptr
  real(c_double), pointer :: view(:)  
  nullify(view)
  array_c_ptr = c_loc_real64(array(1,1))
  call c_f_pointer ( array_c_ptr , view , (/size(array)/) )
end function

function array_view1d_real64_r3(array) result( view )
  use, intrinsic :: iso_c_binding
  real(c_double), intent(in), target :: array(:,:,:)
  type(c_ptr) :: array_c_ptr
  real(c_double), pointer :: view(:)
  nullify(view)
  array_c_ptr = c_loc_real64(array(1,1,1))
  call c_f_pointer ( array_c_ptr , view , (/size(array)/) )
end function

subroutine infero_inference_real64_rank3_rank2( array1, array2 )
  use, intrinsic :: iso_c_binding
  real(c_double), intent(inout) :: array1(:,:,:)
  real(c_double), intent(inout) :: array2(:,:)
  integer(c_int) :: shape1(3)
  integer(c_int) :: shape2(2)
  real(c_double), pointer :: data1(:)
  real(c_double), pointer :: data2(:)

  shape1 = shape(array1)
  data1  => array_view1d( array1 )
  shape2 = shape(array2)
  data2  => array_view1d( array2 )

  call infero_infer_real64(data1, size(shape1), shape1, data2, size(shape2), shape2 )
end subroutine


end module

