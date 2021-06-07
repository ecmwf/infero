module inferof
implicit none
private

public :: infero_inference
public :: infero_handle_open
public :: infero_handle_close


interface

  !-----------------------------------------------------------------------------------------
  ! void infero_inference_real64( char model_path[], char model_type[],
  !                               double data1[], int rank1, int shape1[],
  !                               double data2[], int rank2, int shape2[]  );
  !
  ! ( must be defined within `extern "C" { ... }` scope )
  !-----------------------------------------------------------------------------------------
  subroutine infero_inference_real64( handle_id, data1, rank1, shape1, data2, rank2, shape2 ) &
    & bind(C,name="infero_inference_double")
    use iso_c_binding, only: c_int, c_ptr, c_double, c_char, c_null_char
    integer(c_int), value :: handle_id
    real(c_double), dimension(*) :: data1
    integer(c_int), value :: rank1
    integer(c_int), dimension(*) :: shape1
    real(c_double), dimension(*) :: data2
    integer(c_int), value :: rank2
    integer(c_int), dimension(*) :: shape2
  end subroutine

  subroutine infero_inference_real32( handle_id, data1, rank1, shape1, data2, rank2, shape2 ) &
    & bind(C,name="infero_inference_float")
    use iso_c_binding, only: c_int, c_ptr, c_float, c_char, c_null_char
    integer(c_int), value :: handle_id
    real(c_float), dimension(*) :: data1
    integer(c_int), value :: rank1
    integer(c_int), dimension(*) :: shape1
    real(c_float), dimension(*) :: data2
    integer(c_int), value :: rank2
    integer(c_int), dimension(*) :: shape2
  end subroutine

end interface

interface
  integer(c_int) function infero_handle_open_interf( config_str ) &
    & bind(C,name="infero_handle_open")
    use iso_c_binding, only: c_char, c_int
    character(c_char) :: config_str
  end function
end interface

interface
  subroutine infero_handle_close_interf( handle_id ) &
    & bind(C,name="infero_handle_close")
    use iso_c_binding, only: c_int
    integer(c_int), value :: handle_id
  end subroutine
end interface


! Array views API

interface array_view1d ! function overloading
  module procedure array_view1d_real32_r2
  module procedure array_view1d_real32_r3
  module procedure array_view1d_real64_r2
  module procedure array_view1d_real64_r3
end interface

! Inference API

interface infero_inference ! function overloading
  module procedure infero_inference_real64_rank2_rank2
  module procedure infero_inference_real32_rank2_rank2
  module procedure infero_inference_real64_rank3_rank2
  module procedure infero_inference_real32_rank3_rank2
end interface

interface infero_handle_open
  module procedure infero_handle_open_fun
end interface

interface infero_handle_close
  module procedure infero_handle_close_fun
end interface

contains

!-----------------------------------------------------------------------------------------------------------------------

integer(c_int) function infero_handle_open_fun( config_str )
  use iso_c_binding, only: c_char, c_int
  character(c_char) :: config_str
  infero_handle_open_fun = infero_handle_open_interf(config_str)
end function

subroutine infero_handle_close_fun( handle_id )
  use iso_c_binding, only: c_int
  integer(c_int), value :: handle_id
  call infero_handle_close_interf( handle_id )
end subroutine

!-----------------------------------------------------------------------------------------------------------------------

!!! Fortran to C addresses

function c_loc_real32(x)
  use, intrinsic :: iso_c_binding
  real(c_float), target :: x
  type(c_ptr) :: c_loc_real32
  c_loc_real32 = c_loc(x)
end function

function c_loc_real64(x)
  use, intrinsic :: iso_c_binding
  real(c_double), target :: x
  type(c_ptr) :: c_loc_real64
  c_loc_real64 = c_loc(x)
end function

!-----------------------------------------------------------------------------------------------------------------------

!!! Fortran Arrays into C pointers

function array_view1d_real32_r2(array) result( view )
  use, intrinsic :: iso_c_binding
  real(c_float), intent(in), target :: array(:,:)
  type(c_ptr) :: array_c_ptr
  real(c_float), pointer :: view(:)  
  nullify(view)
  array_c_ptr = c_loc_real32(array(1,1))
  call c_f_pointer ( array_c_ptr , view , (/size(array)/) )
end function

function array_view1d_real32_r3(array) result( view )
  use, intrinsic :: iso_c_binding
  real(c_float), intent(in), target :: array(:,:,:)
  type(c_ptr) :: array_c_ptr
  real(c_float), pointer :: view(:)
  nullify(view)
  array_c_ptr = c_loc_real32(array(1,1,1))
  call c_f_pointer ( array_c_ptr , view , (/size(array)/) )
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

!-----------------------------------------------------------------------------------------------------------------------

!!! Inference

subroutine infero_inference_real32_rank2_rank2(handle_id, array1, array2 )
  use, intrinsic :: iso_c_binding
  integer(c_int), value :: handle_id
  real(c_float), intent(inout) :: array1(:,:)
  real(c_float), intent(inout) :: array2(:,:)
  integer(c_int) :: shape1(2)
  integer(c_int) :: shape2(2)
  real(c_float), pointer :: data1(:)
  real(c_float), pointer :: data2(:)

  integer(c_int) :: i

  shape1 = shape(array1)
  data1  => array_view1d( array1 )
  shape2 = shape(array2)
  data2  => array_view1d( array2 )

  call infero_inference_real32(handle_id, data1, size(shape1), shape1, data2, size(shape2), shape2 )
end subroutine

subroutine infero_inference_real64_rank2_rank2(handle_id, array1, array2 )
  use, intrinsic :: iso_c_binding
  integer(c_int), value :: handle_id
  real(c_double), intent(inout) :: array1(:,:)
  real(c_double), intent(inout) :: array2(:,:)
  integer(c_int) :: shape1(2)
  integer(c_int) :: shape2(2)
  real(c_double), pointer :: data1(:)
  real(c_double), pointer :: data2(:)

  shape1 = shape(array1)
  data1  => array_view1d( array1 )
  shape2 = shape(array2)
  data2  => array_view1d( array2 )

  call infero_inference_real64(handle_id, data1, size(shape1), shape1, data2, size(shape2), shape2 )
end subroutine

subroutine infero_inference_real32_rank3_rank2(handle_id, array1, array2 )
  use, intrinsic :: iso_c_binding
  integer(c_int), value :: handle_id
  real(c_float), intent(inout) :: array1(:,:,:)
  real(c_float), intent(inout) :: array2(:,:)
  integer(c_int) :: shape1(3)
  integer(c_int) :: shape2(2)
  real(c_float), pointer :: data1(:)
  real(c_float), pointer :: data2(:)

  shape1 = shape(array1)
  data1  => array_view1d( array1 )
  shape2 = shape(array2)
  data2  => array_view1d( array2 )

  call infero_inference_real32(handle_id, data1, size(shape1), shape1, data2, size(shape2), shape2 )
end subroutine

subroutine infero_inference_real64_rank3_rank2(handle_id, array1, array2 )
  use, intrinsic :: iso_c_binding
  integer(c_int), value :: handle_id
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

  call infero_inference_real64(handle_id, data1, size(shape1), shape1, data2, size(shape2), shape2 )
end subroutine


end module

