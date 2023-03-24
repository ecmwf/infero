!
! (C) Copyright 1996- ECMWF.
!
! This software is licensed under the terms of the Apache Licence Version 2.0
! which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
! In applying this licence, ECMWF does not waive the privileges and immunities
! granted to it by virtue of its status as an intergovernmental organisation
! nor does it submit to any jurisdiction.
!

module inferof

  use, intrinsic :: iso_c_binding

  use fckit_array_module
  use fckit_c_interop_module
  use fckit_map_module
  use fckit_main_module
  
implicit none


! Error values
integer, public, parameter :: INFERO_SUCCESS = 0
integer, public, parameter :: INFERO_ERROR_GENERAL_EXCEPTION = 1
integer, public, parameter :: INFERO_ERROR_UNKNOWN_EXCEPTION = 2


private

! --------- Infero model (Infero "C"-handle wrapper)
type infero_model
  type(c_ptr) :: impl = c_null_ptr
  logical :: is_finalised = .false.
contains
  procedure :: initialise_from_yaml_string => infero_create_handle_from_yaml_string
  procedure :: initialise_from_yaml_file => infero_create_handle_from_yaml_file

  procedure :: infer_mimo => infer_from_map

  procedure :: infero_inference_r2_r2_f => infero_inference_real32_rank2_rank2
  procedure :: infero_inference_r2_r2_d => infero_inference_real64_rank2_rank2    
  procedure :: infero_inference_r3_r2_f => infero_inference_real32_rank3_rank2
  procedure :: infero_inference_r3_r2_d => infero_inference_real64_rank3_rank2  
  procedure :: infero_inference_r4_r4_f => infero_inference_real32_rank4_rank4
  procedure :: infero_inference_r4_r4_d => infero_inference_real64_rank4_rank4

  generic   :: infer => infer_mimo, &
                        infero_inference_r2_r2_f, &
                        infero_inference_r2_r2_d, &
                        infero_inference_r3_r2_f, &
                        infero_inference_r3_r2_d, &
                        infero_inference_r4_r4_f, &
                        infero_inference_r4_r4_d

  procedure :: print_statistics => infero_print_statistics
  procedure :: print_config => infero_print_config
  procedure :: free => infero_free_handle

#ifdef HAVE_FINAL
  final :: infero_free_handle_sub
#endif

end type

! ---------  public interface
public :: infero_initialise
public :: infero_finalise

public :: infero_check
public :: infero_error_string

public :: infero_model

interface

!===========================================================================================
!===========================================================================================
!===========================================================================================

  !-----------------------------------------------------------------------------------------
  ! int infero_inference_float(infero_model_t* h, 
  !                            int rank1, 
  !                            const float data1[], 
  !                            const int shape1[], 
  !                            int layout1,
  !                            int rank2,
  !                            float data2[], 
  !                            const int shape2[],
  !                            int layout2);
  !
  ! ( must be defined within `extern "C" { ... }` scope )
  !-----------------------------------------------------------------------------------------
  function infero_inference_real32( handle_impl, rank1, data1, shape1, layout1, rank2, data2, shape2, layout2 ) result(err) &
    & bind(C,name="infero_inference_float")
    use iso_c_binding, only: c_int, c_ptr, c_float, c_char, c_null_char
    
    type(c_ptr), intent(in), value :: handle_impl
    
    integer(c_int), value :: rank1
    real(c_float), dimension(*) :: data1    
    integer(c_int), dimension(*) :: shape1
    integer(c_int), value :: layout1

    integer(c_int), value :: rank2
    real(c_float), dimension(*) :: data2    
    integer(c_int), dimension(*) :: shape2
    integer(c_int), value :: layout2

    integer(c_int) :: err
  end function

  !-----------------------------------------------------------------------------------------
  ! int infero_inference_double(infero_model_t* h, 
  !                             int rank1, 
  !                             const double data1[], 
  !                             const int shape1[], 
  !                             int layout1,
  !                             int rank2,
  !                             double data2[], 
  !                             const int shape2[],
  !                             int layout2);
  !-----------------------------------------------------------------------------------------
  function infero_inference_real64( handle_impl, rank1, data1, shape1, layout1, rank2, data2, shape2, layout2 ) result(err) &
    & bind(C,name="infero_inference_double")
    use iso_c_binding, only: c_int, c_ptr, c_double, c_char, c_null_char

    type(c_ptr), intent(in), value :: handle_impl
    
    integer(c_int), value :: rank1
    real(c_double), dimension(*) :: data1        
    integer(c_int), dimension(*) :: shape1
    integer(c_int), value :: layout1
        
    integer(c_int), value :: rank2
    real(c_double), dimension(*) :: data2
    integer(c_int), dimension(*) :: shape2
    integer(c_int), value :: layout2

    integer(c_int) :: err
  end function

  function infero_create_handle_from_yaml_str_interf( config_str, handle_impl ) result(err) &
    & bind(C,name="infero_create_handle_from_yaml_str")
    use iso_c_binding, only: c_char, c_int, c_ptr
    character(c_char) :: config_str
    type(c_ptr), intent(out) :: handle_impl
    integer(c_int) :: err
  end function

  function infero_create_handle_from_yaml_file_interf( config_str, handle_impl ) result(err) &
    & bind(C,name="infero_create_handle_from_yaml_file")
    use iso_c_binding, only: c_char, c_int, c_ptr
    character(c_char) :: config_str
    type(c_ptr), intent(out) :: handle_impl
    integer(c_int) :: err
  end function

  function infero_open_handle_interf( handle_impl ) result(err) &
    & bind(C,name="infero_open_handle")
    use iso_c_binding, only: c_int, c_ptr
    type(c_ptr), intent(in), value :: handle_impl
    integer(c_int) :: err
  end function

  function infero_close_handle_interf( handle_impl ) result(err) &
    & bind(C,name="infero_close_handle")
    use iso_c_binding, only: c_int, c_ptr
    type(c_ptr), intent(in), value :: handle_impl
    integer(c_int) :: err
  end function

  function infero_delete_handle_interf( handle_impl ) result(err) &
    & bind(C,name="infero_delete_handle")
    use iso_c_binding, only: c_int, c_ptr
    type(c_ptr), intent(in), value :: handle_impl
    integer(c_int) :: err
  end function

  function infer_from_map_interf( handle_impl, imap_ptr, omap_ptr ) result(err) &
    & bind(C,name="infero_inference_float_map")
    use iso_c_binding    
    type(c_ptr), intent(in), value :: handle_impl
    type(c_ptr), intent(in), value :: imap_ptr
    type(c_ptr), intent(in), value :: omap_ptr
    integer(c_int) :: err    
  end function

  function infero_print_statistics_interf( handle_impl ) result(err) &
    & bind(C,name="infero_print_statistics")
    use iso_c_binding
    type(c_ptr), intent(in), value :: handle_impl
    integer(c_int) :: err
  end function

function infero_print_config_interf( handle_impl ) result(err) &
  & bind(C,name="infero_print_config")
  use iso_c_binding
  type(c_ptr), intent(in), value :: handle_impl
  integer(c_int) :: err
end function

end interface

! ---------  Inference API
interface infero_inference ! function overloading
  module procedure infero_inference_real32_rank2_rank2
  module procedure infero_inference_real64_rank2_rank2  

  module procedure infero_inference_real32_rank3_rank2
  module procedure infero_inference_real64_rank3_rank2

  module procedure infero_inference_real32_rank4_rank4
  module procedure infero_inference_real64_rank4_rank4
end interface

! ---------  For utility
interface
  pure function strlen(str) result(len) bind(c)
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr), intent(in), value :: str
      integer(c_int) :: len
  end function
end interface

! ---------  error handling
interface
  function infero_error_string_interf(err) result(error_string) bind(c, name='infero_error_string')
    use, intrinsic :: iso_c_binding
    implicit none
    integer(c_int), intent(in), value :: err
    type(c_ptr) :: error_string
  end function
end interface


contains

!===========================================================================================
!===========================================================================================
!===========================================================================================


! ---------  error handling
subroutine infero_check(err)
  integer, intent(in) :: err

  if (err /= INFERO_SUCCESS) then
      print *, "Error: ", infero_error_string(err)
      stop 1
  end if
end subroutine

function infero_error_string(err) result(error_string)
  integer, intent(in) :: err
  character(:), allocatable, target :: error_string
  error_string = fortranise_cstr(infero_error_string_interf(err))
end function

!---------------------------------------------------------------------------------

! --------- Infero Model
function infero_initialise( ) result(err)
  integer :: err
  err = INFERO_SUCCESS
  call fckit_main%initialise()
end function

function infero_finalise( ) result(err)
  integer :: err
  err = INFERO_SUCCESS
  call fckit_main%finalise()
end function

function infero_create_handle_from_yaml_string(handle, config_str) result(err)
  class(infero_model), intent(inout) :: handle
  character(c_char) :: config_str
  integer :: err
  err = infero_create_handle_from_yaml_str_interf(config_str, handle%impl)
  err = infero_open_handle_interf( handle%impl )
end function

function infero_create_handle_from_yaml_file(handle, config_str ) result(err)
  use iso_c_binding, only: c_char, c_int, c_ptr
  class(infero_model), intent(inout) :: handle
  character(c_char) :: config_str
  integer :: err
  err = infero_create_handle_from_yaml_file_interf(config_str, handle%impl)
  err = infero_open_handle_interf( handle%impl )
end function

function infero_free_handle( handle ) result(err)
  use iso_c_binding, only: c_ptr
  class(infero_model), intent(inout) :: handle
  integer :: err

  if (handle%is_finalised .eqv. .false.) then
    write(*,'(a)') "INFO: Finalising Infero model.."
    err = infero_close_handle_interf( handle%impl )
    err = infero_delete_handle_interf( handle%impl )
    handle%is_finalised = .true.
  end if

end function

function infero_print_statistics( handle ) result(err)
  use iso_c_binding, only: c_ptr
  class(infero_model), intent(inout) :: handle
  integer :: err
  err = infero_print_statistics_interf( handle%impl )
end function

function infero_print_config( handle ) result(err)
  use iso_c_binding, only: c_ptr
  class(infero_model), intent(inout) :: handle
  integer :: err
  err = infero_print_config_interf( handle%impl )
end function

#ifdef HAVE_FINAL
subroutine infero_free_handle_sub( handle )
  use iso_c_binding, only: c_ptr
  type(infero_model), intent(inout) :: handle
  integer :: err
  if (handle%is_finalised .eqv. .false.) then
    write(*,'(a)') "INFO: Finalising Infero model.."
    err = infero_close_handle_interf( handle%impl )
    err = infero_delete_handle_interf( handle%impl )
    handle%is_finalised = .true.
  end if
end subroutine
#endif

function infero_inference_real32_rank2_rank2(handle, array1, array2 ) result(err)
  use, intrinsic :: iso_c_binding
  class(infero_model), intent(in) :: handle
  integer :: err
  real(c_float), intent(inout) :: array1(:,:)
  real(c_float), intent(inout) :: array2(:,:)
  integer(c_int) :: shape1(2)
  integer(c_int) :: shape2(2)
  real(c_float), pointer :: data1(:)
  real(c_float), pointer :: data2(:)  

  shape1 = shape(array1)
  data1  => array_view1d( array1 )
  shape2 = shape(array2)
  data2  => array_view1d( array2 )

  err = infero_inference_real32(handle%impl, size(shape1), data1, shape1, 1, size(shape2), data2, shape2, 1 )
end function

function infero_inference_real64_rank2_rank2(handle, array1, array2 ) result(err)
  use, intrinsic :: iso_c_binding
  class(infero_model), intent(in) :: handle
  integer :: err
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

  err = infero_inference_real64(handle%impl, size(shape1), data1, shape1, 1, size(shape2), data2, shape2, 1 )
end function

function infero_inference_real32_rank3_rank2(handle, array1, array2 ) result(err)
  use, intrinsic :: iso_c_binding
  class(infero_model), intent(in) :: handle
  integer :: err
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

  err = infero_inference_real32(handle%impl, size(shape1), data1, shape1, 1, size(shape2), data2, shape2, 1 )
end function

function infero_inference_real64_rank3_rank2(handle, array1, array2 ) result(err)
  use, intrinsic :: iso_c_binding
  class(infero_model), intent(in) :: handle
  integer :: err
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

  err = infero_inference_real64(handle%impl, size(shape1), data1, shape1, 1, size(shape2), data2, shape2, 1 )
end function

function infero_inference_real32_rank4_rank4(handle, array1, array2 ) result(err)
  use, intrinsic :: iso_c_binding
  class(infero_model), intent(in) :: handle
  integer :: err
  real(c_float), intent(inout) :: array1(:,:,:,:)
  real(c_float), intent(inout) :: array2(:,:,:,:)
  integer(c_int) :: shape1(4)
  integer(c_int) :: shape2(4)
  real(c_float), pointer :: data1(:)
  real(c_float), pointer :: data2(:)

  shape1 = shape(array1)
  data1  => array_view1d( array1 )
  shape2 = shape(array2)
  data2  => array_view1d( array2 )

  err = infero_inference_real32(handle%impl, size(shape1), data1, shape1, 1, size(shape2), data2, shape2, 1 )
end function

function infero_inference_real64_rank4_rank4(handle, array1, array2 ) result(err)
  use, intrinsic :: iso_c_binding
  class(infero_model), intent(in) :: handle
  integer :: err
  real(c_double), intent(inout) :: array1(:,:,:,:)
  real(c_double), intent(inout) :: array2(:,:,:,:)
  integer(c_int) :: shape1(4)
  integer(c_int) :: shape2(4)
  real(c_double), pointer :: data1(:)
  real(c_double), pointer :: data2(:)

  shape1 = shape(array1)
  data1  => array_view1d( array1 )
  shape2 = shape(array2)
  data2  => array_view1d( array2 )

  err = infero_inference_real64(handle%impl, size(shape1), data1, shape1, 1, size(shape2), data2, shape2, 1 )
end function

!---------------------------------------------------------------------------------

function infer_from_map( infero_h, imap, omap ) result(err)
  class(infero_model), intent(inout) :: infero_h
  class(fckit_map), intent(inout) :: imap
  class(fckit_map), intent(inout) :: omap
  integer :: err
  err = infer_from_map_interf(infero_h%impl, imap%c_ptr(), omap%c_ptr())
end function


!---------------------------------------------------------------------------------

function fortranise_cstr(cstr) result(fstr)
  type(c_ptr), intent(in) :: cstr
  character(:), allocatable, target :: fstr
  character(c_char), pointer :: tmp(:)
  integer :: length

  length = strlen(cstr)
  allocate(character(length) :: fstr)
  call c_f_pointer(cstr, tmp, [length])
  fstr = transfer(tmp(1:length), fstr)
end function


end module

