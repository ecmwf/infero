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

  procedure :: infer_mimo => infer_from_tensor_set

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


! ---------  Tensor set (Tensor set "C"-handle wrapper)
type infero_tensor_set
  type(c_ptr) :: impl = c_null_ptr
  logical :: is_finalised = .false.
contains
  procedure :: initialise => infero_tensor_set_initialise
  procedure :: push_tensor_rank2 => infero_tensor_set_push_rank2
  procedure :: push_tensor_rank3 => infero_tensor_set_push_rank3
  procedure :: push_tensor_rank4 => infero_tensor_set_push_rank4
  generic   :: push_tensor => push_tensor_rank2, push_tensor_rank3, push_tensor_rank4
  procedure :: print => infero_print_tensor_set
  procedure :: free => infero_tensor_set_free

#ifdef HAVE_FINAL
  final :: infero_tensor_set_free_sub
#endif

end type


! ---------  public interface
public :: infero_initialise
public :: infero_finalise

public :: infero_check
public :: infero_error_string

public :: infero_model
public :: infero_tensor_set


interface

!===========================================================================================
!===========================================================================================
!===========================================================================================

  !-----------------------------------------------------------------------------------------
  ! int infero_inference_float(infero_model_t* h, 
  !                            int rank1, 
  !                            const float data1[], 
  !                            const int shape1[], 
  !                            int rank2,
  !                            float data2[], 
  !                            const int shape2[]);
  !
  ! ( must be defined within `extern "C" { ... }` scope )
  !-----------------------------------------------------------------------------------------
  function infero_inference_real32( handle_impl, rank1, data1, shape1, rank2, data2, shape2 ) result(err) &
    & bind(C,name="infero_inference_float")
    use iso_c_binding, only: c_int, c_ptr, c_float, c_char, c_null_char
    
    type(c_ptr), intent(in), value :: handle_impl
    
    integer(c_int), value :: rank1
    real(c_float), dimension(*) :: data1    
    integer(c_int), dimension(*) :: shape1

    integer(c_int), value :: rank2
    real(c_float), dimension(*) :: data2    
    integer(c_int), dimension(*) :: shape2
    integer(c_int) :: err
  end function

  !-----------------------------------------------------------------------------------------
  ! int infero_inference_double(infero_model_t* h, 
  !                             int rank1, 
  !                             const double data1[], 
  !                             const int shape1[], 
  !                             int rank2,
  !                             double data2[], 
  !                             const int shape2[]);
  !-----------------------------------------------------------------------------------------
  function infero_inference_real64( handle_impl, rank1, data1, shape1, rank2, data2, shape2 ) result(err) &
    & bind(C,name="infero_inference_double")
    use iso_c_binding, only: c_int, c_ptr, c_double, c_char, c_null_char

    type(c_ptr), intent(in), value :: handle_impl
    
    integer(c_int), value :: rank1
    real(c_double), dimension(*) :: data1        
    integer(c_int), dimension(*) :: shape1
        
    integer(c_int), value :: rank2
    real(c_double), dimension(*) :: data2
    integer(c_int), dimension(*) :: shape2
    integer(c_int) :: err
  end function

  function infero_initialise_interf( argc, argv ) result(err) &
    & bind(C,name="infero_initialise")
    use iso_c_binding, only: c_int, c_ptr
    integer(c_int), value :: argc
    type(c_ptr) :: argv(15)
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

  function infero_finalise_interf( ) result(err) &
    & bind(C,name="infero_finalise")
    use iso_c_binding
    integer(c_int) :: err
  end function


  ! --------- tensor set
  function infero_tensors_initialise_interf( handle_impl ) result(err) &
    & bind(C,name="infero_create_tensor_set")
    use iso_c_binding, only: c_int, c_ptr
    type(c_ptr), intent(out) :: handle_impl
    integer(c_int) :: err
  end function

  function infero_tensors_free_interf( handle_impl ) result(err) &
    & bind(C,name="infero_delete_tensor_set")
    use iso_c_binding, only: c_int, c_ptr
    type(c_ptr), intent(in), value :: handle_impl
    integer(c_int) :: err
  end function

  function infero_tensor_set_add_tensor_interf( handle_impl, rank, shape_vec, data_vec, name, c_style ) result(err) &
    & bind(C,name="infero_add_tensor")
    use iso_c_binding
    
    type(c_ptr), intent(in), value :: handle_impl
    integer(c_int), intent(in), value :: rank
    integer(c_int), dimension(*) :: shape_vec
    real(c_float), dimension(*)  :: data_vec
    character(c_char)            :: name
    logical(c_bool), value       :: c_style    
    
    integer(c_int) :: err    
  end function
  
  function infero_print_tensor_set_interf( handle_impl ) result(err) &
    & bind(C,name="infero_print_tensor_set")
    use iso_c_binding, only: c_int, c_ptr
    type(c_ptr), intent(in), value :: handle_impl
    integer(c_int) :: err
  end function

  function infer_from_tensor_set_interf( handle_impl, iset, oset ) result(err) &
    & bind(C,name="infero_inference_float_tensor_set")
    use iso_c_binding
    
    type(c_ptr), intent(in), value :: handle_impl
    type(c_ptr), intent(in), value :: iset
    type(c_ptr), intent(in), value :: oset
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

interface infero_initialise
  module procedure infero_initialise_func
end interface

interface infero_finalise
  module procedure infero_finalise_func
end interface

interface tensor_set_push
  module procedure infero_tensor_set_push_rank2
  module procedure infero_tensor_set_push_rank3
  module procedure infero_tensor_set_push_rank4
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
function infero_initialise_func( ) result(err)
  use iso_c_binding, only: c_int, c_ptr
  integer(c_int) :: argc
  type(c_ptr) :: argv(15)
  integer :: err
  call get_c_commandline_arguments(argc,argv)
  err = infero_initialise_interf( argc,argv )
end function

function infero_finalise_func( ) result(err)
  use iso_c_binding
  integer :: err
  err = infero_finalise_interf( )
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

  err = infero_inference_real32(handle%impl, size(shape1), data1, shape1, size(shape2), data2, shape2 )
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

  err = infero_inference_real64(handle%impl, size(shape1), data1, shape1, size(shape2), data2, shape2 )
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

  err = infero_inference_real32(handle%impl, size(shape1), data1, shape1, size(shape2), data2, shape2 )
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

  err = infero_inference_real64(handle%impl, size(shape1), data1, shape1, size(shape2), data2, shape2 )
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

  err = infero_inference_real32(handle%impl, size(shape1), data1, shape1, size(shape2), data2, shape2 )
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

  err = infero_inference_real64(handle%impl, size(shape1), data1, shape1, size(shape2), data2, shape2 )
end function

!---------------------------------------------------------------------------------

! ---------  tensor set

function infero_tensor_set_initialise( handle ) result(err)
  class(infero_tensor_set), intent(inout) :: handle
  integer :: err
  err = infero_tensors_initialise_interf(handle%impl)
end function

function infero_tensor_set_free( handle ) result(err)
  class(infero_tensor_set), intent(inout) :: handle
  integer :: err
  if (handle%is_finalised .eqv. .false.) then
    write(*,'(a)') "INFO: Finalising Tensor Set.."
    err = infero_tensors_free_interf(handle%impl)
    handle%is_finalised = .true.
  end if
end function

#ifdef HAVE_FINAL
subroutine infero_tensor_set_free_sub( handle )
  type(infero_tensor_set), intent(inout) :: handle
  integer :: err
  if (handle%is_finalised .eqv. .false.) then
    write(*,'(a)') "INFO: Finalising Tensor Set.."
    err = infero_tensors_free_interf(handle%impl)
    handle%is_finalised = .true.
  end if
end subroutine
#endif

function infero_tensor_set_push_rank2( handle, tensor, name, c_style ) result(err)
  use iso_c_binding

  class(infero_tensor_set), intent(inout) :: handle
  real(c_float), intent(in) :: tensor(:,:)
  character(len=*), intent(in) :: name
  
  real(c_float), pointer :: data_vec(:)
  integer(c_int) :: shape_vec(2)
  integer(c_int) :: rank  
  
  logical, intent(in), optional :: c_style
  logical(c_bool) :: c_style_actual = .false.

  integer :: err

  if (present(c_style)) c_style_actual = c_style

  data_vec => array_view1d( tensor )
  shape_vec = shape(tensor)
  rank = size(shape_vec)

  err = infero_tensor_set_add_tensor_interf(handle%impl, &
                                            rank, &
                                            shape_vec, &
                                            data_vec, &
                                            name//c_null_char, &
                                            c_style_actual )
end function


function infero_tensor_set_push_rank3( handle, tensor, name, c_style ) result(err)
  use iso_c_binding

  class(infero_tensor_set), intent(inout) :: handle
  real(c_float), intent(in) :: tensor(:,:,:)
  character(len=*), intent(in) :: name
  
  real(c_float), pointer :: data_vec(:)
  integer(c_int) :: shape_vec(3)
  integer(c_int) :: rank  

  logical, intent(in), optional :: c_style
  logical(c_bool) :: c_style_actual = .false.

  integer :: err

  if (present(c_style)) c_style_actual = c_style

  data_vec => array_view1d( tensor )
  shape_vec = shape(tensor)
  rank = size(shape_vec)

  err = infero_tensor_set_add_tensor_interf(handle%impl, &
                                            rank, &
                                            shape_vec, &
                                            data_vec, &
                                            name//c_null_char, &
                                            c_style_actual )
end function


function infero_tensor_set_push_rank4( handle, tensor, name, c_style) result(err)
  use iso_c_binding

  class(infero_tensor_set), intent(inout) :: handle
  real(c_float), intent(in) :: tensor(:,:,:,:)
  character(len=*), intent(in) :: name
  
  real(c_float), pointer :: data_vec(:)
  integer(c_int) :: shape_vec(4)
  integer(c_int) :: rank  

  logical, intent(in), optional :: c_style
  logical(c_bool) :: c_style_actual = .false.

  integer :: err

  if (present(c_style)) c_style_actual = c_style

  data_vec => array_view1d( tensor )
  shape_vec = shape(tensor)
  rank = size(shape_vec)

  err = infero_tensor_set_add_tensor_interf(handle%impl, &
                                            rank, &
                                            shape_vec, &
                                            data_vec, &
                                            name//c_null_char, &
                                            c_style_actual )
end function


function infero_print_tensor_set( handle ) result(err)
  class(infero_tensor_set), intent(inout) :: handle
  integer :: err
  err = infero_print_tensor_set_interf(handle%impl)
end function


function infer_from_tensor_set( infero_h, iset_h, oset_h ) result(err)
  class(infero_model),     intent(inout) :: infero_h
  class(infero_tensor_set), intent(inout) :: iset_h
  class(infero_tensor_set), intent(inout) :: oset_h
  integer :: err
  err = infer_from_tensor_set_interf(infero_h%impl, iset_h%impl, oset_h%impl)
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

