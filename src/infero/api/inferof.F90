/*
 * (C) Copyright 1996- ECMWF.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation
 * nor does it submit to any jurisdiction.
 */

module inferof
use, intrinsic :: iso_c_binding

implicit none

private


! infero handle type that wraps the c-handle
type infero_handle
  type(c_ptr) :: impl = c_null_ptr
contains
  procedure :: from_yaml_string => infero_create_handle_from_yaml_string
  procedure :: from_yaml_file => infero_create_handle_from_yaml_file
  procedure :: open => infero_open_handle
  ! procedure :: infer => infero_inference
  procedure :: close => infero_close_handle
  procedure :: delete => infero_delete_handle
end type


public :: infero_handle
public :: infero_initialise
public :: infero_inference
public :: infero_finalise

public :: print_tensor


interface

!=========================================================================
!=========================================================================
!=========================================================================


  !-----------------------------------------------------------------------------------------
  ! void infero_inference_real32( void* handle,
  !                               double data1[], int rank1, int shape1[],
  !                               double data2[], int rank2, int shape2[]  );
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

end interface


! Inference API
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

! Array views API

interface array_view1d ! function overloading
  module procedure array_view1d_real32_r2  
  module procedure array_view1d_real64_r2

  module procedure array_view1d_real32_r3
  module procedure array_view1d_real64_r3

  module procedure array_view1d_real32_r4
  module procedure array_view1d_real64_r4
end interface

interface print_tensor
  module procedure print_tensor_rank2
end interface


contains

!-----------------------------------------------------------------------------------------------------------------------
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
  class(infero_handle), intent(inout) :: handle
  character(c_char) :: config_str
  integer :: err
  err = infero_create_handle_from_yaml_str_interf(config_str, handle%impl)
end function

function infero_create_handle_from_yaml_file(handle, config_str ) result(err)
  use iso_c_binding, only: c_char, c_int, c_ptr
  class(infero_handle), intent(inout) :: handle
  character(c_char) :: config_str
  integer :: err
  err = infero_create_handle_from_yaml_file_interf(config_str, handle%impl)
end function

function infero_open_handle( handle ) result(err)
  use iso_c_binding, only: c_ptr
  class(infero_handle), intent(inout) :: handle
  integer :: err
  err = infero_open_handle_interf( handle%impl )
end function

function infero_close_handle( handle ) result(err)
  use iso_c_binding, only: c_ptr
  class(infero_handle), intent(inout) :: handle
  integer :: err
  err = infero_close_handle_interf( handle%impl )
end function

function infero_delete_handle( handle ) result(err)
  use iso_c_binding, only: c_ptr
  class(infero_handle), intent(inout) :: handle
  integer :: err
  err = infero_delete_handle_interf( handle%impl )
end function

function infero_handle_infer( handle ) result(err)
  use iso_c_binding, only: c_ptr
  class(infero_handle), intent(inout) :: handle
  integer :: err
  err = infero_delete_handle_interf( handle%impl )
end function

!!! Inference

function infero_inference_real32_rank2_rank2(handle, array1, array2 ) result(err)
  use, intrinsic :: iso_c_binding
  class(infero_handle), intent(in) :: handle
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
  class(infero_handle), intent(in) :: handle
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
  class(infero_handle), intent(in) :: handle
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
  class(infero_handle), intent(in) :: handle
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
  class(infero_handle), intent(in) :: handle
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
  class(infero_handle), intent(in) :: handle
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

function array_view1d_real64_r2(array) result( view )
  use, intrinsic :: iso_c_binding
  real(c_double), intent(in), target :: array(:,:)
  type(c_ptr) :: array_c_ptr
  real(c_double), pointer :: view(:)  
  nullify(view)
  array_c_ptr = c_loc_real64(array(1,1))
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

function array_view1d_real64_r3(array) result( view )
  use, intrinsic :: iso_c_binding
  real(c_double), intent(in), target :: array(:,:,:)
  type(c_ptr) :: array_c_ptr
  real(c_double), pointer :: view(:)
  nullify(view)
  array_c_ptr = c_loc_real64(array(1,1,1))
  call c_f_pointer ( array_c_ptr , view , (/size(array)/) )
end function


function array_view1d_real32_r4(array) result( view )
  use, intrinsic :: iso_c_binding
  real(c_float), intent(in), target :: array(:,:,:,:)
  type(c_ptr) :: array_c_ptr
  real(c_float), pointer :: view(:)
  nullify(view)
  array_c_ptr = c_loc_real32(array(1,1,1,1))
  call c_f_pointer ( array_c_ptr , view , (/size(array)/) )
end function

function array_view1d_real64_r4(array) result( view )
  use, intrinsic :: iso_c_binding
  real(c_double), intent(in), target :: array(:,:,:,:)
  type(c_ptr) :: array_c_ptr
  real(c_double), pointer :: view(:)
  nullify(view)
  array_c_ptr = c_loc_real64(array(1,1,1,1))
  call c_f_pointer ( array_c_ptr , view , (/size(array)/) )
end function

!-----------------------------------------------------------------------------------------------------------------------

!======================== utility tools =========================

subroutine get_c_commandline_arguments(argc,argv)
  use, intrinsic :: iso_c_binding
  integer(c_int), intent(out) :: argc
  type(c_ptr), intent(inout) :: argv(:)
  character(kind=c_char,len=1), save, target :: args(255)
  character(kind=c_char,len=255), save, target :: cmd
  character(kind=c_char,len=255) :: arg
  integer(c_int) :: iarg, arglen, pos, ich, argpos, i
  call get_command(cmd)
  do ich=1,len(cmd)
    if (cmd(ich:ich) == " ") then
      cmd(ich:ich) = c_null_char
      exit
    endif
  enddo
  argv(1) = c_loc(cmd(1:1))
  argc = command_argument_count()+1
  pos = 1
  do iarg=1,argc
    argpos = pos
    call get_command_argument(iarg, arg )
    arglen = len_trim(arg)
    do ich=1,arglen
      args(pos) = arg(ich:ich)
      pos = pos+1
    end do
    args(pos) = c_null_char;  pos = pos+1
    args(pos) = " ";          pos = pos+1
    argv(iarg+1) = c_loc(args(argpos))
  enddo

end subroutine



subroutine print_tensor_rank2(t, name)
  use, intrinsic :: iso_c_binding
  real(c_float), intent(in) :: t(:,:)
  character(len=*), intent(in) :: name
  
  real(c_float), pointer :: data_vec(:)
  integer(c_int) :: shape_vec(2)
  integer(c_int) :: rank

  data_vec => array_view1d( t )
  shape_vec = shape(t)
  rank = size(shape_vec)

  print*, "name = ", name
  print*, "name len= ", len(name)
  print*, "shape_vec = ", shape_vec
  print*, "rank = ", rank
end subroutine

end module

