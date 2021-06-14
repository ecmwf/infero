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
use iso_c_binding, only : c_double, c_float
implicit none

real(c_float), allocatable :: tensor3f(:,:,:)
real(c_float), allocatable :: tensor2f(:,:)
real(c_float), allocatable :: it2f(:,:)
real(c_float), allocatable :: ot2f(:,:)
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

! float
allocate( it2f(8,191) ) ! input  [ 8, 191 ]
allocate( ot2f(8,126) ) ! output [ 8 , 126 ]

it2f(:,:) = 7._c_float
ot2f(:,:) = 0._c_float

call infero_inference( it2f, ot2f )

end program

