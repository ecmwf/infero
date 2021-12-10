!
! (C) Copyright 1996- ECMWF.
!
! This software is licensed under the terms of the Apache Licence Version 2.0
! which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
! In applying this licence, ECMWF does not waive the privileges and immunities
! granted to it by virtue of its status as an intergovernmental organisation
! nor does it submit to any jurisdiction.
!

module test_infero_model

    use inferof
    use iso_c_binding
    use iso_fortran_env

contains

    subroutine check_api_call(err, msg, success)        
        integer, intent(in) :: err
        character(*), intent(in) :: msg
        logical, intent(inout) :: success

        if (err /= INFERO_SUCCESS) then

            ! print error and move on..
            write(error_unit, *) 'Failed API call: ', msg
            write(error_unit, *) 'Error: ', infero_error_string(err)
            success = .false.
        end if
    end subroutine

end module

program fapi_infero_model

    use test_infero_model
    implicit none
    logical :: success

    success = .true.
    call check_api_call(infero_initialise(), "infero_initialise", success)
    call check_api_call(infero_finalise(),"infero_finalise", success)

    if (.not. success) stop -1

end program