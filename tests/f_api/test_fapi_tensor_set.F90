module test_infero_tensor_set

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

    subroutine test_tensor_set(success)
        type(infero_tensor_set) :: t_set
        logical :: success
        character(len=*), parameter :: t1_name = "t1"
        real(c_float) :: t1(1,1) = 1
        real(c_float) :: t1_backup(1,1) = 1

        character(len=*), parameter :: t2_name = "t2"
        real(c_float) :: t2(1,64) = 1
        real(c_float) :: t2_backup(1,64) = 1

        character(len=*), parameter :: t3_name = "t3"
        real(c_float) :: t3(1,32,128) = 0
        real(c_float) :: t3_backup(1,32,128) = 0

        character(len=*), parameter :: t4_name = "t3"
        real(c_float) :: t4(10) = 0
        real(c_float) :: t4_backup(10) = 0

        success = .true.

        call check_api_call(t_set%initialise(), "t_set_initialise", success)
        call check_api_call(t_set%push_tensor(t1, t1_name), "push_tensor_1", success)
        call check_api_call(t_set%push_tensor(t2, t2_name), "push_tensor_2", success)
        call check_api_call(t_set%push_tensor(t3, t3_name), "push_tensor_3", success)
        call check_api_call(t_set%push_tensor(t3, t4_name), "push_tensor_4", success)

        ! check that inputs are not changed by mistake
        if (any(t1 /= t1_backup)) then
            write(error_unit, *) 'T1 and original T1 differ!' 
            success = .false.
        end if

        if (any(t2 /= t2_backup)) then 
            write(error_unit, *) 'T2 and original T2 differ!' 
            success = .false.
        end if

        if (any(t3 /= t3_backup)) then 
            write(error_unit, *) 'T3 and original T3 differ!' 
            success = .false.
        end if

        if (any(t4 /= t4_backup)) then 
            write(error_unit, *) 'T4 and original T4 differ!' 
            success = .false.
        end if

        call check_api_call(t_set%free(), "t_set_finalise", success)
        
    end subroutine

end module

program fapi_tensor_set

    use test_infero_tensor_set
    implicit none
    logical :: success

    success = .true.

    ! init infero
    call check_api_call(infero_initialise(), "infero_initialise", success)

    ! do various checks on the tensor set
    call test_tensor_set(success)

    ! infero finalise
    call check_api_call(infero_finalise(),"infero_finalise", success)

    if (.not. success) stop -1

end program