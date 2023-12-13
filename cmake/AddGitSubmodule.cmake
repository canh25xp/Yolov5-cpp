find_package(Git QUIET)

function(add_submodule dir)
    if(GIT_FOUND AND EXISTS ${PROJECT_SOURCE_DIR}/.git)
        message(STATUS "Checking ${dir} submodule")
        if(NOT EXISTS ${CMAKE_SOURCE_DIR}/${dir}/CMakeLists.txt)
            execute_process(
                COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive -- ${dir}
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                RESULT_VARIABLE GIT_SUBMODULE_RESULT
            )
            if(NOT GIT_SUBMODULE_RESULT EQUAL "0")
                message(FATAL_ERROR "add_submodule ${dir} go bruh")
            endif()
        else()
            message(STATUS "${dir} already initialized")
        endif()
    endif()
    add_subdirectory(${dir})
endfunction()

function(deinit_submodule dir)
    if(GIT_FOUND AND EXISTS ${CMAKE_SOURCE_DIR}/${dir}/.git)
        execute_process(
            COMMAND ${GIT_EXECUTABLE} submodule deinit -- ${dir}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            RESULT_VARIABLE GIT_SUBMODULE_RESULT
        )
    endif()
endfunction()
