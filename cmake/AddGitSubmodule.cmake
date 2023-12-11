find_package(Git QUIET)

function(add_submodule dir)
    if(GIT_FOUND AND EXISTS ${PROJECT_SOURCE_DIR}/.git)
        option(GIT_SUBMODULE "Check submodules during build" ON)
        if(GIT_SUBMODULE AND NOT EXISTS ${dir}/CMakeLists.txt)
            message(STATUS "Submodule update")
            execute_process(
                COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive -- ${dir}
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                RESULT_VARIABLE GIT_SUBMODULE_RESULT
            )
            if(NOT GIT_SUBMODULE_RESULT EQUAL "0")
                message(FATAL_ERROR "AddGitSubmodule go bruh")
            endif()
        endif()
    endif()
    add_subdirectory(${dir})
endfunction()
