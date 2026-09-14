function(belief_update_apply_project_options target)
  target_compile_features(${target} PRIVATE cxx_std_20)

  if(MSVC)
    target_compile_options(${target} PRIVATE /W4 /permissive-)
    if(BELIEF_UPDATE_WARNINGS_AS_ERRORS)
      target_compile_options(${target} PRIVATE /WX)
    endif()
  else()
    target_compile_options(
      ${target}
      PRIVATE
        -Wall
        -Wextra
        -Wconversion
        -Wpedantic
        -Wshadow
    )
    if(BELIEF_UPDATE_WARNINGS_AS_ERRORS)
      target_compile_options(${target} PRIVATE -Werror)
    endif()
  endif()

  if(BELIEF_UPDATE_ENABLE_SANITIZERS)
    if(MSVC)
      message(FATAL_ERROR "BELIEF_UPDATE_ENABLE_SANITIZERS is not configured for MSVC")
    endif()
    target_compile_options(${target} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(${target} PRIVATE -fsanitize=address,undefined)
  endif()
endfunction()
