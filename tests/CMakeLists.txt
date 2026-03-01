option(DAX_TEST_CURRENT_IS_1 "Read ping-1 buffers as current state in CUDA tests" OFF)

add_executable(tier1_numeric_tests tier1_numeric_tests.cu)
add_executable(tier2_physics_tests tier2_physics_tests.cu)
add_executable(tier3_cross_validation tier3_cross_validation.cu)

target_include_directories(tier1_numeric_tests PRIVATE ${CMAKE_SOURCE_DIR})
target_include_directories(tier2_physics_tests PRIVATE ${CMAKE_SOURCE_DIR})
target_include_directories(tier3_cross_validation PRIVATE ${CMAKE_SOURCE_DIR})

# Replace dax_core with your project library target when integrating.
if(TARGET dax_core)
  target_link_libraries(tier1_numeric_tests PRIVATE dax_core CUDA::cudart)
  target_link_libraries(tier2_physics_tests PRIVATE dax_core CUDA::cudart)
  target_link_libraries(tier3_cross_validation PRIVATE dax_core CUDA::cudart)
endif()

add_test(NAME tier1_numeric_tests COMMAND tier1_numeric_tests)
add_test(NAME tier2_physics_tests COMMAND tier2_physics_tests)
add_test(NAME tier3_cross_validation COMMAND tier3_cross_validation)

if(DAX_TEST_CURRENT_IS_1)
  target_compile_definitions(tier1_numeric_tests PRIVATE DAX_TEST_CURRENT_IS_1=1)
  target_compile_definitions(tier2_physics_tests PRIVATE DAX_TEST_CURRENT_IS_1=1)
  target_compile_definitions(tier3_cross_validation PRIVATE DAX_TEST_CURRENT_IS_1=1)
endif()
