add_library(StressTools
    stress_calculator.cpp
    )

target_include_directories(StressTools
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    )

target_include_directories(StressTools
    PUBLIC
    ${SRC_DIR}
    )

target_link_libraries(StressTools
    PUBLIC
    deal.II::Interface
    )

install(TARGETS StressTools LIBRARY DESTINATION lib)
