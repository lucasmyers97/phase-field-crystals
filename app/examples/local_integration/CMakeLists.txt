add_executable(is_in_neighborhood_test is_in_neighborhood_test.cpp)
target_link_libraries(is_in_neighborhood_test
    PRIVATE 
    deal.II::Interface)

add_executable(visit_neighbors_test visit_neighbors_test.cpp)
target_link_libraries(visit_neighbors_test
    PRIVATE 
    deal.II::Interface)

install(TARGETS 
        is_in_neighborhood_test 
        visit_neighbors_test
        DESTINATION 
        bin/examples/local_integration)
