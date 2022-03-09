cmake_minimum_required(VERSION 3.5)
project(boost_Pro_P)
set(SOURCES
    src/boost_program_options_Pra.cpp
)
# set(BOOST_ROOT /use/include/boost)
# set(BOOST_NO_SYSTEM_PATHS ON)
FIND_PACKAGE( Boost  COMPONENTS program_options REQUIRED )
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )
add_executable(boost_Pro_P ${SOURCES})
target_link_libraries(boost_Pro_P LINK_PUBLIC ${Boost_LIBRARIES} )