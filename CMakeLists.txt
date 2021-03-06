cmake_minimum_required(VERSION 2.8)

# Some useful macros
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/macros.cmake)

################################################################################
#                            GLOBAL CONFIGURATION                              #
################################################################################

# Ensures that we are doing an out-of-source build
MACRO_ENSURE_OUT_OF_SOURCE_BUILD("Requires an out of source build.")

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Debug' as none was specified.")
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(
      CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
      "Debug" "Release" "MinSizeRel" "RelWithDebInfo"
    )
endif()


# Project name
project(NeuroEvolution)

# Version
set(NEURO_EVOLUTION_MAJOR_VERSION 0)
set(NEURO_EVOLUTION_MINOR_VERSION 1)
set(NEURO_EVOLUTION_PATCH_VERSION 0)
set(NEURO_EVOLUTION_VERSION
  ${NEURO_EVOLUTION_MAJOR_VERSION}.${NEURO_EVOLUTION_MINOR_VERSION}.${NEURO_EVOLUTION_PATCH_VERSION}
)

# Configuration file
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/config.cmake)

# Static library name
set(NEURO_EVOLUTION_STATIC_LIBRARY NeuroEvolutionLib)

# Executable name
set(NEURO_EVOLUTION_EXECUTABLE_NAME NeuroEvolution)

# Binary directory
set(NEURO_EVOLUTION_BINARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/bin/${CMAKE_BUILD_TYPE})


################################################################################
#                             COMPILATION FLAGS                                #
################################################################################

# Common flags
set(CXX_DEFINES "-DNEURO_EVOLUTION_NN_DEBUG=0 ${CXX_DEFINES}")
set(CXX_DEFINES "-DNEURO_EVOLUTION_NN_PROGRESSION_INFO=1 ${CXX_DEFINES}")
set(CXX_DEFINES "-DNEURO_EVOLUTION_NN_NO_OUTPUT=1 ${CXX_DEFINES}")
set(CXX_DEFINES "-DNEURO_EVOLUTION_OMP_MATRIX=0 ${CXX_DEFINES}")


if(NEURO_EVOLUTION_COMPILER_GCC OR NEURO_EVOLUTION_COMPILER_CLANG)

    # C++11 flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp")

    # Base compiler flags
    # https://gcc.gnu.org/onlinedocs/gcc/Warning-Options.html
    # https://gcc.gnu.org/onlinedocs/gcc/C_002b_002b-Dialect-Options.html
    # http://stackoverflow.com/a/9862800
    set(CXX_BASE_FLAGS
        "-W -Wall -Wextra -pedantic"
        "-Wold-style-cast -Wparentheses -Wlogical-op -Woverloaded-virtual"
        "-Wfloat-equal -Wpointer-arith -Wcast-align -Wcast-qual"
        "-Winit-self -Wwrite-strings -Wstrict-null-sentinel"
        "-Wredundant-decls -Wswitch-default -Wundef -Wunused-parameter"
        "-Wmissing-include-dirs"
        "-Wdisabled-optimization"
        "-Wnoexcept"

        #"-Weffc++"
        #"-Wshadow"
        #"-Wsign-conversion"
        #"-Wconversion"
        #"-Wstrict-overflow=5"
        #"-Winline"
        #"-Wpadded"
    )
    string(REGEX REPLACE ";" " " CXX_BASE_FLAGS "${CXX_BASE_FLAGS}")

    # Release mode
    set(CXX_FLAGS_RELEASE "-O3 -Werror ${CXX_BASE_FLAGS}")
    set(CMAKE_CXX_FLAGS_RELEASE "${CXX_DEFINES} ${CXX_FLAGS_RELEASE}")
    set(CMAKE_CXX_FLAGS_RELEASE ${CXX_FLAGS_RELEASE})

    if(CMAKE_BUILD_TYPE STREQUAL Release)
        add_definitions(-DNDEBUG)
    endif()

    # Debug mode
    set(CXX_FLAGS_DEBUG "-O0 -g ${CXX_BASE_FLAGS}")
    set(CMAKE_CXX_FLAGS_DEBUG "${CXX_DEFINES} ${CXX_FLAGS_DEBUG}")
    set(CMAKE_CXX_FLAGS_DEBUG ${CXX_FLAGS_DEBUG})

    # Release with debug symbols
    set(CXX_FLAGS_RELWITHDEBINFO "-O3 -g -Werror ${CXX_BASE_FLAGS}")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO
        "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} ${CXX_FLAGS_RELWITHDEBINFO}"
    )

    # Release with stripping of extra symbols
    set(CXX_FLAGS_MINSIZEREL "${CXX_BASE_FLAGS}")
    set(CMAKE_CXX_FLAGS_MINSIZEREL
        "${CMAKE_CXX_FLAGS_MINSIZEREL} ${CXX_FLAGS_MINSIZEREL}"
    )


elseif(NEURO_EVOLUTION_COMPILER_MSVC)

    # Base compiler flags
    set(CXX_BASE_FLAGS
        "/MP4"
		"/W0"
		#"/Wall"
    )
    string(REGEX REPLACE ";" " " CXX_BASE_FLAGS "${CXX_BASE_FLAGS}")

    # Release mode
    set(CXX_FLAGS_RELEASE ${CXX_BASE_FLAGS})
    set(CXX_FLAGS_RELEASE ${CXX_FLAGS_RELEASE} "/O2")
	string(REGEX REPLACE ";" " " CXX_FLAGS_RELEASE "${CXX_FLAGS_RELEASE}")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${CXX_FLAGS_RELEASE}")

    # Debug mode
    set(CXX_FLAGS_DEBUG ${CXX_BASE_FLAGS})
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${CXX_FLAGS_DEBUG}")

    # Release with debug symbols
    set(CXX_FLAGS_RELWITHDEBINFO "${CXX_BASE_FLAGS}")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO
        "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} ${CXX_FLAGS_RELWITHDEBINFO}"
    )

    # Release with stripping of extra symbols
    set(CXX_FLAGS_MINSIZEREL "${CXX_BASE_FLAGS}")
    set(CMAKE_CXX_FLAGS_MINSIZEREL
        "${CMAKE_CXX_FLAGS_MINSIZEREL} ${CXX_FLAGS_MINSIZEREL}"
    )


	# Remove SL security warnings with Visual C++
    add_definitions(-D_CRT_SECURE_NO_DEPRECATE)
	add_definitions(-D_SCL_SECURE_NO_WARNINGS)

else()

    message(FATAL_ERROR "Unsupported compiler")
    return()

endif()

################################################################################
#                                   EXECUTABLE                                 #
################################################################################

set(NEURO_EVOLUTION_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/code/include)
set(NEURO_EVOLUTION_SOURCE_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/code/src)

# Include directories
include_directories(${NEURO_EVOLUTION_INCLUDE_DIR} ${NEURO_EVOLUTION_SOURCE_DIR})
include_directories(SYSTEM ${NEURO_EVOLUTION_EXTERN_INCLUDE_DIR})

# Source files
set(NEURO_EVOLUTION_HEADERS
    ${NEURO_EVOLUTION_INCLUDE_DIR}/functions.hpp
    ${NEURO_EVOLUTION_INCLUDE_DIR}/io_utils.hpp
    ${NEURO_EVOLUTION_INCLUDE_DIR}/matrix.hpp
    ${NEURO_EVOLUTION_INCLUDE_DIR}/neural_network.hpp
    ${NEURO_EVOLUTION_INCLUDE_DIR}/neural_network_synthetizer.hpp
    ${NEURO_EVOLUTION_INCLUDE_DIR}/serialization.hpp
    ${NEURO_EVOLUTION_INCLUDE_DIR}/types.hpp
)

# Source files
set(NEURO_EVOLUTION_SOURCES
    ${NEURO_EVOLUTION_SOURCE_DIR}/functions.cpp
    ${NEURO_EVOLUTION_SOURCE_DIR}/io_utils.cpp
    ${NEURO_EVOLUTION_SOURCE_DIR}/neural_network.cpp
    ${NEURO_EVOLUTION_SOURCE_DIR}/neural_network_synthetizer.cpp
    ${NEURO_EVOLUTION_SOURCE_DIR}/serialization.cpp
)

# Static library
set(LIBRARY_OUTPUT_PATH ${NEURO_EVOLUTION_BINARY_DIR})
add_library(${NEURO_EVOLUTION_STATIC_LIBRARY}
    STATIC ${NEURO_EVOLUTION_HEADERS} ${NEURO_EVOLUTION_SOURCES}
)
target_link_libraries(${NEURO_EVOLUTION_STATIC_LIBRARY} ${NEURO_EVOLUTION_EXTERN_LIBRARIES})


# Executable
set(EXECUTABLE_OUTPUT_PATH ${NEURO_EVOLUTION_BINARY_DIR})
add_executable(${NEURO_EVOLUTION_EXECUTABLE_NAME} ${NEURO_EVOLUTION_SOURCE_DIR}/main.cpp)
target_link_libraries(${NEURO_EVOLUTION_EXECUTABLE_NAME} ${NEURO_EVOLUTION_STATIC_LIBRARY})

# Parent scope
set(NEURO_EVOLUTION_INCLUDE_DIR ${NEURO_EVOLUTION_INCLUDE_DIR}
    CACHE STRING "NeuroEvolution include directory"
)
set(NEURO_EVOLUTION_SOURCE_DIR  ${NEURO_EVOLUTION_SOURCE_DIR}
    CACHE STRING "NeuroEvolution source directory"
)
set(NEURO_EVOLUTION_EXTERN_INCLUDE_DIR ${NEURO_EVOLUTION_EXTERN_INCLUDE_DIR}
    CACHE STRING "NeuroEvolution extern include directory"
)
set(NEURO_EVOLUTION_STATIC_LIBRARY ${NEURO_EVOLUTION_STATIC_LIBRARY}
    CACHE STRING "NeuroEvolution static library"
)
