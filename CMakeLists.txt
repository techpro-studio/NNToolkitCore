cmake_minimum_required(VERSION 3.17)
project(NNToolkitCore C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 14)

set(SOURCE_DIR "" CACHE PATH
    "Directory that contains the nntoolkitcore project"
)

option(USE_SIGNAL_API "Use signal API" ON)

if(NOT SOURCE_DIR)
    get_filename_component(SOURCE_DIR
        "${CMAKE_CURRENT_LIST_DIR}"
        ABSOLUTE
)
endif()

include_directories(
    BEFORE
    ${SOURCE_DIR}
)

set (PUBLIC_HEADERS nntoolkitcore/include/NNToolkitCore.h
    nntoolkitcore/layers/time_distributed_dense.h
    nntoolkitcore/layers/gru_2.h
    nntoolkitcore/train/optimizers.h
    nntoolkitcore/layers/dense.h
    nntoolkitcore/layers/gru.h
    nntoolkitcore/train/loss.h
    nntoolkitcore/layers/lstm.h
    nntoolkitcore/layers/activation.h
    nntoolkitcore/layers/activation_default.h
    nntoolkitcore/layers/batch_norm.h
    nntoolkitcore/layers/conv_1d.h
    nntoolkitcore/core/debug.h
)

set(PRIVATE_HEADERS
    nntoolkitcore/core/types.h
    nntoolkitcore/core/ops.h
    nntoolkitcore/core/loop.h
)

set(SOURCES
    nntoolkitcore/layers/activation.c
    nntoolkitcore/layers/activation_default.c
    nntoolkitcore/layers/batch_norm.c
    nntoolkitcore/layers/conv_1d.c
    nntoolkitcore/core/debug.c
    nntoolkitcore/layers/dense.c
    nntoolkitcore/layers/gru.c
    nntoolkitcore/train/loss.c
    nntoolkitcore/layers/lstm.c
    nntoolkitcore/layers/time_distributed_dense.c
    nntoolkitcore/train/optimizers.c
    nntoolkitcore/layers/gru_2.c
)

if(APPLE)
    list(APPEND SOURCES nntoolkitcore/core/apple_ops.c)
else()
    list(APPEND SOURCES nntoolkitcore/core/default_ops.cc)
endif()

if (USE_SIGNAL_API)
    list(APPEND SOURCES
        nntoolkitcore/signal/dft.c
        nntoolkitcore/signal/dft.h
        nntoolkitcore/signal/spectrogram.c
        nntoolkitcore/signal/window.c
    )
    list(APPEND PUBLIC_HEADERS
        nntoolkitcore/signal/window.h
        nntoolkitcore/signal/spectrogram.h
    )
    if(NOT APPLE)
        list(APPEND SOURCES
            third_party/kissfft/kiss_fft.h
            third_party/kissfft/kiss_fft.c
            third_party/kissfft/_kiss_fft_guts.h
        )
    endif()

endif ()

add_library(${PROJECT_NAME} ${SOURCES} ${PUBLIC_HEADERS} ${PRIVATE_HEADERS})

if(APPLE)
    target_compile_options(${PROJECT_NAME}
        PUBLIC
        "-fembed-bitcode"
    )

    target_link_libraries(${PROJECT_NAME} PRIVATE "-framework Accelerate")

    set_target_properties (${PROJECT_NAME} PROPERTIES
        FRAMEWORK TRUE
        INSTALL_NAME_DIR "@rpath"
        PRIVATE_HEADER "${PRIVATE_HEADERS}" # Needed for set_target_properties to work for framework
        PUBLIC_HEADER "${PUBLIC_HEADERS}" # Needed for set_target_properties to work for framework
        MACOSX_FRAMEWORK_IDENTIFIER "studio.techpro.${PROJECT_NAME}" #CFBundleIdentifier
        MACOSX_FRAMEWORK_SHORT_VERSION_STRING "${PROJECT_VERSION}" #CFBundleShortVersionString
    )

    # Install
    install (TARGETS ${PROJECT_NAME}
        FRAMEWORK DESTINATION . # relative to CMAKE_INSTALL_PREFIX
        PRIVATE_HEADER
        PUBLIC_HEADER
    )
endif()