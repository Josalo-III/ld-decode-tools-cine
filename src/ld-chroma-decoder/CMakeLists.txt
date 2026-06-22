# Library for chroma decoders

# For M_PI constant
add_compile_definitions(_USE_MATH_DEFINES)

add_library(lddecode-chroma STATIC
	decoder.cpp
	decoderpool.cpp
	cadenceassembler.cpp
    carriergrammar.cpp
    comb.cpp
    combcandidate.cpp
    combreach.cpp
    comblocked.cpp
    componentframe.cpp
    framecanvas.cpp
    outputwriter.cpp
    palcolour.cpp
    sourcefield.cpp
    transformpal.cpp
    transformpal2d.cpp
    transformpal3d.cpp
	monodecoder.cpp
)

target_include_directories(lddecode-chroma PUBLIC . ${FFTW_INCLUDE_DIR})

target_link_libraries(lddecode-chroma
    PUBLIC
        ${FFTW_LIBRARY}
    PRIVATE
        Qt::Core
        lddecode-library
)

# ld-chroma-decoder

add_executable(ld-chroma-decoder
    decoder.cpp
    decoderpool.cpp
    main.cpp
    monodecoder.cpp
    ntscdecoder.cpp
    paldecoder.cpp
)

set_target_properties(lddecode-chroma PROPERTIES AUTOMOC ON)

target_include_directories(ld-chroma-decoder PRIVATE ${FFTW_INCLUDE_DIR})

target_link_libraries(ld-chroma-decoder PRIVATE Qt::Core lddecode-library lddecode-chroma ${FFTW_LIBRARY})

install(TARGETS ld-chroma-decoder)
