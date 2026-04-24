add_executable(ld-tbc-trim
    ld-tbc-trim.cpp
    tbc-trim-utils.cpp
)

target_link_libraries(ld-tbc-trim PRIVATE Qt::Core lddecode-library)

install(TARGETS ld-tbc-trim)
