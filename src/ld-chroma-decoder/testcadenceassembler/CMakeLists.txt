add_executable(testcadenceassembler
    testcadenceassembler.cpp
)

target_link_libraries(testcadenceassembler PRIVATE
    Qt::Core
    lddecode-library
    lddecode-chroma
)

add_test(NAME testcadenceassembler COMMAND testcadenceassembler)
