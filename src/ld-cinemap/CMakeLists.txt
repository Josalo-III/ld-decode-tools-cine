add_executable(ld-cinemap
	cinedisc.cpp
	cinemap.cpp
    main.cpp
    segmenter.cpp
    vbiprobe.cpp
    visualedits.cpp
)

target_link_libraries(ld-cinemap PRIVATE Qt::Core lddecode-library)


install(TARGETS ld-cinemap)