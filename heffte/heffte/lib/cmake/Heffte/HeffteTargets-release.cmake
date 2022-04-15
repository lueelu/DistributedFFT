#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "Heffte::Heffte" for configuration "Release"
set_property(TARGET Heffte::Heffte APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(Heffte::Heffte PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libheffte.so.2.1.0"
  IMPORTED_SONAME_RELEASE "libheffte.so.2"
  )

list(APPEND _IMPORT_CHECK_TARGETS Heffte::Heffte )
list(APPEND _IMPORT_CHECK_FILES_FOR_Heffte::Heffte "${_IMPORT_PREFIX}/lib/libheffte.so.2.1.0" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
