#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "clapfft::clapfft" for configuration ""
set_property(TARGET clapfft::clapfft APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(clapfft::clapfft PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "CXX"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libclapfft.a"
  )

list(APPEND _cmake_import_check_targets clapfft::clapfft )
list(APPEND _cmake_import_check_files_for_clapfft::clapfft "${_IMPORT_PREFIX}/lib/libclapfft.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
