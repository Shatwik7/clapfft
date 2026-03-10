#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "clapfft::clapfft" for configuration "Release"
set_property(TARGET clapfft::clapfft APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clapfft::clapfft PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclapfft.so"
  IMPORTED_SONAME_RELEASE "libclapfft.so"
  )

list(APPEND _cmake_import_check_targets clapfft::clapfft )
list(APPEND _cmake_import_check_files_for_clapfft::clapfft "${_IMPORT_PREFIX}/lib/libclapfft.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
