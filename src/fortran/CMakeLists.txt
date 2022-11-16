add_executable(XXXXXXXX ZZZZZZZZ.F90)

target_link_libraries(XXXXXXXX PUBLIC opencmisslibs)

# This application requires an input file to execute correctly we have several options
# as to how we can make this happen.
#   1. Copy resource.txt file from src directory to build directory.
#      - This works well for small files.
#   2. Modify the example so that we can pass the file location through as a parameter.
#      - This is a good option for bigger files.
#   3. Create symlinks to the resource file with CMake.
#      - This is an awkward option with lots of drawbacks, most notably it won't work on Windows.
#   4. Configure the source file to contain the absolute location of the resource file.
#      - Usually this would be done in a header file and works well but requires a bit of CMaking.
# 
# For this example situation we will use option 1. as it is the best fit.
# We will make use the configure file CMake command.  This command will copy over a new file if any
# changes have been made to the original.  Consult the CMake documentation for explanation on the
# parameters of this function.
configure_file(inputs/resource.txt ${PROJECT_BINARY_DIR}/resource.txt)
