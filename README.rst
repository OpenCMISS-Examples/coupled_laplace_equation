

========================================
Template OpenCMISS example - input files
========================================

This example shows how input files required for an examplar fortran example can be implemented.  For more detailed information refer to the `master <https://github.com/OpenCMISS-Examples/template_example/tree/master>`_ branch.

Input files
===========

If the example requires files to be used as resources for the application, for example a CellML file, 
to get the expected results then resource files such as tha CellML file should be added inside a directory named inputs. The documentation needs to state which resource files are required to result in the expected results.

The CMake file *src/fortran/CMakeLists.txt* has a minor change to copy over the resource file into the build directory.  There are other solutions to the problem of managing how a resource file is used in an applicaiton and we should choose the option that best matches the needs of the example.  Below is the modified *src/fortran/CMakeLists.txt* file that shows how this example has dealt with a small resource file::

  add_executable(XXXXXXXX ZZZZZZZZ.F90)

  target_link_libraries(XXXXXXXX PUBLIC opencmisslibs mpi)

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

Remember
========

Replace this file with a description about your example.
