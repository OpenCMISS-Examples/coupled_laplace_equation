

=====================================
Template OpenCMISS example - tutorial
=====================================

This example shows the basic template for a tutorial Fortran example.  For more detailed information refer to the `master <https://github.com/OpenCMISS-Examples/template_example/tree/master>`_ branch.

CMake files
===========

The *CMakeLists.txt* file in the root directory uses the following *find_package* command::

  find_package(OpenCMISSLibs 1.3.0 COMPONENTS iron REQUIRED CONFIG)

What is different here from the basic template in the `master <https://github.com/OpenCMISS-Examples/template_example/tree/master>`_ branch is that we have made use of the *COMPONENTS* option to indicate that only the Iron library is required for this example.

Documentation
=============

A tutorial is an example with accompanying detailed documentation, this documentation should be written in reStructured-text in the docs directory.  The documentation should be buildable with `Sphinx <https://pypi.python.org/pypi/Sphinx>`_.  An initial skeleton Sphinx documentation is added here.  To build this documentation in html format do the following::

   cd docs
   make html

Remember
========

Replace this file with a description about your example using the `README.template.rst <README.template.rst>`_ file as a starting point.
