Phase-field crystals
====================
----

This repository contains code for a finite element solver used to implement the [phase-field crystal model](https://doi.org/10.1103/PhysRevB.97.054113).
The code is based on the [deal.II finite element library](dealii.org).
The first step is to write code for the time evolution of a 2-dimensional hexagonal lattice.
This README will be updated as the repository is expanded and organizational choices are made.

Dependencies
------------
----

This library relies on the [deal.II finite element library](dealii.org).
Deal.II can also optionally rely on a whole slew of other libraries to enable various extended functionality.
At the current time, a vanilla installation of deal.II is all that is required, but we plan to include parallelization in the future which will require MPI support at the very least, and likely an external linear algebra library such as Trilinos.

Build and installation instructions
-----------------------------------
----

Building and installation is handled using CMake.

### Basic installation

To generate build files in a `build` directory:
```
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=/path/where/pfc/should/be/installed /path/to/pfc/source
```
To build and install libraries and binaries:
```
cmake --build /path/to/build/directory --target install
```
If you would like to run the build in parallel, append the `-- -jn` command to the build command, and replace `n` with the number of processors you would like to run the build on.
Beware, if you run on many processors, you may run out of memory before you run out of processors.

### Advanced installation

CMake works like any scripting language, in that there are commands that are executed sequentially in script files, called `CMakeLists.txt`. 
Many directories contain such files, and CMake works by top-level scripts calling scripts in subdirectories to deal with generating build files specific to those subdirectories.
Additionally, CMake uses variables dubbed "cache variables" which can be set by default within the scripts, but can also be overridden by users who wish to specify particular aspects of how the code is built.
These are set either upon the first call to cmake from the build directory (i.e. `cmake /path/to/pfc/source -DCACHE_VARIABLE_NAME=cache_variable_value`), or set using the `ccmake` GUI.
In the latter case, one has to navigate to the build directory, and then type `ccmake /path/to/pfc/source`, in which case a navigable GUI allows users to change the values of cache variables.
Below we detail several such cache variables:

* `-DDEAL_II_DIR`: Directory where a user's installation of deal.II is. 
By default this is empty, and so CMake will search the default locations (usually `/usr/local/`).
If your deal.II installation is located elsewhere, or you want to link to a different version of deal.II then the default-installed one, you must set this variable.

* `-DCMAKE_BUILD_TYPE`: Indicates whether CMake should build source in Debug or Release mode.
One main difference is that Debug mode includes debugging symbols which may be used by a debugger like GDB to debug your program.
If a program does not have these symbols, debugging becomes impossible with such a program.
Additionally, there are several checks in the code that can be turned off in Release mode for purposes of high performance (it takes time to do checks). 
Note that this setting also toggles which build type of deal.II the library is linked to.
Default is Release.

* `-DCMAKE_INSTALL_PREFIX`: Indicates where built code should be installed.
Defaults to `usr/local/`.

* `-DCMAKE_EXPORT_COMPILE_COMMANDS`: Indicates whether the compile flags should be written to a `compile_commands.json` file.
This is useful because language server protocols (LSP) such as clangd rely on these files to give useful code hints and autocompletion. 
Since this file is written in the build directory, I usually create a symlink in the main source directory for working on code.

Typically I make separate Debug and Release builds, and so my installation looks like:
```
cd /path/to/phase-field-crystal/source

mkdir build
mkdir build/Debug
mkdir install
mkdir install/Debug

cd build/Debug
cmake ../.. -DCMAKE_INSTALL_PREFIX=/path/to/phase-field-crystal/source/install/Debug -DCMAKE_BUILD_TYPE=Debug -DDEAL_II_DIR=/path/to/dealii

cd ../..
cmake --build build/Debug --target install -- -jn
```
for Debug.
Then for a Release build:
```
cd /path/to/phase-field-crystal/source

mkdir build/Release
mkdir install/Release

cd build/Release
cmake ../.. -DCMAKE_INSTALL_PREFIX=/path/to/phase-field-crystal/source/install/Release -DCMAKE_BUILD_TYPE=Release -DDEAL_II_DIR=/path/to/dealii

cd ../..
cmake --build build/Release --target install -- -jn
```
One final note on installation is that, by default, installed binaries have their Runpath specified to both external libraries and libraries within the phase-field crystal library.
A Runpath is just a path which is hardcoded into an executable's file for where to search for a shared library.
For external libraries, the Runpaths are absolute paths.
For internal libraries they are relative to where the binary is.
That means that if you move an external library, it is likely that the binary will not be able to find it again.
If this happens, one can either add the library's new location to the `LD_LIBRARY_PATH` environment variable, or one may rebuild and reinstall the entire library so that everything is linked properly.
