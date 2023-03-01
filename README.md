# Installation
## GaIn
* Get GaIn from TODO.
* Build it as a shared library. On Linux:
```
./configure FCFLAGS="-O2 -g -fPIC" CPPFLAGS="-O2 -g -fPIC"
make
cd src
gfortran -shared -o lib.so *.o */*.o -lmpfr -lmpc
```
(wait a moment, it takes a while)

## Julia
* Get Julia (this was tested with julia 1.8, and should work with any later 1.x version)
* Install this package as a development package: `] dev https://github.com/antoine-levitt/PhotoionizationGTO.jl` in Julia. This installs it in `~/.julia/dev/PhotoionizationGTO` (on Linux). This package includes in its manifest the version numbers of all libraries used, which makes sure it will work basically forever, even if the libraries get updated.
* Set path to GaIn in GaIn.jl

## Python and PySCF
* In Julia, do `] add PyCall`, `using PyCall`, and run `PyCall.python` to figure out what Python is used by Julia
* Based on that, use the appropriate Python magic (pip, conda, whatever) to install pyscf

# Usage
* Activate project: `] activate ~/.julia/dev/PhotoionizationGTO`; this makes the dependencies available for `using`.
* Edit tddft.jl to your liking
* Run it
