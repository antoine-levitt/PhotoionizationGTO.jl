# Installation
* Get GaIn from TODO.
* Build it as a shared library. On linux:
```
./configure FCFLAGS="-O2 -g -fPIC" CPPFLAGS="-O2 -g -fPIC"
make
cd src
gfortran -shared -o lib.so *.o */*.o -lmpfr -lmpc
```
(wait a moment, it takes a while)
* Get Julia (this was tested with julia 1.8, and should work with any later 1.x version)
* Install this package as a development package: `] dev https://github.com/antoine-levitt/PhotoionizationGTO.jl` in Julia
* Set path to GaIn in GaIn.jl
* Activate project: `] activate ~/.julia/dev/PhotoionizationGTO`
