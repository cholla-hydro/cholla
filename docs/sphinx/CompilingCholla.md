# Compiling Cholla

## Overview
The Cholla build structure is designed to provide a streamlined process to accommodate build environments for different machines, and to provide some default settings for frequently-used configurations of Cholla (such as hydro-only). When compiling Cholla on a new machine for the first time, you will need to edit a few files in the builds directory in order to get started.

## The builds directory
Within the builds directory, there are several types of files:
* machine.sh
* make.host.*
* make.type.*
* setup.*.sh

To compile Cholla on a new machine, you should create a new make.host.[yourmachinename] file with the relevant information about which compiler to use, where to find the relevant libraries, etc. You should also edit machine.sh to include your new host. If you are running on a cluster, you will also likely need load the relevant modules in order to run whatever type of test you need; you may wish to also add a setup.sh script for your machine for this purpose. For example, for basic hydro, you only need gcc and a cuda or hip compiler, but for multi-GPU runs you will need an mpi compiler, for output hdf5 is recommended, and if you want to use testing, googletest is required.

If you don't know your host name, you can type "hostname --fqdn" to determine it. This is what will be used by default. If your host has an unpleasant name, you may wish to use something simpler in your make.host.<name> file and in machine.sh. In that case, you can override the default host name by setting "export CHOLLA_MACHINE=<name>" in the terminal or in your setup script.

## The Makefile
The top-level cholla directory includes a Makefile, which is not designed to be edited directly. If no make type is specified when compiling, the Makefile assumes you want to build for basic hydro.

## make.type
If you want to run using a preset collection of Makefile flags, you can just type "make" in the top-level directory, which will build a hydro-only version of cholla and put the executable in the "bin" directory. If you would like to build for one of the other presets, you can use, for example "make TYPE=gravity", which includes all the necessary flags for the default hydro build, plus flags needed for the FFT gravity solver. Take a look at the other make.type.* files for more examples. You can also define your own make.type. file to build your favorite version of Cholla. The wiki page [Makefile](https://github.com/cholla-hydro/cholla/wiki/Makefile) contains most of the available flags, along with brief descriptions.
