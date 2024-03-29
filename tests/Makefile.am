## Process this file with automake to produce Makefile.inx
##
## Original author:
##     Mohammad Akhlaghi <mohammad@akhlaghi.org>
## Contributing author(s):
## Copyright (C) 2015-2023 Free Software Foundation, Inc.
##
## Gnuastro is free software: you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## Gnuastro is distributed in the hope that it will be useful, but
## WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
## General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with Gnuastro. If not, see <http://www.gnu.org/licenses/>.





## Translate conditions that came from './configure' into variables that can
## be used here. This includes the programs and teh dependencies of the
## checks.
##
## Dependency structure
## ====================
##
## The tests depend on each other. One way to fix this is to call them in
## the order that their dependence is satisfied. But that is prone to
## errors and a lot of problem. The user also not want to build all the
## programs. Also, Make allows us to work with multiple threads (with
## '-jN') and in that case, it is vital to define the dependencies.
##
## The output of all the tests that should be used (is recognized by
## Automake and will not mess the system) is the name of the test file
## appended by a '.log'. It contains the output of the program on standard
## output and error. THIS IS IMPORTANT: So even if the program fails, the
## .log file is created. The check if the input for a test exists or not
## should be checked in the test that depends on it, it can't be done here
## in the Makefile.
if COND_CHECK_WITH_VALGRIND
  MAYBE_CHECK_WITH_PROGRAM = "valgrind"
endif
if COND_HASGHOSTSCRIPT
  MAYBE_HASGHOSTSCRIPT = "yes"
endif
if COND_HASLIBJPEG
  MAYBE_HASLIBJPEG = "yes"
endif
if COND_HASLIBTIFF
  MAYBE_HASLIBTIFF = "yes"
endif
if COND_HASCXX
  MAYBE_CXX_PROGS    = versioncxx
  MAYBE_CXX_TESTS    = lib/versioncxx.sh
  versioncxx_SOURCES = lib/versioncxx.cpp
endif
if COND_ARITHMETIC
  MAYBE_ARITHMETIC_TESTS = arithmetic/or.sh \
                           arithmetic/where.sh \
                           arithmetic/snimage.sh \
                           arithmetic/onlynumbers.sh \
                           arithmetic/connected-components.sh \
                           arithmetic/mknoise-sigma-from-mean.sh \
                           arithmetic/mknoise-sigma-from-mean-3d.sh
  arithmetic/or.sh: segment/segment.sh.log
  arithmetic/onlynumbers.sh: prepconf.sh.log
  arithmetic/where.sh: noisechisel/noisechisel.sh.log
  arithmetic/snimage.sh: noisechisel/noisechisel.sh.log
  arithmetic/mknoise-sigma-from-mean.sh: warp/warp_scale.sh.log
  arithmetic/mknoise-sigma-from-mean-3d.sh: mkprof/3d-cat.sh.log
  arithmetic/connected-components.sh: noisechisel/noisechisel.sh.log
endif
if COND_BUILDPROG
  MAYBE_BUILDPROG_TESTS = buildprog/simpleio.sh
  buildprog/simpleio.sh: mkprof/mosaic1.sh.log
endif
if COND_CONVERTT
  MAYBE_CONVERTT_TESTS = convertt/blankch.sh \
                         convertt/fitstotxt.sh \
                         convertt/jpegtotxt.sh \
                         convertt/fitstopdf.sh \
                         convertt/jpegtofits.sh \
                         convertt/fitstotiff.sh \
                         convertt/fitstojpeg.sh \
                         convertt/fitstojpegcmyk.sh
  convertt/fitstopdf.sh: crop/section.sh.log
  convertt/blankch.sh: mkprof/mosaic1.sh.log
  convertt/fitstotxt.sh: mkprof/mosaic1.sh.log
  convertt/fitstojpeg.sh: mkprof/mosaic1.sh.log
  convertt/fitstotiff.sh: mkprof/mosaic1.sh.log
  convertt/jpegtofits.sh: convertt/blankch.sh.log
  convertt/jpegtotxt.sh: convertt/fitstojpeg.sh.log
  convertt/fitstojpegcmyk.sh: mkprof/mosaic1.sh.log
endif
if COND_CONVOLVE
  MAYBE_CONVOLVE_TESTS = convolve/spatial.sh \
                         convolve/frequency.sh \
                         convolve/psf-match.sh \
                         convolve/spectrum-1d.sh
  convolve/spectrum-1d.sh: prepconf.sh.log
  convolve/spatial.sh: mkprof/mosaic1.sh.log
  convolve/psf-match.sh: mkprof/mosaic1.sh.log
  convolve/frequency.sh: mkprof/mosaic1.sh.log
endif
if COND_COSMICCAL
  MAYBE_COSMICCAL_TESTS = cosmiccal/simpletest.sh

  cosmiccal/simpletest.sh: prepconf.sh.log
endif
if COND_CROP
  MAYBE_CROP_TESTS = crop/imgcat.sh \
                     crop/wcscat.sh \
                     crop/section.sh \
                     crop/imgcenter.sh \
                     crop/wcscenter.sh \
                     crop/imgpolygon.sh \
                     crop/wcspolygon.sh \
                     crop/imgpolygonout.sh \
                     crop/imgcenternoblank.sh
  crop/imgcat.sh: mkprof/mosaic1.sh.log
  crop/wcscat.sh: mkprof/mosaic1.sh.log \
                  mkprof/mosaic2.sh.log \
                  mkprof/mosaic3.sh.log \
                  mkprof/mosaic4.sh.log
  crop/section.sh: mkprof/mosaic1.sh.log
  crop/imgcenter.sh: mkprof/mosaic1.sh.log
  crop/wcscenter.sh: mkprof/mosaic1.sh.log \
                     mkprof/mosaic2.sh.log \
                     mkprof/mosaic3.sh.log \
                     mkprof/mosaic4.sh.log
  crop/imgpolygon.sh: mkprof/mosaic1.sh.log
  crop/wcspolygon.sh: mkprof/mosaic1.sh.log \
                      mkprof/mosaic2.sh.log \
                      mkprof/mosaic3.sh.log \
                      mkprof/mosaic4.sh.log
  crop/imgpolygonout.sh: mkprof/mosaic1.sh.log
  crop/imgcenternoblank.sh: mkprof/mosaic1.sh.log

endif
if COND_FITS
  MAYBE_FITS_TESTS = fits/write.sh \
                     fits/print.sh \
                     fits/update.sh \
                     fits/delete.sh \
                     fits/copyhdu.sh
  fits/print.sh: fits/write.sh.log
  fits/update.sh: fits/write.sh.log
  fits/delete.sh: fits/write.sh.log
  fits/write.sh: mkprof/mosaic1.sh.log
  fits/copyhdu.sh: fits/write.sh.log mkprof/mosaic2.sh.log
endif
if COND_MATCH
  MAYBE_MATCH_TESTS = match/sort-based.sh \
                      match/merged-cols.sh \
                      match/kdtree-internal.sh \
                      match/kdtree-separate.sh
  match/sort-based.sh: prepconf.sh.log
  match/merged-cols.sh: prepconf.sh.log
  match/kdtree-internal.sh: prepconf.sh.log
  match/kdtree-separate.sh: prepconf.sh.log
endif
if COND_MKCATALOG
  MAYBE_MKCATALOG_TESTS = mkcatalog/detections.sh \
                          mkcatalog/simple-3d.sh \
                          mkcatalog/objects-clumps.sh \
                          mkcatalog/aperturephot.sh
  mkcatalog/simple-3d.sh: segment/segment-3d.sh.log
  mkcatalog/objects-clumps.sh: segment/segment.sh.log
  mkcatalog/aperturephot.sh: noisechisel/noisechisel.sh.log \
                             mkprof/clearcanvas.sh.log
  mkcatalog/detections.sh: arithmetic/connected-components.sh.log
endif
if COND_MKPROF
  MAYBE_MKPROF_TESTS = mkprof/3d-cat.sh \
                       mkprof/mosaic1.sh \
                       mkprof/mosaic2.sh \
                       mkprof/mosaic3.sh \
                       mkprof/mosaic4.sh \
                       mkprof/radeccat.sh \
                       mkprof/3d-kernel.sh \
                       mkprof/clearcanvas.sh \
                       mkprof/ellipticalmasks.sh
  mkprof/3d-cat.sh: prepconf.sh.log
  mkprof/mosaic1.sh: prepconf.sh.log
  mkprof/mosaic2.sh: prepconf.sh.log
  mkprof/mosaic3.sh: prepconf.sh.log
  mkprof/mosaic4.sh: prepconf.sh.log
  mkprof/radeccat.sh: prepconf.sh.log
  mkprof/3d-kernel.sh: prepconf.sh.log
  mkprof/clearcanvas.sh: arithmetic/mknoise-sigma-from-mean.sh.log
  mkprof/ellipticalmasks.sh: arithmetic/mknoise-sigma-from-mean.sh.log
endif
if COND_NOISECHISEL
  MAYBE_NOISECHISEL_TESTS = noisechisel/noisechisel.sh \
                            noisechisel/noisechisel-3d.sh
  noisechisel/noisechisel.sh: arithmetic/mknoise-sigma-from-mean.sh.log
  noisechisel/noisechisel-3d.sh: arithmetic/mknoise-sigma-from-mean-3d.sh.log
endif
if COND_SEGMENT
  MAYBE_SEGMENT_TESTS = segment/segment.sh \
                        segment/segment-3d.sh
  segment/segment.sh: noisechisel/noisechisel.sh.log
  segment/segment-3d.sh: noisechisel/noisechisel-3d.sh.log
endif
if COND_STATISTICS
  MAYBE_STATISTICS_TESTS = statistics/basicstats.sh \
                           statistics/from-stdin.sh \
                           statistics/estimate_sky.sh \
                           statistics/fitting-polynomial-robust.sh
  statistics/from-stdin.sh: prepconf.sh.log
  statistics/fitting-polynomial-robust.sh: prepconf.sh.log
  statistics/basicstats.sh: arithmetic/mknoise-sigma-from-mean.sh.log
  statistics/estimate_sky.sh: arithmetic/mknoise-sigma-from-mean.sh.log
endif
if COND_TABLE
  MAYBE_TABLE_TESTS = table/arith-img-to-wcs.sh \
                      table/txt-to-fits-ascii.sh \
                      table/fits-ascii-to-txt.sh \
                      table/txt-to-fits-binary.sh \
                      table/fits-binary-to-txt.sh \
                      table/sexagesimal-to-deg.sh
  table/txt-to-fits-ascii.sh: prepconf.sh.log
  table/txt-to-fits-binary.sh: prepconf.sh.log
  table/sexagesimal-to-deg.sh: prepconf.sh.log
  table/fits-ascii-to-txt.sh: table/txt-to-fits-ascii.sh.log
  table/fits-binary-to-txt.sh: table/txt-to-fits-binary.sh.log
  table/arith-img-to-wcs.sh: arithmetic/mknoise-sigma-from-mean.sh.log
endif
if COND_WARP
  MAYBE_WARP_TESTS = warp/warp_scale.sh \
                     warp/homographic.sh
  warp/warp_scale.sh: convolve/spatial.sh.log
  warp/homographic.sh: convolve/spatial.sh.log
endif

# Script tests.
SCRIPT_TESTS = script/psf-unite.sh \
               script/psf-stamp.sh \
               script/zeropoint.sh \
               script/psf-subtract.sh \
               script/sort-by-night.sh \
               script/radial-profile.sh \
               script/psf-scale-factor.sh \
               script/psf-select-stars.sh

# We want to have several FITS files as input for this script.
script/psf-unite.sh: mkprof/mosaic2.sh.log
script/psf-stamp.sh: mkprof/mosaic2.sh.log
script/psf-subtract.sh: mkprof/mosaic1.sh.log
script/radial-profile.sh: mkprof/mosaic2.sh.log
script/psf-scale-factor.sh: mkprof/mosaic1.sh.log
script/psf-select-stars.sh: segment/segment.sh.log
script/zeropoint.sh: table/arith-img-to-wcs.sh.log
script/sort-by-night.sh: mkcatalog/aperturephot.sh.log

# Environment variables for the test scripts.
AM_TESTS_ENVIRONMENT = \
export LANG=C; \
export AWK=$(AWK); \
export LC_NUMERIC=C; \
export mkdir_p="$(MKDIR_P)"; \
export progbdir=programs-built; \
export topsrc=$(abs_top_srcdir); \
export topbuild=$(abs_top_builddir); \
export haslibjpeg=$(MAYBE_HASLIBJPEG); \
export haslibtiff=$(MAYBE_HASLIBTIFF); \
export hasghostscript=$(MAYBE_HASGHOSTSCRIPT); \
export check_with_program=$(MAYBE_CHECK_WITH_PROGRAM);





# Library checks
# ==============
#
# The Gnuastro library is checked by compiling programs and linking them
# with the library. As described in the last paragraph of the "Scripts
# based test suites" section of the Automake manual, all targets specified
# by 'check_PROGRAMS' are compiled prior to actually running the targets of
# 'TESTS'. So they do not need to be specified as any dependency, they will
# be present when the '.sh' based tests are run.

# The 'gnuastro/config.h' (needed by Gnuastro's library) is built by
# '../lib/Makefile.am' and is only meant for outside users (to be tested
# here). Thus (unlike the programs, which use 'config.h') we need to add
# the top build directory to the include search directories ('-I').
LDADD = -lgnuastro $(CONFIG_LDADD)
AM_LDFLAGS = -L\$(top_builddir)/lib
AM_CPPFLAGS = -I\$(top_srcdir)/lib -I\$(top_builddir)/lib

# Rest of library check settings.
check_PROGRAMS = multithread $(MAYBE_CXX_PROGS)
multithread_SOURCES = lib/multithread.c
lib/multithread.sh: mkprof/mosaic1.sh.log





# Final Tests
# ===========
TESTS = prepconf.sh \
        lib/multithread.sh \
        $(MAYBE_CXX_TESTS) \
        $(MAYBE_ARITHMETIC_TESTS) \
        $(MAYBE_BUILDPROG_TESTS) \
        $(MAYBE_CONVERTT_TESTS) \
        $(MAYBE_CONVOLVE_TESTS) \
        $(MAYBE_COSMICCAL_TESTS) \
        $(MAYBE_CROP_TESTS) \
        $(MAYBE_FITS_TESTS) \
        $(MAYBE_MATCH_TESTS) \
        $(MAYBE_MKCATALOG_TESTS) \
        $(MAYBE_MKPROF_TESTS) \
        $(MAYBE_NOISECHISEL_TESTS) \
        $(MAYBE_SEGMENT_TESTS) \
        $(MAYBE_STATISTICS_TESTS) \
        $(MAYBE_SUBTRACTSKY_TESTS) \
        $(MAYBE_TABLE_TESTS) \
        $(MAYBE_WARP_TESTS) \
        $(SCRIPT_TESTS)





# Files to distribute within the tarball (sorted alphabetically).
EXTRA_DIST = $(TESTS) during-dev.sh \
  buildprog/simpleio.c \
  convolve/spectrum.txt \
  crop/cat.txt \
  match/positions-1.txt \
  match/positions-2.txt \
  mkprof/3d-cat.txt \
  mkprof/clearcanvas.txt \
  mkprof/ellipticalmasks.txt \
  mkprof/mkprofcat1.txt \
  mkprof/mkprofcat2.txt \
  mkprof/mkprofcat3.txt \
  mkprof/mkprofcat4.txt \
  mkprof/radeccat.txt \
  statistics/fitting-data.txt \
  statistics/stdin-input.txt \
  table/table.txt





# Files that must be cleaned with 'make clean'.
CLEANFILES = *.log *.txt *.jpg *.fits *.pdf *.eps *.tif simpleio





# CLEANFILES is only for files, not directories. Therefore we are using
# Automake's extending rules to clean the temporary '.gnuastro' directory
# that was built by the 'prepconf.sh' scripot. See "Extending Automake
# rules", and the "What Gets Cleaned" sections of the Automake manual.
clean-local:; rm -rf .gnuastro
