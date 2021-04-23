## Process this file with automake to produce Makefile.inx
##
## Original author:
##     Mohammad Akhlaghi <mohammad@akhlaghi.org>
## Contributing author(s):
## Copyright (C) 2019-2021, Free Software Foundation, Inc.
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





## List of programs (scripts in this directory) to install under the
## 'prefix/bin' directory ('bin_SCRIPTS'), files necessary to distribute
## with the tarball ('EXTRA_DIST') and output files (to be cleaned with
## 'make clean').
bin_SCRIPTS = astscript-ds9-region \
              astscript-radial-profile \
              astscript-sort-by-night

EXTRA_DIST = ds9-region.in \
             radial-profile.in \
             sort-by-night.in

CLEANFILES = $(bin_SCRIPTS)





## Command to do basic substitutions (anything surrounded by an '@').
do_subst = sed -e 's,[@]VERSION[@],$(VERSION),g' \
               -e 's,[@]SCRIPT_NAME[@],$@,g'





## Rules to install the scripts.
astscript-ds9-region: ds9-region.in Makefile
	$(do_subst) < $(srcdir)/ds9-region.in > $@
	chmod +x $@

astscript-radial-profile: radial-profile.in Makefile
	$(do_subst) < $(srcdir)/radial-profile.in > $@
	chmod +x $@

astscript-sort-by-night: sort-by-night.in Makefile
	$(do_subst) < $(srcdir)/sort-by-night.in > $@
	chmod +x $@