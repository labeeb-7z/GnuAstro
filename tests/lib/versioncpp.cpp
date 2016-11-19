/*********************************************************************
A test program to get and use the version number of Gnuastro within C++.

Original author:
     Mohammad Akhlaghi <akhlaghi@gnu.org>
Contributing author(s):
Copyright (C) 2015, Free Software Foundation, Inc.

Gnuastro is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your
option) any later version.

Gnuastro is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with Gnuastro. If not, see <http://www.gnu.org/licenses/>.
**********************************************************************/

#include <cstdlib>
#include <iostream>

#include <gnuastro/config.h>

int
main(void)
{
  std::cout << "Gnuastro version is: " << GAL_CONFIG_VERSION << ".\n";
  return EXIT_SUCCESS;
}
