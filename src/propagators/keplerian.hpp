/*************************************************************************
 * Copyright (C) 2011 by Saleh Dindar and the Swarm-NG Development Team  *
 *                                                                       *
 * This program is free software; you can redistribute it and/or modify  *
 * it under the terms of the GNU General Public License as published by  *
 * the Free Software Foundation; either version 3 of the License.        *
 *                                                                       *
 * This program is distributed in the hope that it will be useful,       *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 * GNU General Public License for more details.                          *
 *                                                                       *
 * You should have received a copy of the GNU General Public License     *
 * along with this program; if not, write to the                         *
 * Free Software Foundation, Inc.,                                       *
 * 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ************************************************************************/

/*! \file keplerian.hpp
 *   \brief Defines a solver for differential Kepler's equation in universal variable x. 
 *
 * solving differential Kepler's equation
 * in universal variable x
 * using Laguerre method as outlined by Prusing+C eqn 2.43
 * code adapted from Alice Quillen's Qymsym code 
 * see http://astro.pas.rochester.edu/~aquillen/qymsym/
 *
 */

#pragma once

//! functions needed for kepstep
// code adapted from Alice Quillen's Qymsym code 
// see http://astro.pas.rochester.edu/~aquillen/qymsym/

//! equation 2.40a Prussing + Conway
GPUAPI double C_prussing(double y); 

//! equation 2.40b Prussing +Conway
GPUAPI double S_prussing(double y); 

//! equation 2.40a Prussing + Conway
GPUAPI void SC_prussing(double y, double& S, double &C); 

//! equation 2.40a Prussing + Conway
__device__ void SC_prussing_fast(double y, double& S, double &C); 

GPUAPI double solvex(double r0dotv0, double alpha,
		     double sqrtM1, double r0, double dt);


///////////////////////////////////////////////////////////////
//! advance a particle using f,g functions and universal variables
// for differental kepler's equation
//  has an error catch for r0=0 so can be run with central body
// Based on equations by Prussing, J. E. and Conway, B. A. 
// Orbital Mechanics 1993, Oxford University Press, NY,NY  chap 2 
// npos,nvel are new positions and velocity
// pos, vel are old ones
// code adapted from Alice Quillen's Qymsym code 
// see http://astro.pas.rochester.edu/~aquillen/qymsym/
///////////////////////////////////////////////////////////////
//GPUAPI void kepstep(double4 pos, double4 vel, double4* npos, double4* nvel, double deltaTime, double GM)
GPUAPI void drift_kepler(double& x_old, double& y_old, double& z_old, double& vx_old, double& vy_old, double& vz_old, const double sqrtGM, const double deltaTime);
