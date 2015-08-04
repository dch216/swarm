/*! \file closebinary.hpp
 *   \brief Defines \ref swarm::gpu::bppt::CloseBinaryPropagator - the GPU implementation 
 *          of John Chamber's Close Binary Propagator.
 *
 */
/* Correct units for input parameters:
 * Mass: Solar masses
 * Positions: AU
 * Velocities: AU/day
 */

#include "swarm/common.hpp"
#include "swarm/swarmplugin.h"
#include <math.h>
#include <stdio.h>

// Separate namespace for use of 'keplerian' functions
namespace kep_cb { 
#include "keplerian.hpp" 
}

namespace swarm {

  namespace gpu {
    namespace bppt {

      /*! Parameters for CloseBinaryPropagator
       * \ingroup propagator_parameters
       *
       */
      struct CloseBinaryPropagatorParams {
	double time_step;
	//! Constructor for CloseBinaryPropagatorParams
	CloseBinaryPropagatorParams(const config& cfg){
	  time_step = cfg.require("time_step", 0.0);
	}
      };

      /*! GPU implementation of John Chamber's Close Binary propagator
       * \ingroup propagators
       *
       * 
       */
      template<class T,class Gravitation>
      struct CloseBinaryPropagator {
	typedef CloseBinaryPropagatorParams params;
	static const int nbod = T::n;
	
	params _params;


	//! Runtime variables
	ensemble::SystemRef& sys;
	Gravitation& calcForces;
	int b;
	int c;
	int ij;

        //Note: Using kepcoords class with members .sma(), .ecc(), .inc(), .lon(), .arg(), .man()
        //	for both star B and the planets. To initialize or compute new keplerian coordinates,
	//	give member function .calc_coords() a single parameter - its body number (var 'b').
	//	It has access to global sys class.

	double sqrtGM;
        double MBin;
	double max_timestep;
  
	double acc_bc;
        double mvsum_c;

        int NBin;

        //! Constructor for CloseBinaryPropagator
	GPUAPI CloseBinaryPropagator(const params& p,ensemble::SystemRef& s,
				     Gravitation& calc)
	  :_params(p),sys(s),calcForces(calc){}
	
	__device__ bool is_in_body_component_grid()
        { return  ( (b < nbod) && (c < 3)); }

  	__device__ bool is_in_body_component_grid_no_prim()
        { return  ( (b != 0) && (b < nbod) && (c < 3)); }

	__device__ bool is_in_body_component_grid_no_star()
        { return ( (b > 1) && (b < nbod) && (c < 3) ); }	

	__device__ bool is_first_thread_in_system()
        { return (thread_in_system()==0); }	

	static GENERIC int thread_per_system(){
	  return nbod * 3;
	}
	
	static GENERIC int shmem_per_system() {
	  return 0;
	}



	/// Shift into Jacobi coordinate system 
	/// Initialization tasks executed before entering loop
        /// Cache sqrtGM, shift coord system, cache acceleration data for this thread's body and component
	GPUAPI void init()  {
	  convert_mass_to_gauss();
	  MBin = sys[0].mass() + sys[1].mass();
	  sqrtGM = sqrt(sys[0].mass() * sys[0].mass() / MBin);

	  if ((fabs(sys[0][0].pos()) > 1.0e-13) || (fabs(sys[0][1].pos()) > 1.0e-13) || (fabs(sys[0][2].pos()) > 1.0e-13) || (fabs(sys[0][0].vel()) > 1.0e-13) || (fabs(sys[0][1].vel()) > 1.0e-13) || (fabs(sys[0][2].vel()) > 1.0e-13))
	    {
	      if (is_first_thread_in_system())
		printf("Warning: Input coordinates of System %d not centered on primary. CB propagator will produce output centered on primary!\n", sys.id());

	      orig_to_helio();
	    }
	    
	  convert_hel_to_jacobi_coord_without_shared();
	  acc_bc = calcForces.acc_planets_cb(ij,b,c);

	  //Determine NBin using semi-major axes
	  NBin = int(0.5 + pow((min_sma() / calc_sma(1)), 1.5));
	}

	/// Before exiting, convert back to standard cartesian coordinate system
	GPUAPI void shutdown() { 
	convert_jacobi_to_hel_coord_without_shared();
	//helio_to_orig();
	convert_mass_to_solar();
	}

        ///Convert mass to Gaussian units
        GPUAPI void convert_mass_to_gauss()   {
	  if ( c == 0 )
	    sys[b].mass() *= 2.959122082855911e-04;
	  
	  __syncthreads();

        }

        ///Convert Gaussian mass back to solar units
        GPUAPI void convert_mass_to_solar()   {
	  if ( c == 0 )
	    sys[b].mass() /= 2.959122082855911e-04;

	  __syncthreads();

        }

        ///Move from original coordinates to heliocentric
        GPUAPI void orig_to_helio() {
	  if( is_in_body_component_grid_no_prim() )
	    {
	      sys[b][c].pos() -= sys[0][c].pos();
	      sys[b][c].vel() -= sys[0][c].vel();
	    }
	  __syncthreads();

	  if (is_first_thread_in_system())
	    for (int k = 0; k < 3; k++)
	      {
		sys[0][k].pos() = 0.0;
		sys[0][k].vel() = 0.0;
	      }
	  __syncthreads();
	}

      ///Move from heliocentric to original coordinates
        GPUAPI void helio_to_orig() {
	  double mtot = 0.0;
	  for (int j = 0; j < nbod; j++)
	    mtot += sys[j].mass();

	  if( is_in_body_component_grid_no_prim() )
	    {
	      sys[0][c].pos() -= 1.0/mtot*sys[b].mass()*sys[b][c].pos();
	      sys[0][c].vel() -= 1.0/mtot*sys[b].mass()*sys[b][c].vel();
	    }
	  __syncthreads();
	  
	  if( is_in_body_component_grid_no_prim() )
	    {
	      sys[b][c].pos() += sys[0][c].pos();
	      sys[b][c].vel() += sys[0][c].vel();
	    }
	  __syncthreads();
	}

	///Convert to Close Binary Coordinates from Cartesian
	GPUAPI void convert_hel_to_jacobi_coord_without_shared()  { 
	  double tmp = sys[1].mass() / (sys[0].mass() + sys[1].mass());
	  if( is_in_body_component_grid_no_star() )
	    {
	      sys[b][c].pos() += tmp*sys[1][c].pos();
	      sys[b][c].vel() += tmp*sys[1][c].vel();
	    }
	  __syncthreads();

	  double stdcoord_A,stdcoord_B;
	  double sum_masspos = 0., sum_mom = 0., mtot = 0.;
	  double mA = 0., mB = 0., nuA = 0., nuB = 0., momA = 0., momB = 0.;
	  double jacobipos = 0., jacobimom = 0.;
	  if( is_in_body_component_grid() )
	    {
	      //convert Binary Element A's position over to Jacobi
	      stdcoord_A = sys[0][c].pos(); //Star A's cartesian coord
	      stdcoord_B = sys[1][c].pos(); //Star B's cartesian coord
	      mA = sys[0].mass(); //Star A's mass
	      mB = sys[1].mass(); //Star B's mass
	      nuA = mA/(mA+mB); //mass fraction of A
	      nuB = mB/(mA+mB); //mass fraction of B
	      momA = mA * sys[0][c].vel(); //momentum of Star A
	      momB = mB * sys[1][c].vel(); //momentum of Star B
	      
	      //Sum both the mass*pos, mass*vel (momentum) of each planet, total mass of planets, 
	      for(int j=2;j<nbod;j++)
		{
		  const double mj = sys[j].mass();
		  mtot += mj;
		  
		  sum_masspos += mj*sys[j][c].pos();
		  sum_mom += mj*sys[j][c].vel();
		}
	      mtot += mA + mB; //add in star mass to total mass
	      
	      //calculate jacobi position of star A, B and the planet:
	      if (b == 0)
		jacobipos = 0.0;
	      else if (b == 1)
		jacobipos = stdcoord_B - stdcoord_A;
	      else
		jacobipos = sys[b][c].pos() - (nuA*stdcoord_A + nuB*stdcoord_B);
	      __syncthreads();
	      
	      //calculate jacobi/conjugate momenta of the stars A and B, and the planet's:
	      if (b == 0)
		jacobimom = 0.0;
	      else if (b == 1)
		jacobimom = nuA*momB;
	      else
		jacobimom = sys[b].mass() * sys[b][c].vel() - sys[b].mass()*(momA + momB + sum_mom)/mtot;
	    }
	  __syncthreads();

	  if (is_in_body_component_grid())
	    {
	      sys[b][c].pos() = jacobipos; //Finally switch to jacobi coordinates.
	      sys[b][c].vel() = jacobimom / sys[b].mass(); // Coord transforms are done in momentum space. Saving velocity
	    }
	  __syncthreads();
	  
	}

	///Convert back to Cartesian, from Close Binary
        GPUAPI void convert_jacobi_to_hel_coord_without_shared()  { 
	  
	  double mA = sys[0].mass(), mB = sys[1].mass(),
	    nuA = mA / (mA + mB), nuB = mB / (mA + mB), 
	    sum_masspos = 0.,sum_mom = 0.;
	  double CartCoord = 0., StdMom = 0.;
	    
	    if( is_in_body_component_grid() )
	      {
		//Calculate SUM(mj*Jj)
		for(int j = 2;j<nbod;j++)
		  sum_mom += sys[j].mass()*sys[j][c].vel();

		//Calculate Cartesian Coordinates:
		if (b == 0)
		  CartCoord = 0.0;
		else if (b == 1)
		  CartCoord = sys[1][c].pos();
		else
		  CartCoord = sys[b][c].pos() + nuB*sys[1][c].pos();
		  			
		//calculate Momenta in Cartesian Coords
		if (b == 0)
		  StdMom = 0.0;
		else if (b == 1)
		  StdMom = sys[b].mass()*sys[b][c].vel()/nuA;
		else
		  StdMom = sys[b].mass()*(sys[b][c].vel() + 1.0/(mA+mB)*sum_mom + mB/mA*sys[1][c].vel());
		  
	      }
	    __syncthreads();

	    if( is_in_body_component_grid() )
	      {
		sys[b][c].pos() = CartCoord;
		sys[b][c].vel() = StdMom / sys[b].mass();
	      }
	    __syncthreads();

	    double tmp = sys[1].mass() / (sys[0].mass() + sys[1].mass());
	    if( is_in_body_component_grid_no_star() )
	      {
		sys[b][c].pos() -= tmp*sys[1][c].pos();
		sys[b][c].vel() -= tmp*sys[1][c].vel();
	      }
	    __syncthreads();
	}

	/// Standardized member name to call convert_jacobi_to_std_coord_without_shared() 
        GPUAPI void convert_internal_to_std_coord() 
	{ 
	  convert_jacobi_to_hel_coord_without_shared();
	  helio_to_orig();
	  convert_mass_to_solar();
	} 

	/// Standardized member name to call convert_std_to_jacobi_coord_without_shared()
        GPUAPI void convert_std_to_internal_coord() 
	{ 
	  convert_mass_to_gauss();
	  orig_to_helio();
	  convert_hel_to_jacobi_coord_without_shared();
	}

        //Calculate semi-major axis of object using energy
	GPUAPI double calc_sma(int b)
	{
	  double r, v2;
	  double gm, x, y, z, vx, vy, vz;
	  
	  x = sys[b][0].pos();
	  y = sys[b][1].pos();
	  z = sys[b][2].pos();
	  vx = sys[b][0].vel();
	  vy = sys[b][1].vel();
	  vz = sys[b][2].vel();
	  gm = 0.0;
	  
	  if (b == 0)
	    return 0.0;
	  else if (b == 1)
	    gm = sqrtGM * sqrtGM;
	  else
	    gm = MBin + sys[b].mass();
	  
	  r = sqrt(x*x + y*y + z*z);
	  v2 = vx*vx + vy*vy + vz*vz;
	  return gm * r / (2.0 * gm - r * v2);
	}
	
  	//Return minimum semi-major 
	GPUAPI double min_sma()
	{
	  double temp;
	  double minsma = 1.0e30;
	  
	  //Calculate individual sma
	  for (int b = 2; b < nbod; b++)
	    {
	      temp = calc_sma(b);
	      if (temp < minsma)
		minsma = temp;
	    }
	  return minsma;
	}
	
        // Return component sum of momentum
	GPUAPI double mvsum(int c)
	{
	  double mv = 0.0;
	  for (int b = 2; b < nbod; b++)
	    mv += sys[b].mass() * sys[b][c].vel();
	  return mv;
	}
	
        // Return binary terms of acceleration
	GPUAPI double bin_acc(int b, int c)
	{
	  double dx, dy, dz, s_3;
	  double tmp;
	  double x_i, y_i, z_i, x_2, y_2, z_2;
	  double acc = 0.0;

	  x_i = sys[b][0].pos();
	  y_i = sys[b][1].pos();
	  z_i = sys[b][2].pos();
	  x_2 = sys[1][0].pos();
	  y_2 = sys[1][1].pos();
	  z_2 = sys[1][2].pos();
	  
	  s_3 = pow(x_i*x_i + y_i*y_i + z_i*z_i, -1.5);
	  tmp = MBin * s_3;
	  acc += tmp * sys[b][c].pos();
	  
	  dx = MBin*x_i + sys[1].mass() * x_2;
	  dy = MBin*y_i + sys[1].mass() * y_2;
	  dz = MBin*z_i + sys[1].mass() * z_2;
	  s_3 = pow(dx*dx + dy*dy + dz*dz, -1.5);
	  tmp = MBin * MBin * sys[0].mass() * s_3;
	  acc -= tmp*(MBin*sys[b][c].pos() + sys[1].mass() * sys[1][c].pos());
	  
	  dx = MBin*x_i - sys[0].mass() * x_2;
	  dy = MBin*y_i - sys[0].mass() * y_2;
	  dz = MBin*z_i - sys[0].mass() * z_2;
	  s_3 = pow(dx*dx + dy*dy + dz*dz, -1.5);
	  tmp = MBin * MBin * sys[1].mass() * s_3;
	  acc -= tmp*(MBin*sys[b][c].pos() - sys[0].mass() * sys[1][c].pos());
	  
	  return acc;
	}
	
	// Return secondary acceleration components
	GPUAPI double sec_acc(int c)
	{
	  double dx, dy, dz, s_3;
	  double tmp;
	  double x_i, y_i, z_i, x_2, y_2, z_2;
	  double acc = 0.0;
	  
	  for (int ntmp = 2; ntmp < nbod; ntmp++)
	    {
	      x_i = sys[ntmp][0].pos();
	      y_i = sys[ntmp][1].pos();
	      z_i = sys[ntmp][2].pos();
	      x_2 = sys[1][0].pos();
	      y_2 = sys[1][1].pos();
	      z_2 = sys[1][2].pos();
	      
	      dx = MBin*x_i + sys[1].mass() * x_2;
	      dy = MBin*y_i + sys[1].mass() * y_2;
	      dz = MBin*z_i + sys[1].mass() * z_2;
	      s_3 = pow(dx*dx + dy*dy + dz*dz, -1.5);
	      tmp = MBin * sys[0].mass() * sys[ntmp].mass() * s_3;
	      acc -= tmp*(MBin*sys[ntmp][c].pos() + sys[1].mass() * sys[1][c].pos());
	      
	      dx = MBin*x_i - sys[0].mass() * x_2;
	      dy = MBin*y_i - sys[0].mass() * y_2;
	      dz = MBin*z_i - sys[0].mass() * z_2;
	      s_3 = pow(dx*dx + dy*dy + dz*dz, -1.5);
	      tmp = MBin * sys[0].mass() * sys[ntmp].mass() * s_3;
	      acc += tmp*(MBin*sys[ntmp][c].pos() - sys[0].mass() * sys[1][c].pos());
	    }
	  return acc;
	}
	
	// Advance system by one time unit
	GPUAPI void advance()
	{
	  double h = min(_params.time_step, max_timestep);
	  
	  //Steps from John Chamber's Close Binary Propagator outlined in "N-Body Integrators for Planets in Binary Star Systems",
	  //arXiv: 07053223v1
	  
	  ///Advance H, Planet Interaction by 0.5 * timestep
	  if (is_in_body_component_grid_no_star())
	    sys[b][c].vel() += h/2.0 * acc_bc;
	  __syncthreads();
	  
	  ///Repeat NBin Times:
	  for(int NStep = 0; NStep < NBin; NStep++)
	    {
	      //Advance H, Star B Interaction by (0.5 * timestep) / NBin
	      if (b == 1)
		acc_bc = sec_acc(c);
	      else if (is_in_body_component_grid_no_star())
		acc_bc = bin_acc(b,c);
	      //acc_bc = calcForces.acc_binary_cb(ij,b,c);
	      if (is_in_body_component_grid_no_prim())
		sys[b][c].vel() += h/2.0/NBin * acc_bc;
	      __syncthreads();
	      
	      //Advance H, Star B Kep by (0.5 * timestep) / NBin
	      if( is_first_thread_in_system() ) 
		{
		  kep_cb::drift_kepler(sys[1][0].pos(), sys[1][1].pos(), sys[1][2].pos(), sys[1][0].vel(), sys[1][1].vel(), sys[1][2].vel(), sqrtGM, h/2.0/NBin*MBin/sys[0].mass());
		}
	      __syncthreads();
	    }
	  
	  ///Advance H, Jump by 0.5 * timestep
	  mvsum_c = mvsum(c);
	  __syncthreads();
	  if (is_in_body_component_grid_no_star())
	    sys[b][c].pos() += h/2.0/MBin * mvsum_c;
	  __syncthreads();
	  
	  ///Advance H, Planet Kep by timestep
	  if ((ij > 1) && (ij < nbod))
	    kep_cb::drift_kepler(sys[ij][0].pos(), sys[ij][1].pos(), sys[ij][2].pos(), sys[ij][0].vel(), sys[ij][1].vel(), sys[ij][2].vel(), sqrt(MBin), h);
	  __syncthreads();
	  
	  ///Advance H, Jump by 0.5 * timestep
	  mvsum_c = mvsum(c);
	  __syncthreads();
	  if (is_in_body_component_grid_no_star())
	    sys[b][c].pos() += h/2.0/MBin * mvsum_c;
	  __syncthreads();
	  
	  ///Repeat NBin Times:
	  for(int NStep = 0; NStep < NBin; NStep++)
	    {
	      //Advance H, Star B Kep by (0.5 * timestep) / NBin
	      if( is_first_thread_in_system() ) 
		kep_cb::drift_kepler(sys[1][0].pos(), sys[1][1].pos(), sys[1][2].pos(), sys[1][0].vel(), sys[1][1].vel(), sys[1][2].vel(), sqrtGM, h/2.0/NBin*MBin/sys[0].mass());
	      __syncthreads();
	      
	      //Advance H, Star B Interaction by (0.5 * timestep) / NBin
	      if (b == 1)
		acc_bc = sec_acc(c);
	      else if (is_in_body_component_grid_no_star())
		acc_bc = bin_acc(b,c);
	      if (is_in_body_component_grid_no_prim())
		sys[b][c].vel() += h/2.0/NBin * acc_bc;
	      __syncthreads();
	    }
	  
	  ///Advance H, Planet Interaction by 0.5 * timestep
	  acc_bc = calcForces.acc_planets_cb(ij,b,c);
	  if (is_in_body_component_grid_no_star())
	    sys[b][c].vel() += h/2.0 * acc_bc;
	  
	  // Advance time for first thread
	  if( is_first_thread_in_system() ) 
	    sys.time() += h;
	}
      };
    }
  }
}
