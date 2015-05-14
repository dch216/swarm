/*! \file closebinary.hpp
 *   \brief Defines \ref swarm::gpu::bppt::CloseBinaryPropagator - the GPU implementation 
 *          of John Chamber's Close Binary Propagator.
 *
 */

#include "swarm/common.hpp"
#include "swarm/swarmplugin.h"
#include "keplerian.hpp"
#include <math.h>

namespace swarm {

namespace gpu {
namespace bppt {

/*! Paramaters for CloseBinaryPropagator
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
        kepcoords kep_planet_b[nbod-2];
	kepcoords kep_star_B;

	double sqrtGM;
        double MBin;
	double max_timestep;
	double acc_bc[nbod-2];

        int NBin;

        //! Constructor for CloseBinaryPropagator
	GPUAPI CloseBinaryPropagator(const params& p,ensemble::SystemRef& s,
			Gravitation& calc)
		:_params(p),sys(s),calcForces(calc){}

	__device__ bool is_in_body_component_grid()
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
	        MBin = sys[0].mass() + sys[1].mass();
		sqrtGM = sqrt(sys[0].mass() * sys[1].mass() / MBin);
		convert_std_to_jacobi_coord_without_shared();
		__syncthreads();
		if (is_in_body_component_grid_no_star())
		  {
		    //initialize Keplarian coordinates for planet_b:
		    kep_planet_b[b-2].init(b);
		  }
		__syncthreads();
		
		//initialize Keplarian coordinates for star_B:
	       	kep_star_B.init(1);

		//Determine NBin using semi-major axes
		NBin = int(0.5 + pow((min_sma() / calc_sma(1)), 1.5))

	}

	/// Before exiting, convert back to standard cartesian coordinate system
	GPUAPI void shutdown() { 
	convert_jacobi_to_std_coord_without_shared();
	}

	///Convert to Jacobi Coordinates from Cartesian
	GPUAPI void convert_std_to_jacobi_coord_without_shared()  { 
	    double stdcoord_A,stdcoord_B;
		double sum_masspos = 0., sum_mom = 0., mtot = 0.;
		double mA = 0., mB = 0., nuA = 0., nuB = 0., momA = 0., momB = 0.;
		double jacobipos[nbod] = 0., jacobimom[nbod] = 0.;
		if( is_in_body_component_grid() )
		{
			//convert Binary Element A's position over to Jacobi
			stdcoord_A = sys[0][c].pos(); //Star A's cartesian coord
			stdcoord_B = sys[1][c].pos(); //Star B's cartesian coord
			mA = sys[0].mass(); //Star A's mass
			mB = sys[1].mass() //Star B's mass
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
			jacobipos[0] = (sum_masspos + stdcoord_A*mA + stdcoord_B*mB)/mtot;
			jacobipos[1] = stdcoord_B - stdcoord_A;
			jacobipos[b] = sys[b][c].pos() - (nuA*stdcoord_A + nuB*stdcoord_B);
			
			//calculate jacobi/conjugate momenta of the stars A and B, and the planet's:
			jacobimom[0] = momA + momB + sum_mom;
			jacobimom[1] = momB - nuB * (momA - momB);
			jacobimom[b] = sys[b].mass() * sys[b][c].vel() - sys[b].mass()*(momA + momB + sum_mom)/mtot;
	
			}
		}
		__syncthreads();

		if( is_in_body_component_grid() )
		{
			sys[b][c].pos() = jacobipos[b]; //Finally switch to jacobi coordinates.
			sys[b][c].vel() = jacobimom / sys[b].mass(); // Coord transforms are done in momentum space. Saving velocity
		}
	}

	///Convert back to Cartesian, from Jacobi
	GPUAPI void convert_jacobi_to_std_coord_without_shared()  { 
		
		double JPos_A = sys[0][c].pos(),
			   JPos_B = sys[1][c].pos(),
			   mA = sys[0].mass(),
			   mB = sys[1].mass(),
			   mplan = 0.,
			   mtot = 0.,
			   nuA = mA / (mA + mB),
			   nuB = mB / (mA + mB),
			   sum_masspos = 0.,
			   momA = 0.,
			   momB =0.,
			   sum_mom = 0.;
		double CartCoord_A = 0.,
			   CartCoord_B = 0.,
			   CartCoord_planet = 0.,
			   StdMom_A = 0.,
			   StdMom_B = 0.,
			   StdMom_planet = 0.;
			   
		if( is_in_body_component_grid() )
		{
			//Calculate SUM(mj*Jj)
			for(int j = 2;j<nbod;j++)
			{
				sum_masspos += sys[j].mass()*sys[j][c].pos();
				mplan += sys[j].mass();
				sum_mom += sys[j].mass()*sys[j][c].vel();
			}
			mtot = mA + mB + mplan;
			
			//Calculate Cartesian Coordinates:
			CartCoord_A = (JPos_A*mtot - (mB +mplan*nuB)*JPos_B - sum_masspos) / (mA + mB + mplan*(nuA + nuB));
			CartCoord_B = JPos_B + CartCoord_A;
			CartCoord_planet = sys[b][c].pos() + nuA*CartCoord_A + nuB*CartCoord_B;
			
			//calculate Momenta in Cartesian Coords
			StdMom_A = (1.0 - nuB) * ((1.0-mplan/mtot)*sys[0][c].vel()*mA - sum_mom - (sys[1][c].vel()*mB)/(1.0-nuB));
			StdMom_B = (sys[1][c].vel()*mB + nuB*StdMom_A)/(1.0-nuB);
			StdMom_planet = sys[b][c].vel()*sys[b].mass() + (sys[b].mass/mtot)*sys[0][c].vel()*mA;
			
		}
		__syncthreads();

		if( is_in_body_component_grid() )
		{
			sys[0][c].pos() = CartCoord_A;
			sys[1][c].pos() = CartCoord_B;
			sys[b][c].pos() = CartCoord_planet;
			
			sys[0][c].vel() = StdMom_A / mA;
			sys[1][c].vel() = StdMom_B / mB;
			sys[b][c].vel() = StdMom_planet / sys[b].mass();
		}

	}

	/// Standardized member name to call convert_jacobi_to_std_coord_without_shared() 
	GPUAPI void convert_internal_to_std_coord() 
	{ convert_jacobi_to_std_coord_without_shared();	} 

	/// Standardized member name to call convert_std_to_jacobi_coord_without_shared()
        GPUAPI void convert_std_to_internal_coord() 
	{ convert_std_to_jacobi_coord_without_shared(); }

	//Keplerian Coordinates Class
	class kepcoords
	{
	  //Constants of Use:
	  double eccentricity,semiMajorAxis, inclination,longitude,argumentP,meanMotion,MeanAnomaly;
	  double rx,ry,rz,vx,vy,vz,lx,ly,lz;
	  double GravC; // use sqrtGM!
	  double GM = sqrtGM*sqrtGM;
	  double ecc_x,ecc_y,ecc_z; // Eccentricity Vector
	  
	  public:
	  //Calculate new (?) coordinates
	  void init(double bodynum)
	  {		
	    //Intermediate Variables:
	    double specE,rad_dist,L2,mu; //Specific Energy, radial distance, (ang momentum)^2, reduced mass
	    double cen_mass[3]; // Center of Mass
	       
	    //Calculate Center of Mass:
	    //X:
	    cen_mass[0] = (sys[0].mass()*sys[0][0].pos() + sys[1].mass()*sys[1][0].pos())/(sys[0].mass()+sys[1].mass());
	    //Y:
	    cen_mass[1] = (sys[0].mass()*sys[0][1].pos() + sys[1].mass()*sys[1][1].pos())/(sys[0].mass()+sys[1].mass());
	    //Z:
	    cen_mass[2] = (sys[0].mass()*sys[0][2].pos() + sys[1].mass()*sys[1][2].pos())/(sys[0].mass()+sys[1].mass());
	    
	    //Calculate Reduced Mass ((mA+mB)*mPlanet)/(mA+mB+mPlanet)
	    mu = ((sys[0].mass() + sys[1].mass())*sys[bodynum].mass())/(sys[0].mass()+sys[1].mass()+sys[bodynum].mass());
	    
	    //Calculate Radial Distance from barycenter:
	    rad_dist = sqrt(pow(sys[b][0].pos()-cen_mass[0],2) + pow(sys[b][1].pos()-cen_mass[1],2) + pow(sys[b][2].pos()-cen_mass[2],2));
	    //Calculate Specific Orbital Energy
	    specE = (pow(sys[b][0].vel(),2) + pow(sys[b][1].vel(),2) + pow(sys[b][1].vel(),2))/2.0 - GravC*(sys[b].mass() + sys[0].mass())/rad_dist;
	    //Calculate L^2
	    L2 = pow(sys[b].mass(),2)*
	      (pow(sys[b][1].pos()*sys[b][2].vel() - sys[b][2].pos()*sys[b][2].vel(),2) +
	       pow(sys[b][2].pos()*sys[b][0].vel() - sys[b][0].pos()*sys[b][2].vel(),2) +
	       pow(sys[b][0].pos()*sys[b][1].vel() - sys[b][1].pos()*sys[b][0].vel(),2));
	    //Calculate eccentricity
	    eccentricity = sqrt(1.0+(2.0*specE*L2/pow(mu,2)));
	    //Calculate SemiMajorAxis
	    semiMajorAxis = GravC * (sys[b].mass() + sys[0].mass()) / (2.0*eccentricity);
	    //Calculate Inclination
	    //vector components
	    rx = sys[b][0].pos() - cen_mass[0];
	    ry = sys[b][1].pos() - cen_mass[1];
	    rz = sys[b][2].pos() - cen_mass[2];
	    vx = sys[b][0].vel();
	    vy = sys[b][0].vel();
	    vz = sys[b][0].vel();
	    lx = sys[b].mass()*(ry*vz - rz*vy);
	    ly = sys[b].mass()*(rz*vx - rx*vz);
	    lz = sys[b].mass()*(rx*vy - ry*vx);
	    lmag = sqrt(lx*lx + ly*ly + lz*lz); //ERROR HANDLE use stderr / cerr if lmag == 0?
	    inclination = acos(lz/lmag); // in radians
	    //Calculate Longitude of the Ascending Node, with reference direction (+1,0,0)
	    longitude = acos(-ly/sqrt(ly*ly+lx+lx));
	    //Calculate Argument of Periapsis
	    //FIRST: Eccentricity Vector Components:
	    ecc_x = (sqrt(vx*vx+vy*vy+vz*vz)/GM)*rx - ((rx*vx+ry*vy+rz*vz)/GM)*vx - (1.0/sqrt(rx*rx+ry*ry+rz*rz))*rx;
	    ecc_y = (sqrt(vx*vx+vy*vy+vz*vz)/GM)*ry - ((rx*vx+ry*vy+rz*vz)/GM)*vy - (1.0/sqrt(rx*rx+ry*ry+rz*rz))*ry;
	    ecc_z = (sqrt(vx*vx+vy*vy+vz*vz)/GM)*rz - ((rx*vx+ry*vy+rz*vz)/GM)*vz - (1.0/sqrt(rx*rx+ry*ry+rz*rz))*rz;
	    
	    if(inclination == 0)
	      argumentP = atan2(ecc_y,ecc_x); // If orbit is equitorial
	    else
	      argumentP = acos(ecc_x/sqrt(ecc_x*ecc_x+ecc_y*ecc_y+ecc_z*ecc*z)); // For cases where orbit is NOT equitorial. Here we define the ascending node as before (+1,0,0)
	    
	    //If clockwise orbit, ecc_z < 0. Handle this case:
	    if (ecc_z < 0)
	      argumentP = 2.0*3.14159265358979323846 - argumentP;
	    
	    //Calculate Mean Motion
	    meanMotion = sqrt((GravC * (sys[0].mass()+sys[1].mass()+sys[b].mass()))/pow(semiMajorAxis,3));
	  }
	  
	  ///Return & Write Functions:
	  //eccentricity
	  double ecc()
	  {
	      return eccentricity;
	  }
	  void w_ecc(double in_ecc)
	  {
	    eccentricity = in_ecc;
	  }

	  //Semi Major Axis
	  double sma()
	  {
	      return semiMajorAxis;
	  }
	  void w_sma(double in_sma)
	  {
	    semiMajorAxis = in_sma;
	  }

	  //inclination
	  double inc()
	  {
	      return inclination;
	  }
	  void w_inc(double in_inc)
	  {
	    inclination = in_inc;
	  }

	  //Longitude of Ascending Node
	  double lon()
	  {
	      return longitude;
	  }
	  void w_lon(double in_lon)
	  {
	    longitude = in_lon;
	  }
	  
	  //Argument of Periapsis
	  double arg()
	  {
	    return argumentP;
	  }
	  void w_arg(double in_arg)
	  {
	    argumentP = in_arg;
	  }

	  //Mean Motion
	  double mm()
	  {
	    return meanMotion;
	  }
	  //Mean motion is a constant, no writing necessary
	  
	}

	//Convert kepcoords BACK into Cartesian
	  GPUAPI void convert_kep_to_cart(kepcoords kep_body)
	  {
	    //Calculate position

	    //Calculate velocity

	    ///Write Values:
	    sys[b][0].pos() = rx;
	    sys[b][1].pos() = ry;
	    sys[b][2].pos() = rz;
	    sys[b][0].vel() = vx;
	    sys[b][1].vel() = vy;
	    sys[b][2].vel() = vz;
	  }

        //Calculate semi-major axis of object using energy
          GPUAPI double calc_sma(int b)
	  {
	    double r, v2, sma;
	    double x, y, z, vx, vy, vz;

	    x = sys[b][0].pos();
	    y = sys[b][1].pos();
	    z = sys[b][2].pos();
	    vx = sys[b][0].vel();
	    vy = sys[b][1].vel();
	    vz = sys[b][2].vel();

	    r = sqrt(x*x + y*y + z*z);
	    v2 = vx*vx + vy*vy + vz*vz;
	    return sqrtGM*sqrtGM * r / (2.0 * sqrtGM * sqrtGM - r * v2);
	  }

  	//Return minimum semi-major 
	  GPUAPI double min_sma()
	  {
	    double temp, minsma = 1.0e30;
	    
	    //Calculate individual sma
            for (int b = 2; b < nbod; b++)
	      {
		temp = calc_sma(b);
		if (temp < minsma)
		  minsma = temp;
	      }
	    return minsma;
	  }

        //Return component sum of momentum
          GPUAPI double mvsum(int c)
	  {
	    double mv = 0.0;
	    for (int b = 2; b < nbod; b++)
	      {
		mv += sys[b].mass() * sys[b][c].vel();
	      }
	    return mv;
	  }

	// Advance system by one time unit
	  GPUAPI void advance()
	  {
	    double h = min(_params.time_step, max_timestep);

	    //Steps from John Chamber's Close Binary Propagator outlined in "N-Body Integrators for Planets in Binary Star Systems",
	    //arXiv: 07053223v1

    
	    ///Advance H, Planet Interaction by 0.5 * timestep
	    if (is_in_body_component_grid_no_star())
	      {
		sys[b][c].vel() += h/2.0 * calcForces.acc_planets_cb(ij,b,c);
	      }
	    
	    __syncthreads();
	    ///Repeat NBin Times:
	    for(int NStep = 0; NStep < NBin; NStep++)
	      {
		if (is_in_body_component_grid())
		  {
		    //Advance H, Star B Interaction by (0.5 * timestep) / NBin
		    sys[b][c].vel() += h/2.0/NBin*calcForces.acc_binary_cb(ij,b,c);
		  }
		
		    //Advance H, Star B Kep by (0.5 * timestep) / NBin
		drift_kepler(sys[1][0].pos(), sys[1][1].pos(), sys[1][2].pos(), sys[1][0].vel(), sys[1][1].vel(), sys[1][2].vel(), sqrtGM, h/2.0/NBin*MBin/sys[0].mass());	  
	      }
	    __syncthreads();
	    
	    ///Advance H, Jump by 0.5 * timestep
	    if (is_in_body_component_grid_no_star())
	      {
		sys[b][c].pos() += h/2.0/MBin * mvsum(c);
	      }
	    __syncthreads();
	    
	    ///Advance H, Planet Kep by timestep
	    if (is_in_body_component_grid_no_star())
	      {
		drift_kepler(sys[b][0].pos(), sys[b][1].pos(), sys[b][2].pos(), sys[b][0].vel(), sys[b][1].vel(), sys[b][2].vel(), sqrt(MBin), h);
	      }
	    __syncthreads();
	    
	    ///Advance H, Jump by 0.5 * timestep
	    	    if (is_in_body_component_grid_no_star())
	      {
		sys[b][c].pos() += h/2.0/MBin * mvsum(c);
	      }
	    __syncthreads();
	    
	    ///Repeat NBin Times:
	    for(int NStep = 0; NStep < NBin; NStep++)
	      {
		//Advance H, Star B Kep by (0.5 * timestep) / NBin
		drift_kepler(sys[1][0].pos(), sys[1][1].pos(), sys[1][2].pos(), sys[1][0].vel(), sys[1][1].vel(), sys[1][2].vel(), sqrtGM, h/2.0/NBin*MBin/sys[0].mass());
		
		if (is_in_body_component_grid())
		  {
		    //Advance H, Star B Interaction by (0.5 * timestep) / NBin
		    sys[b][c].vel() += h/2.0/NBin*calcForces.acc_binary_cb(ij,b,c);
		  }  
	      }
	    __syncthreads();
	    
	    ///Advance H, Planet Interaction by 0.5 * timestep
	    if (is_in_body_component_grid_no_star())
	      {
		sys[b][c].vel() += h/2.0 * calcForces.acc_planets_cb(ij,b,c);
	      }

	    // Advance time for first thread
	    if( is_first_thread_in_system() ) 
	      sys.time() += h;
	  }
};
  



}
}
}
