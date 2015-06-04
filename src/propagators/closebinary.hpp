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
        //kepcoords kep_planet_b[nbod-2];
	//kepcoords kep_star_B;

        double coord_shift[3];
        double vel_shift[3];

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
	  /*
	  coord_shift[0] = sys[0][0].pos();
	  coord_shift[1] = sys[0][1].pos();
	  coord_shift[2] = sys[0][2].pos();
	  vel_shift[0] = sys[0][0].vel();
	  vel_shift[1] = sys[0][1].vel();
	  vel_shift[2] = sys[0][2].vel();
	  
	  bary_to_prim();
	  */
	  convert_std_to_jacobi_coord_without_shared();
	  acc_bc = calcForces.acc_planets_cb(ij,b,c);
      	        
	  /*
	    if (is_in_body_component_grid_no_star())
	    {
	    //initialize Keplarian coordinates for planet_b:
	    kep_planet_b[b-2].init(b);
	    }
	    __syncthreads();
	    
	    
	    //initialize Keplarian coordinates for star_B:
	    kep_star_B.init(1);
	  */
	  
	  //Determine NBin using semi-major axes
	  NBin = int(0.5 + pow((min_sma() / calc_sma(1)), 1.5));
	}

	/// Before exiting, convert back to standard cartesian coordinate system
	GPUAPI void shutdown() { 
	convert_jacobi_to_std_coord_without_shared();
	//prim_to_bary();
	convert_mass_to_solar();
	}

        ///Convert mass to Gaussian units
        GPUAPI void convert_mass_to_gauss()   {
	  if ( is_in_body_component_grid() )
	    sys[b].mass() *= 2.959122082855911e-04;
	  
	  __syncthreads();

        }

        ///Convert Gaussian mass back to solar units
        GPUAPI void convert_mass_to_solar()   {
	  if ( is_in_body_component_grid() )
	    sys[b].mass() /= 2.959122082855911e-04;

	  __syncthreads();

        }

        ///Move origin from barycenter to primary
        GPUAPI void bary_to_prim() {
	  if( is_in_body_component_grid() )
	    {
	      sys[b][c].pos() -= coord_shift[c];
	      sys[b][c].vel() -= vel_shift[c];
	    }
	  __syncthreads();
	}

      ///Move origin from primary to barycenter
        GPUAPI void prim_to_bary() {
	  if( is_in_body_component_grid() )
	    {
	      sys[b][c].pos() += coord_shift[c];
	      sys[b][c].vel() += vel_shift[c];
	    }
	  __syncthreads();
	}

	///Convert to Jacobi Coordinates from Cartesian
	GPUAPI void convert_std_to_jacobi_coord_without_shared()  { 
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
	      __syncthreads();
	      
	      //calculate jacobi position of star A, B and the planet:
	      if (b == 0)
		jacobipos = (sum_masspos + stdcoord_A*mA + stdcoord_B*mB)/mtot;
	      else if (b == 1)
		jacobipos = stdcoord_B - stdcoord_A;
	      else
		jacobipos = sys[b][c].pos() - (nuA*stdcoord_A + nuB*stdcoord_B);
	      __syncthreads();
	      
	      //calculate jacobi/conjugate momenta of the stars A and B, and the planet's:
	      if (b == 0)
		jacobimom = momA + momB + sum_mom;
	      else if (b == 1)
		jacobimom = momB - nuB * (momA + momB);
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

	///Convert back to Cartesian, from Jacobi
        GPUAPI void convert_jacobi_to_std_coord_without_shared()  { 
	  
	  double JPos_A = 0., JPos_B = 0.,
	    mA = sys[0].mass(), mB = sys[1].mass(), mplan = 0., mtot = 0.,
	    nuA = mA / (mA + mB), nuB = mB / (mA + mB), sum_masspos = 0.,
	    momA = 0., momB =0., sum_mom = 0.,
	    tmp1 = 0., tmp2 = 0., tmp3 = 0., tmp4 = 0.;
	  double CartCoord = 0., StdMom = 0.;
	    
	    if( is_in_body_component_grid() )
	      {
		JPos_A = sys[0][c].pos();
		JPos_B = sys[1][c].pos();
		
		//Calculate SUM(mj*Jj)
		for(int j = 2;j<nbod;j++)
		  {
		    sum_masspos += sys[j].mass()*sys[j][c].pos();
		    mplan += sys[j].mass();
		    sum_mom += sys[j].mass()*sys[j][c].vel();
		  }
		mtot = mA + mB + mplan;
		__syncthreads();

		tmp1 = (JPos_A*mtot - (mB +mplan*nuB)*JPos_B - sum_masspos) / (mA + mB + mplan*(nuA + nuB));
		tmp2 = ((1.0 - nuB) * ((1.0-(mplan/mtot))*sys[0][c].vel()*mA - sum_mom) - (sys[1][c].vel()*mB));
		tmp3 = ((sys[1][c].vel()*mB)+(nuB*tmp2))/(1.0-nuB);
		tmp4 = (sum_mom + (mplan/mtot)*(tmp2+tmp3))/(1.0-(mplan/mtot));
		__syncthreads;
		
		//Calculate Cartesian Coordinates:
		if (b == 0)
		  CartCoord = tmp1;
		else if (b == 1)
		  CartCoord = JPos_B + tmp1;
		else
		  CartCoord = sys[b][c].pos() + nuA*tmp1 + nuB*tmp2;
			
		//calculate Momenta in Cartesian Coords
		if (b == 0)
		  StdMom = tmp2;
		else if (b == 1)
		  StdMom = tmp3;
		else
		  StdMom = sys[b][c].vel()*sys[b].mass() + (sys[b].mass()/mtot)*(tmp2+tmp3+tmp4);
			
	      }
	    __syncthreads();

	    if( is_in_body_component_grid() )
	      {
		sys[b][c].pos() = CartCoord;
		sys[b][c].vel() = StdMom / sys[b].mass();
	      }
	    __syncthreads();
	}

	/// Standardized member name to call convert_jacobi_to_std_coord_without_shared() 
        GPUAPI void convert_internal_to_std_coord() 
	{ convert_jacobi_to_std_coord_without_shared();	} 

	/// Standardized member name to call convert_std_to_jacobi_coord_without_shared()
        GPUAPI void convert_std_to_internal_coord() 
	{ convert_std_to_jacobi_coord_without_shared(); }

  /*
	//Keplerian Coordinates Class
	class kepcoords
	{
	  //Constants of Use:
	  double eccentricity,semiMajorAxis, inclination,longitude,argumentP,meanMotion,MeanAnomaly;
	  double rx,ry,rz,vx,vy,vz,lx,ly,lz;
	  double GravC; // use sqrtGM!
	  double GM;
	  double ecc_x,ecc_y,ecc_z; // Eccentricity Vector
	  
	  public:
	  //Calculate new (?) coordinates
	  void init(double bodynum)
	  {		
            GM = sqrtGM*sqrtGM;

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
	//  GPUAPI void convert_kep_to_cart(kepcoords kep_body)
	//  {
	    //Calculate position

	    //Calculate velocity

	    ///Write Values:
	//    sys[b][0].pos() = rx;
	//    sys[b][1].pos() = ry;
	//    sys[b][2].pos() = rz;
	//    sys[b][0].vel() = vx;
	//    sys[b][1].vel() = vy;
	//    sys[b][2].vel() = vz;
	//  }
	*/	

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
	      {
		mv += sys[b].mass() * sys[b][c].vel();
	      }
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
		/*
		if (b == 1)
		  {
		    acc_bc = calcForces.acc_planets_cb(ij,b,c);
		    sys[b][c].vel() += h/2.0/NBin * acc_bc;
		  }
		__syncthreads();
		*/
		
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
		  kep_cb::drift_kepler(sys[1][0].pos(), sys[1][1].pos(), sys[1][2].pos(), sys[1][0].vel(), sys[1][1].vel(), sys[1][2].vel(), sqrtGM, h/2.0/NBin*MBin/sys[0].mass());	
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
		/*
		if (b == 1)
		  {
		    acc_bc = calcForces.acc_planets_cb(ij,b,c);
		    sys[b][c].vel() += h/2.0/NBin * acc_bc;
		  }
		__syncthreads();
		*/
		if (b == 1)
		  acc_bc = sec_acc(c);
		else if (is_in_body_component_grid_no_star())
		  acc_bc = bin_acc(b,c);
		//acc_bc = calcForces.acc_binary_cb(ij,b,c);
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
