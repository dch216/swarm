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

#include "swarm/common.hpp"
#include "swarm/gpu/bppt.hpp"
#include "monitors/stop_on_ejection.hpp"


namespace swarm {

namespace gpu {
namespace bppt {

struct FixedTimeStep {
	const static bool adaptive_time_step = false;
	const static bool conditional_accept_step = false;
};

struct AdaptiveTimeStep {
	const static bool adaptive_time_step = true;
	const static bool conditional_accept_step = true;
};

/*! Runge Kutta Cash Karp integrator Fixed/Adaptive
 *
 * \ingroup integrators
 *
 *  This integrator comes in two flavors: Fixed time step, Adaptive time step
 *
 *
 */
template< class AdaptationStyle, template<class L> class Monitor >
class rkck: public integrator {
	typedef integrator base;
	typedef  Monitor<gpulog::device_log> monitor_t;
	typedef  typename monitor_t::params mon_params_t;
	private:
	double _min_time_step;
	double _max_time_step;
	double _error_tolerance;
	int _iteration_count;
	mon_params_t _mon_params;

	public:
	rkck(const config& cfg): base(cfg),_min_time_step(0.001),_max_time_step(0.1), _mon_params(cfg) {
		if(!cfg.count("min time step")) ERROR("Integrator rkck requires a min timestep ('min time step' keyword in the config file).");
		_min_time_step = atof(cfg.at("min time step").c_str());
		if(!cfg.count("max time step")) ERROR("Integrator rkck requires a max timestep ('max time step' keyword in the config file).");
		_max_time_step = atof(cfg.at("max time step").c_str());

		if(!cfg.count("error tolerance")) ERROR("Integrator rkck requires a error tolerance ('error tolerance' keyword in the config file).");
		_error_tolerance = atof(cfg.at("error tolerance").c_str());
	}

	virtual void launch_integrator() {
		_iteration_count = _destination_time / _max_time_step;
		launch_templatized_integrator(this);
	}


	template<class T>
	__device__ void kernel(T compile_time_param){

////////////////////// RKCK Constants /////////////////////////////
	// Cash-Karp constants From GSL
	// Step 1 coefficient
	const double b1 = 1.0 / 5.0;
	// Step 2 coefficient
	const double b2[]  = { 3.0 / 40.0, 9.0 / 40.0 };
	// Step 3 coefficient
	const double b3[]  = { 0.3, -0.9, 1.2 };
	// Step 4 coefficient
	const double b4[]  = { -11.0 / 54.0, 2.5, -70.0 / 27.0, 35.0 / 27.0 };
	// Step 5 coefficient
	const double b5[]  = { 1631.0 / 55296.0, 175.0 / 512.0, 575.0 / 13824.0, 44275.0 / 110592.0, 253.0 / 4096.0 };
	// Step 6 coefficient
	const double b6[]  = { 37.0 / 378.0, 0, 250.0 / 621.0, 125.0 / 594.0, 0 , 512.0 / 1771.0 } ;
	// Error estimation coefficients
	const double ecc[] = { 37.0 / 378.0 - 2825.0 / 27648.0, 0.0, 250.0 / 621.0 - 18575.0 / 48384.0, 125.0 / 594.0 - 13525.0 / 55296.0, -277.00 / 14336.0, 512.0 / 1771.0 - 0.25 };

		if(sysid()>=_dens.nsys()) return;
		// References to Ensemble and Shared Memory
		ensemble::SystemRef sys = _dens[sysid()];
//		typedef typename Gravitation<T::n>::shared_data grav_t;
//		GravitationT::n> calcForces(sys,*( (grav_t*) system_shared_data_pointer(compile_time_param) ) );
		typedef typename GravitationAccOnly<T::n>::shared_data grav_t;
//		GravitationAccOnly<T::n> calcForces(sys,*( (grav_t*) system_shared_data_pointer(compile_time_param) ) );
		GravitationAccOnly<T::n> calcForces(sys,sysid_in_block());

		// Local variables
		const int nbod = T::n;
		// Body number
		int b = thread_body_idx(nbod);
		// Component number
		int c = thread_component_idx(nbod);
		int ij = thread_in_system();
		bool body_component_grid = (b < nbod) && (c < 3);
		bool first_thread_in_system = thread_in_system() == 0;


		// local variables
		monitor_t montest(_mon_params,sys,*_log) ;

		// NB: We use the same shared memory for two purpose and overwrite each other
		extern __shared__ char shared_mem[];
		char*  system_shmem =( shared_mem + sysid_in_block() * shmem_per_system(nbod) );

		// TODO: used Coalesced array structure
		double (&shared_mag)[2][nbod][3] = * (double (*)[2][nbod][3]) system_shmem;

		double time_step = _max_time_step;

		// local information per component per body
		double pos = 0, vel = 0 ;
		if( body_component_grid )
			pos = sys[b][c].pos() , vel = sys[b][c].vel();


		////////// INTEGRATION //////////////////////

		for(int iter = 0 ; (iter < _iteration_count) && sys.is_active() ; iter ++ ) {

			double h = time_step;

			if( sys.time() + h > _destination_time ) {
				h = _destination_time - sys.time();
			}

			//// RKCK   integrate system  ////////////////////////////////////////////////////////////////
			double p0 = pos, v0 = vel;

			// Step 1
			double k1_acc = calcForces.acc(ij,b,c,p0,v0);
			double k1_vel = v0;

			double p1 = pos + h * b1 * k1_vel;
			double v1 = vel + h * b1 * k1_acc;

			// Step 2
			double k2_acc = calcForces.acc(ij,b,c,p1,v1);
			double k2_vel = v1;

			double p2 = pos + h * ( b2[0] * k1_vel + b2[1] * k2_vel );
			double v2 = vel + h * ( b2[0] * k1_acc + b2[1] * k2_acc );

			// Step 3
			double k3_acc = calcForces.acc(ij,b,c,p2,v2);
			double k3_vel = v2;

			double p3 = pos + h * ( b3[0] * k1_vel + b3[1] * k2_vel + b3[2] * k3_vel );
			double v3 = vel + h * ( b3[0] * k1_acc + b3[1] * k2_acc + b3[2] * k3_acc );

			// Step 4
			double k4_acc = calcForces.acc(ij,b,c,p3,v3);
			double k4_vel = v3;

			double p4 = pos + h * ( b4[0] * k1_vel + b4[1] * k2_vel + b4[2] * k3_vel + b4[3] * k4_vel );
			double v4 = vel + h * ( b4[0] * k1_acc + b4[1] * k2_acc + b4[2] * k3_acc + b4[3] * k4_acc );

			// Step 5
			double k5_acc = calcForces.acc(ij,b,c,p4,v4);
			double k5_vel = v4;

			double p5 = pos + h * ( b5[0] * k1_vel + b5[1] * k2_vel + b5[2] * k3_vel + b5[3] * k4_vel + b5[4] * k5_vel );
			double v5 = vel + h * ( b5[0] * k1_acc + b5[1] * k2_acc + b5[2] * k3_acc + b5[3] * k4_acc + b5[4] * k5_acc );

			// Step 6
			double k6_acc = calcForces.acc(ij,b,c,p5,v5);
			double k6_vel = v5;

			double p6 = pos + h * ( b6[0] * k1_vel + b6[1] * k2_vel + b6[2] * k3_vel + b6[3] * k4_vel + b6[4] * k5_vel + b6[5] * k6_vel );
			double v6 = vel + h * ( b6[0] * k1_acc + b6[1] * k2_acc + b6[2] * k3_acc + b6[3] * k4_acc + b6[4] * k5_acc + b6[5] * k6_acc );


			// Error estimate
			double pos_error = h * ( ecc[0] * k1_vel + ecc[1] * k2_vel + ecc[2] * k3_vel + ecc[3] * k4_vel + ecc[4] * k5_vel + ecc[5] * k6_vel );
			double vel_error = h * ( ecc[0] * k1_acc + ecc[1] * k2_acc + ecc[2] * k3_acc + ecc[3] * k4_acc + ecc[4] * k5_acc + ecc[5] * k6_acc );


			bool accept_step = true;

			if( AdaptationStyle::adaptive_time_step ) {
				////////////////////////  Adapting Time step algorithm /////////////////////////////
				const int   integrator_order = 5;
				//! Value used as power in formula to produce larger time step
				const float step_grow_power = -1./(integrator_order+1.);
				//! Value used as power in formula to produce smaller time step
				const float step_shrink_power = -1./integrator_order;
				//! Safety factor to prevent extreme changes in time step
				const float step_guess_safety_factor = 0.9;
				//! Maximum growth of step size allowed at a time
				const float step_grow_max_factor = 5.0; 
				//! Maximum shrinkage of step size allowed at a time
				const float step_shrink_min_factor = 0.2; 

				//  Calculate the error estimate
				if( body_component_grid ) {

					sys[b][c].pos() = p6 * p6 , sys[b][c].vel() = v6 * v6;
					shared_mag[0][b][c] = pos_error * pos_error;
					shared_mag[1][b][c] = vel_error * vel_error;

					__syncthreads();
					if ( (c == 0) && (b == 0) ) {

						double max_error = 0;
						for(int i = 0; i < nbod ; i++){
							double pos_error_mag = shared_mag[0][i][0] + shared_mag[0][i][1] + shared_mag[0][i][2];
							double pos_mag = sys[i][0].pos() + sys[i][1].pos() + sys[i][2].pos();
							double pe = pos_error_mag / pos_mag ;

							double vel_error_mag = shared_mag[1][i][0] + shared_mag[1][i][1] + shared_mag[1][i][2];
							double vel_mag = sys[i][0].vel() + sys[i][1].vel() + sys[i][2].vel();
							double ve = vel_error_mag / vel_mag ;

							max_error = max ( max( pe, ve) , max_error );
						}

						double normalized_error = max_error / _error_tolerance;

						// Calculate New time_step
						double step_guess_power = (normalized_error<1.) ? step_grow_power : step_shrink_power;

						/// factor of 0.5 below due to use of squares in calculate_normalized_error, should we change to match gsl?
						/// gsl uses 1.1, but that seems dangerous, any reason we shouldn't use 1?
						double step_change_factor = ((normalized_error<0.5)||(normalized_error>1.0)) ? step_guess_safety_factor*pow(normalized_error,0.5*step_guess_power) : 1.0;


						//// Update the time step
						double new_time_step = (normalized_error>1.) ? max( time_step * max(step_change_factor,step_shrink_min_factor), _min_time_step ) 
							: min( time_step * max(min(step_change_factor,step_grow_max_factor),1.0), _max_time_step );

						bool accept = ( normalized_error < 1.0 ) || (abs(time_step - new_time_step) < 1e-10) ;

						shared_mag[0][0][0] = accept ? 0.0 : 1.0;
						shared_mag[0][0][1] = new_time_step;
					}

				}
				__syncthreads();

				time_step = shared_mag[0][0][1];
				accept_step = AdaptationStyle::conditional_accept_step ? (shared_mag[0][0][0] == 0.0) : true;
				////////////////////////// End of Adaptive time step algorithm  ////////////////////////////////////////////
			}


			if ( accept_step ) {
				// Set the new positions and velocities and time
				pos = p6;
				vel = v6;

				// Finalize the step
				if( body_component_grid )
					sys[b][c].pos() = pos , sys[b][c].vel() = vel;
				if( first_thread_in_system ) 
					sys.time() += h;

				if( first_thread_in_system )  {
					montest();
					if( sys.time() >= _destination_time ) 
						sys.set_inactive();
				}
			}

			__syncthreads();

		}

	}


};

integrator_plugin_initializer<rkck< AdaptiveTimeStep, monitors::stop_on_ejection> >
	rkck_adaptive_plugin("rkck_adaptive");

integrator_plugin_initializer<rkck< FixedTimeStep, monitors::stop_on_ejection> >
	rkck_Fixed_plugin("rkck_fixed");

}
}
}
