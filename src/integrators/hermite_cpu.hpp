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

/*! \file hermite_cpu.hpp
 *   \brief Defines and implements \ref swarm::cpu::hermite_cpu class - the 
 *          CPU implementation of PEC2 Hermite integrator.
 *
 */

#ifdef _OPENMP
#include <omp.h>
#endif


#include "swarm/common.hpp"
#include "swarm/integrator.hpp"
#include "swarm/plugin.hpp"

namespace swarm { namespace cpu {
/*! CPU implementation of PEC2 Hermite integrator
 *
 * \ingroup integrators
 *
 *   This is used as a reference implementation to
 *   test other GPU implementations of other integrators
 *   
 *   This integrator can be used as an example of CPU integrator
 *
 */
template< class Monitor >
class hermite_cpu : public integrator {
	typedef integrator base;
	typedef Monitor monitor_t;
	typedef typename monitor_t::params mon_params_t;
	private:
	double _time_step;
	mon_params_t _mon_params;

public:  //! Construct for hermite_cpu class
	hermite_cpu(const config& cfg): base(cfg),_time_step(0.001), _mon_params(cfg) {
		_time_step =  cfg.require("time_step", 0.0);
	}

	virtual void launch_integrator() {
		#ifdef _OPENMP
		#pragma omp parallel for
		#endif
		for(int i = 0; i < _ens.nsys(); i++){
			integrate_system(_ens[i]);
		}
	}

        //! defines inner product of two arrays
	inline static double inner_product(const double a[3],const double b[3]){
		return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
	}

        //! Calculate the force field. 
	void calcForces(ensemble::SystemRef& sys, double acc[][3],double jerk[][3]){
		const int nbod = sys.nbod();

		/// Clear acc and jerk
		for(int b = 0; b < nbod; b++)	for(int c =0; c < 3; c++) 
			acc[b][c] = 0, jerk[b][c] = 0;

		/// Loop through all pairs
		for(int i=0; i < nbod-1; i++) for(int j = i+1; j < nbod; j++) {

			double dx[3] = { sys[j][0].pos()-sys[i][0].pos(),
				sys[j][1].pos()-sys[i][1].pos(),
				sys[j][2].pos()-sys[i][2].pos()
			};
			double dv[3] = { sys[j][0].vel()-sys[i][0].vel(),
				sys[j][1].vel()-sys[i][1].vel(),
				sys[j][2].vel()-sys[i][2].vel()
			};

			/// Calculated the magnitude
			double r2 = dx[0]*dx[0] + dx[1]*dx[1] + dx[2] * dx[2];
			double rinv = 1 / ( sqrt(r2) * r2 ) ;
			double rv =  inner_product(dx,dv) * 3. / r2;

			/// Update acc/jerk for i
			const double scalar_i = +rinv*sys[j].mass();
			for(int c = 0; c < 3; c++) {
				acc[i][c] += dx[c]* scalar_i;
				jerk[i][c] += (dv[c] - dx[c] * rv) * scalar_i;
			}

			/// Update acc/jerk for j
			const double scalar_j = -rinv*sys[i].mass();
			for(int c = 0; c < 3; c++) {
				acc[j][c] += dx[c]* scalar_j;
				jerk[j][c] += (dv[c] - dx[c] * rv) * scalar_j;
			}
		}
	}

        //! Integrate ensembles
	void integrate_system(ensemble::SystemRef sys){
		const int nbod = sys.nbod();
		double pre_pos[nbod][3];
		double pre_vel[nbod][3];
		double acc0[nbod][3];
		double acc1[nbod][3];
		double jerk0[nbod][3];
		double jerk1[nbod][3];

		calcForces(sys,acc0,jerk0);

		monitor_t montest (_mon_params,sys,*_log);


		for(int iter = 0 ; (iter < _max_iterations) && sys.is_active() ; iter ++ ) {
			double h = _time_step;

			if( sys.time() + h > _destination_time ) {
				h = _destination_time - sys.time();
			}

			/// Predict
			for(int b = 0; b < nbod; b++)	for(int c =0; c < 3; c++) {
					sys[b][c].pos() += h * (sys[b][c].vel()+h*0.5*(acc0[b][c]+h/3*jerk0[b][c]));
					sys[b][c].vel() += h * (acc0[b][c]+h*0.5*jerk0[b][c]);
				}

			/// Copy positions
			for(int b = 0; b < nbod; b++) for(int c =0; c < 3; c++)
					pre_pos[b][c] = sys[b][c].pos(), pre_vel[b][c] = sys[b][c].vel();

			///Integrate, Round one
			{
				calcForces(sys,acc1,jerk1);

				// Correct
				for(int b = 0; b < nbod; b++)	for(int c =0; c < 3; c++) {
					sys[b][c].pos() = pre_pos[b][c] 
						+ (.1-.25) * (acc0[b][c] - acc1[b][c]) * h * h 
						- 1/60.0 * ( 7 * jerk0[b][c] + 2 * jerk1[b][c] ) * h * h * h;

					sys[b][c].vel() = pre_vel[b][c] 
						+ ( -.5 ) * (acc0[b][c] - acc1[b][c] ) * h 
						-  1/12.0 * ( 5 * jerk0[b][c] + jerk1[b][c] ) * h * h;
				}
			}

			/// Integrate, Round two
			{
				calcForces(sys,acc1,jerk1);

				// Correct
				for(int b = 0; b < nbod; b++)	for(int c =0; c < 3; c++) {
					sys[b][c].pos() = pre_pos[b][c] 
						+ (.1-.25) * (acc0[b][c] - acc1[b][c]) * h * h 
						- 1/60.0 * ( 7 * jerk0[b][c] + 2 * jerk1[b][c] ) * h * h * h;

					sys[b][c].vel() = pre_vel[b][c] 
						+ ( -.5 ) * (acc0[b][c] - acc1[b][c] ) * h 
						-  1/12.0 * ( 5 * jerk0[b][c] + jerk1[b][c] ) * h * h;
				}
			}

			for(int b = 0; b < nbod; b++)	for(int c =0; c < 3; c++) 
				acc0[b][c] = acc1[b][c], jerk0[b][c] = jerk1[b][c];
			
			sys.time() += h;
//                         if (sys.time() >= 1)
//                         {
//                           printf("System at time:%.12f, iter = %d\n", sys.time(), iter);
//                           for(int b = 0; b < nbod; b++)
//                             printf("  Body %d Pos:%.12f %.12f %.12f Vel:%.12f %.12f %.12f\n", b, sys[b][0].pos(), sys[b][1].pos(), sys[b][2].pos(), sys[b][0].vel(),sys[b][1].vel(),sys[b][2].vel());
//                             
//                         }

			if( sys.is_active() )  {
				montest(0);
				if( sys.time() > _destination_time - 1e-12 ) 
					sys.set_inactive();
			}

		}
	}
};



} } // Close namespaces
