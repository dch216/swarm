/*************************************************************************
 * Copyright (C) 2010 by Aaron Boley  and the Swarm-NG Development Team  *
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

/*! \file hermite_adap_gpu.cpp
 * \brief Constructor & factory for gpu_hermite_adap_integrator
 * 
*/
#include <cassert>
#include "swarm.h"
#include "hermite_adap_gpu.h"
#include <cuda.h>
#include <cuda_runtime.h>

namespace swarm {

namespace hermite_adap_gpu {

/**
 * Constructor for hermite gpu integrator
 *
 * @tparam real_hi double
 * @tparam real_lo float for single and mixed, double for double
 * @param[in] cfg configuration file needs a timestep, precision, stepfac,  and block size.
 */
template<typename real_hi, typename real_lo>
gpu_hermite_adap_integrator<real_hi,real_lo>::gpu_hermite_adap_integrator(const config &cfg)
{
	if(!cfg.count("min time step")) ERROR("Integrator gpu_hermite_adap needs a timestep ('min time step' keyword in the config file).");
	h = atof(cfg.at("min time step").c_str());

	if(!cfg.count("time step factor")) ERROR("Integrator gpu_hermite_adap needs a time step factor ('time step factor' keyword in the config file).");
	stepfac = atof(cfg.at("time step factor").c_str());

	if(!cfg.count("precision")) ERROR("Integrator gpu_hermite_adap needs precision ('precision' keyword in the config file).");
	prec = atoi(cfg.at("precision").c_str());

	threadsPerBlock = cfg.count("threads per block") ? atoi(cfg.at("threads per block").c_str()) : 64;
}

/**
 * Create double/single/mixed hermite gpu integrator based on precision
 *
 * @param[in] cfg configuration file needs precision for hermite integrator    
 */
extern "C" integrator *create_gpu_hermite_adap(const config &cfg)
{
	if(!cfg.count("precision")) ERROR("Integrator gpu_hermite_adap needs precision ('precision' keyword in the config file).");
	int prec = atoi(cfg.at("precision").c_str());
	assert((prec>=1)&&(prec<=3));
	//single precision
	if (prec==2)
	  return new gpu_hermite_adap_integrator<float,float>(cfg);
	//mixed precision
	else if (prec==3)
	  return new gpu_hermite_adap_integrator<double,float>(cfg);
	//double precision
	else 
	  return new gpu_hermite_adap_integrator<double,double>(cfg);
}

} // end namespace hermite_adap_gpu
} // end namespace swarm
