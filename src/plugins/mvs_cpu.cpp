/*************************************************************************
 * Copyright (C) 2011 by Eric Ford and the Swarm-NG Development Team  *
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
#include "integrators/mvs_cpu.hpp"
#include "monitors/log_time_interval.hpp"
#include "monitors/stop_on_ejection.hpp"
#include "monitors/composites.hpp"

typedef gpulog::host_log L;
using namespace swarm::monitors;
using namespace swarm::cpu;
using swarm::integrator_plugin_initializer;


integrator_plugin_initializer<
  mvs_cpu< stop_on_ejection<L> >
	> mvs_cpu_plugin("mvs_cpu");



/*integrator_plugin_initializer<
		mvs_cpu< combine< L, stop_on_ejection<L>, stop_on_close_encounter<L> > >
	> mvs_cpu_plugin_crossing_orbit("mvs_cpu_crossing");*/

integrator_plugin_initializer<
  mvs_cpu< stop_on_ejection_or_close_encounter<L> >
	> mvs_cpu_plugin_ejection_or_close_encounter(
		"mvs_cpu_ejection_or_close_encounter"
	);


integrator_plugin_initializer<
  mvs_cpu< log_time_interval<L> >
	> mvs_cpu_log_plugin("mvs_cpu_log");


