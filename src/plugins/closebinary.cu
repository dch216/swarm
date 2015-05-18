/*! \file closebinary.cu
 *   \brief Initializes the GPU version of the close binary  propagator plugins.
 *
 */

#include "propagators/closebinary.hpp"
#include "monitors/composites.hpp"
#include "monitors/stop_on_ejection.hpp"
#include "monitors/log_time_interval.hpp"
#include "swarm/gpu/gravitation_acc.hpp"

//! Declare device_log variable 
typedef gpulog::device_log L;
using namespace swarm::monitors;
using namespace swarm::gpu::bppt;
using swarm::integrator_plugin_initializer;

//! Initialize the integrator plugin for close binary propagator
integrator_plugin_initializer< generic< CloseBinaryPropagator, stop_on_ejection<L>, GravitationAcc > >
	closebinary_prop_plugin("closebinary"
			,"This is the integrator based on the close binary propagator");



