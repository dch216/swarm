#include "hermite_gpu_bpt.h"
#include "meta.hpp"
#include "storage.hpp"

namespace swarm {
namespace hermite_gpu_bpt {

		
	__constant__ ensemble gpu_hermite_ens;

	inline __device__ static double inner_product(const double a[3],const double b[3]){
		return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
	}


	/**!
	 * helper function for accjerk_updater that operates on each component
	 * it gets scalar part of acceleration as input and calculates one component of
	 * acceleration and jerk at a time
	 *
	 */
	template<int nbod>
	__device__ static void accjerk_updater_component(int c,int i
			,double dx[3],double dv[3],double scalar,double rv
			,double (&acc)[3][nbod],double (&jerk)[3][nbod]){
		acc[c][i] += dx[c]* scalar;
		jerk[c][i] += (dv[c] - dx[c] * rv) * scalar;

	}

	template<int nbod>
		struct systemdata {
			double pos[3][nbod];
			double vel[3][nbod];
			double acc[3][nbod];
			double jerk[3][nbod];
		} ;

	/*! 
	 * templatized function object to calculate acceleration and jerk
	 * It updates accleration and jerk for one body: bodid. this function
	 * object is body of a n*n loop. so it should get called for every pair
	 *
	 */
	template<int nbod>
	struct accjerk_updater {
		ensemble::systemref& sysref;
		systemdata<nbod> &system;
		const int i;
		__device__ accjerk_updater(const int bodid,ensemble::systemref& sysref,systemdata<nbod> &system)
			:sysref(sysref),system(system),i(bodid){
				system.acc[0][i] = system.acc[1][i] = system.acc[2][i] = 0.0;
				system.jerk[0][i] = system.jerk[1][i] = system.jerk[2][i] = 0.0;
			}
		__device__ void operator()(int j)const{
			if(i != j){

				double dx[3] =  { system.pos[0][j]-system.pos[0][i],system.pos[1][j]-system.pos[1][i],system.pos[2][j]-system.pos[2][i]};
				double dv[3] =  { system.vel[0][j]-system.vel[0][i],system.vel[1][j]-system.vel[1][i],system.vel[2][j]-system.vel[2][i]};

				// computing scalar part of the acceleration
				double r2 =  dx[0]*dx[0] + dx[1]*dx[1] + dx[2]*dx[2] ;
				double rv =  inner_product(dx,dv) * 3. / r2;
				double rinv = rsqrt(r2)  / r2;

				// vectorized part
				const double scalar_i = +rinv*sysref[j].mass();
				for(int c = 0; c < 3 ; c++)
					accjerk_updater_component<nbod>(c,i,dx,dv,scalar_i,rv,system.acc,system.jerk);

			}
		}
		__device__ void compute(){
			//for(int j = 0; j < nbod; j++) (*this)(j);
			Unroller<0,nbod>::step(*this);
		}
	};


	template<int nbod>
		__device__ static void predictor(int i,int c,systemdata<nbod> &s,systemdata<nbod> &d,double h){
			d.pos[c][i] = s.pos[c][i] +  h*(s.vel[c][i]+(h*0.5)*(s.acc[c][i]+(h/3.)*s.jerk[c][i]));
			d.vel[c][i] = s.vel[c][i] +  h*(s.acc[c][i]+(h*0.5)*s.jerk[c][i]);
		}

	template<int nbod>
	__device__ static void corrector(int i,const int& c,systemdata<nbod> &s,systemdata<nbod> &d,const double& h){
		d.pos[c][i] = s.pos[c][i] + (h*0.5) * ( (s.vel[c][i]+d.vel[c][i]) 
				+ (h*7.0/30.)*( (s.acc[c][i]-d.acc[c][i]) + (h/7.) * (s.jerk[c][i]+d.jerk[c][i])));
		d.vel[c][i] = s.vel[c][i] + (h*0.5) * ( (s.acc[c][i]+d.acc[c][i]) + (h/6.) * (s.jerk[c][i]-d.jerk[c][i]));
	}

	__device__ void copy(double src[3],double des[3]){
		des[0] = src[0], des[1] = src[1] , des[2] = src[2];
	}

	template<int nbod>
	__device__ void load_to_shared(systemdata<nbod> &system,ensemble::systemref& sysref,int c,int i){
		system.pos[c][i] = sysref[i].p(c), system.vel[c][i] = sysref[i].v(c);
	}

	template<int nbod>
	__device__ void store_from_shared(systemdata<nbod> &system,ensemble::systemref& sysref,int c,int i){
		sysref[i].p(c) = system.pos[c][i] ,  sysref[i].v(c) = system.vel[c][i];
	}


	template<int nbod>
	__global__ void gpu_hermite_bpt_integrator_kernel(double destination_time, double time_step){
		
		// this kernel will process specific body of specific system
		// getting essential pointers
		int sysid = ((blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x) * blockDim.y + threadIdx.y;
		int sysid_in_block = threadIdx.y;
		int bodid = threadIdx.x;
		ensemble &ens = gpu_hermite_ens;
		if(sysid >= ens.nsys() || bodid >= nbod) { return; }
		ensemble::systemref gsys ( ens[sysid] );



			
		// TODO: you can put LocalMemory instead of SharedMemory here to tell the class to store
		// data in local memory.
		typedef storage< systemdata<nbod>[2], SharedMemory > systemst_t;

		// shared memory allocation
		extern __shared__ char shared_mem[];
		char local_memory_pointer[systemst_t::local_memory_usage];
		char * shared_memory_pointer = shared_mem + sysid_in_block * systemst_t::shared_memory_usage;

		// Storage class for System. it will act as a smart pointer class
		systemst_t systemst(local_memory_pointer,shared_memory_pointer);

		systemdata<nbod> (&system)[2] = *systemst;

//		double mass[nbod];
		// local memory allocation

		double t_start = gsys.time(), t = t_start;
		double t_end = min(t_start + destination_time,gsys.time_end());

		// Load data into shared memory (cooperative load)
		load_to_shared<nbod>(system[0],gsys,0,bodid);
		load_to_shared<nbod>(system[0],gsys,1,bodid);
		load_to_shared<nbod>(system[0],gsys,2,bodid);
//		#pragma unroll
//		for(int i = 0; i < nbod; i++) mass[i] = gsys[i].mass();

		
		__syncthreads(); // load should complete before calculating acceleration and jerk

		// Calculate acceleration and jerk
		accjerk_updater<nbod> accjerk_updater_instance(bodid,gsys,system[0]);
		accjerk_updater_instance.compute();

		while(t < t_end){
			for(int k = 0; k < 2; k++)
			{
				double h = min(time_step, t_end - t);
				// these two variable determine how each half of pos/vel/acc/jerk arrays
				// are going to be used to avoid unnecessary copying.
				const int s = k, d = 1-k; 
				// Predict 
#pragma unroll
				for(int c = 0; c < 3; c++)
					predictor<nbod>(bodid,c,system[s],system[d],h);


				// Do evaluation and correction two times (PEC2)
				for(int l = 0; l < 2; l++)
				{
					__syncthreads();
					// Calculate acceleration and jerk
					accjerk_updater<nbod> accjerk_updater_instance(bodid,gsys,system[d]);
					accjerk_updater_instance.compute();

					//__syncthreads(); // to prevent WAR. corrector updates pos/vel that accjerk_updater would read

					// Correct
#pragma unroll
					for(int c = 0; c < 3; c++)
						corrector<nbod>(bodid,c,system[s],system[d],h);

				}
				t += h;
			}

			/*debug_hook();

			if(bodid == 0) 
				gsys.increase_stepcount();
*/
			if(log::needs_output(ens, t, sysid))
			{
				// Save pos/vel to global memory
				store_from_shared<nbod>(system[0],gsys,0,bodid);
				store_from_shared<nbod>(system[0],gsys,1,bodid);
				store_from_shared<nbod>(system[0],gsys,2,bodid);
				if(bodid == 0) {
					gsys.set_time(t);
					log::output_system(dlog, ens, t, sysid);
				}
			}

		}

		if(bodid == 0) 
			gsys.set_time(t);
		// Save pos/vel to global memory
		store_from_shared<nbod>(system[0],gsys,0,bodid);
		store_from_shared<nbod>(system[0],gsys,1,bodid);
		store_from_shared<nbod>(system[0],gsys,2,bodid);

	}

	//! Simple template Function Object to execute appropriate kernel at runtime
	template<int nbod>
		struct kernel_launcher {
			template<class P>
				static void choose(P p){
					double dT = p.dT;
					double h = p.h;
					gpu_hermite_bpt_integrator_kernel<nbod><<<p.gridDim, p.threadDim, p.shared_memory_size>>>(dT, h);
				}
		};



	inline int min_power_2(const int& x){
		int y;
		for(y = 1; y < x; y*=2);
		return y;
	}
	/*!
	 * \brief host function to invoke a kernel (double precision) 
	 *
	 * Currently maximum number of bodies is set to 10.
	 * In order to change, add if statement. 
	 * @param[in,out] ens gpu_ensemble for data communication
	 * @param[in] dT destination time 
	 */
		void gpu_hermite_bpt_integrator::integrate(gpu_ensemble &ens, double dT)
		{
			/* Upload the kernel parameters */ 
			if(ens.last_integrator() != this) 
			{ 
				int system_per_block = threadsPerBlock / ens.nbod();
				threadDim.x = ens.nbod();
				threadDim.y = system_per_block;
				
				// 2: new and old copies
				// 4: {pos,vel,acc,jerk}
				// 3: x,y,z
				this->shared_memory_size = system_per_block * ens.nbod() * 2 * 4 * 3 * sizeof(double);

				ens.set_last_integrator(this); 
				configure_grid(gridDim,  system_per_block , ens.nsys()); 
				cudaMemcpyToSymbol(gpu_hermite_ens,	&ens, sizeof(gpu_hermite_ens) ); 
				if(dT == 0.) { return; } 
			} 
			// flush CPU/GPU output logs
			log::flush(log::memory | log::if_full);

			this->dT = dT;
			const int MAX_NBODIES = 10;
			if(ens.nbod() <= MAX_NBODIES){
				choose<kernel_launcher,3,MAX_NBODIES,void>(ens.nbod(),*this);
			} else {
				// How do we get an error message out of here?
				ERROR("Invalid number of bodies. (only up to 10 bodies per system)");
				return;
			}
			// flush CPU/GPU output logs
			log::flush(log::memory);

		}

}
}
