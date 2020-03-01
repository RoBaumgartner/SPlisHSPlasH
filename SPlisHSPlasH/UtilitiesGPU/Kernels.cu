#include "Kernels.cuh"
#include "../Simulation.h"

using namespace SPH;

#define TEX_VERSION_A

#ifdef PRECOMPUTED_KERNEL_AS_TEXTURE
#ifdef TEX_VERSION_A
#else
texture<Real, cudaTextureType1D, cudaReadModeElementType> precomputedKernelTexW;
texture<Real, cudaTextureType1D, cudaReadModeElementType> precomputedKernelTexGradW;
#endif
cudaArray * d_precomputedKernelArrayW = nullptr;
cudaArray * d_precomputedKernelArrayGradW = nullptr;
#endif

//////////////////////////////////////////////////////////////////
// Helper host methods 
//////////////////////////////////////////////////////////////////

KernelData::KernelData()
{
	CudaHelper::CudaMalloc(&d_W, PRECOMPUTED_KERNEL_SIZE);
	CudaHelper::CudaMalloc(&d_gradW, PRECOMPUTED_KERNEL_SIZE + 1);
}

KernelData::~KernelData()
{
	CudaHelper::CudaFree(d_W);
	CudaHelper::CudaFree(d_gradW);
}

void updateKernelData(KernelData &data)
{
	data.radius = PrecomputedKernel<CubicKernel, PRECOMPUTED_KERNEL_SIZE>::getRadius();
	data.invStepSize = PrecomputedKernel<CubicKernel, PRECOMPUTED_KERNEL_SIZE>::getInvStepSize();
	data.radius2 = data.radius * data.radius;
	CudaHelper::MemcpyHostToDevice(PrecomputedKernel<CubicKernel, PRECOMPUTED_KERNEL_SIZE>::getWeightField(), data.d_W, PRECOMPUTED_KERNEL_SIZE);
	CudaHelper::MemcpyHostToDevice(PrecomputedKernel<CubicKernel, PRECOMPUTED_KERNEL_SIZE>::getGradField(), data.d_gradW, PRECOMPUTED_KERNEL_SIZE + 1);

#ifdef PRECOMPUTED_KERNEL_AS_TEXTURE
	// kernel value texture
	{
#ifdef TEX_VERSION_A
		// create the array that stores the texture data
		cudaChannelFormatDesc channelDescW = cudaCreateChannelDesc(8 * sizeof(Real), 0, 0, 0, cudaChannelFormatKindFloat);
		cudaMallocArray(&d_precomputedKernelArrayW, &channelDescW, PRECOMPUTED_KERNEL_SIZE, 1);
		//CudaHelper::MemcpyHostToDevice(PrecomputedKernel<CubicKernel, PRECOMPUTED_KERNEL_SIZE>::getWeightField(), d_precomputedKernelArrayW, PRECOMPUTED_KERNEL_SIZE);
		cudaMemcpyToArray(d_precomputedKernelArrayW, 0, 0, PrecomputedKernel<CubicKernel, PRECOMPUTED_KERNEL_SIZE>::getWeightField(), sizeof(Real)*PRECOMPUTED_KERNEL_SIZE, cudaMemcpyHostToDevice);
		// create the texture resource that uses the array created above
		cudaResourceDesc texWRes;
		memset(&texWRes, 0, sizeof(cudaResourceDesc));
		texWRes.resType = cudaResourceTypeArray;
		texWRes.res.array.array = d_precomputedKernelArrayW;
		// the texture description sets the filter mode etc.
		cudaTextureDesc texWDescr;
		memset(&texWDescr, 0, sizeof(cudaTextureDesc));
		texWDescr.normalizedCoords = true;
		texWDescr.filterMode = cudaFilterModeLinear;
		texWDescr.addressMode[0] = cudaAddressModeClamp;
		texWDescr.readMode = cudaReadModeElementType;
		// create a texture object with thte resource and description from above
		cudaCreateTextureObject(&data.texW, &texWRes, &texWDescr, nullptr);

#else

		cudaMallocArray(&d_precomputedKernelArrayW, &precomputedKernelTexW.channelDesc, PRECOMPUTED_KERNEL_SIZE, 1);
		//CudaHelper::MemcpyHostToDevice(PrecomputedKernel<CubicKernel, PRECOMPUTED_KERNEL_SIZE>::getWeightField(), d_precomputedKernelArrayW, PRECOMPUTED_KERNEL_SIZE);
		cudaError_t cudaStatus = cudaMemcpyToArray(d_precomputedKernelArrayW, 0, 0, PrecomputedKernel<CubicKernel, PRECOMPUTED_KERNEL_SIZE>::getWeightField(), sizeof(Real)*PRECOMPUTED_KERNEL_SIZE, cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess)
		{
			throw CUDAMemCopyException("cudaMemcpy() from host to device failed!");
		}
		cudaBindTextureToArray(&precomputedKernelTexW, d_precomputedKernelArrayW, &precomputedKernelTexW.channelDesc);
		precomputedKernelTexW.normalized = false;
		precomputedKernelTexW.filterMode = cudaFilterModeLinear;
		precomputedKernelTexW.addressMode[0] = cudaAddressModeClamp;
#endif
	}

	// kernel gradient texture
	{
#ifdef TEX_VERSION_A
		// create the array that stores the texture data
		cudaChannelFormatDesc channelDescGradW = cudaCreateChannelDesc(8 * sizeof(Real), 0, 0, 0, cudaChannelFormatKindFloat);
		cudaMallocArray(&d_precomputedKernelArrayGradW, &channelDescGradW, PRECOMPUTED_KERNEL_SIZE + 1, 1);
		//CudaHelper::MemcpyHostToDevice(PrecomputedKernel<CubicKernel, PRECOMPUTED_KERNEL_SIZE>::getGradField(), d_precomputedKernelArrayGradW, PRECOMPUTED_KERNEL_SIZE + 1);
		cudaMemcpyToArray(d_precomputedKernelArrayGradW, 0, 0, PrecomputedKernel<CubicKernel, PRECOMPUTED_KERNEL_SIZE>::getGradField(), sizeof(Real)*(PRECOMPUTED_KERNEL_SIZE+1), cudaMemcpyHostToDevice);
		// create the texture resource that uses the array created above
		cudaResourceDesc texGradWRes;
		memset(&texGradWRes, 0, sizeof(cudaResourceDesc));
		texGradWRes.resType = cudaResourceTypeArray;
		texGradWRes.res.array.array = d_precomputedKernelArrayGradW;
		// the texture description sets the filter mode etc.
		cudaTextureDesc texGradWDescr;
		memset(&texGradWDescr, 0, sizeof(cudaTextureDesc));
		texGradWDescr.normalizedCoords = true;
		texGradWDescr.filterMode = cudaFilterModeLinear;
		texGradWDescr.addressMode[0] = cudaAddressModeClamp;
		texGradWDescr.readMode = cudaReadModeElementType;
		// create a texture object with thte resource and description from above
		cudaCreateTextureObject(&data.texGradW, &texGradWRes, &texGradWDescr, nullptr);
#else

		cudaMallocArray(&d_precomputedKernelArrayGradW, &precomputedKernelTexGradW.channelDesc, PRECOMPUTED_KERNEL_SIZE + 1, 1);
		//CudaHelper::MemcpyHostToDevice(PrecomputedKernel<CubicKernel, PRECOMPUTED_KERNEL_SIZE>::getWeightField(), d_precomputedKernelArrayGradW, PRECOMPUTED_KERNEL_SIZE);
		cudaError_t cudaStatus = cudaMemcpyToArray(d_precomputedKernelArrayGradW, 0, 0, PrecomputedKernel<CubicKernel, PRECOMPUTED_KERNEL_SIZE>::getGradField(), sizeof(Real)*(PRECOMPUTED_KERNEL_SIZE + 1), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess)
		{
			throw CUDAMemCopyException("cudaMemcpy() from host to device failed!");
		}
		cudaBindTextureToArray(&precomputedKernelTexGradW, d_precomputedKernelArrayGradW, &precomputedKernelTexGradW.channelDesc);
		precomputedKernelTexGradW.normalized = false;
		precomputedKernelTexGradW.filterMode = cudaFilterModeLinear;
		precomputedKernelTexGradW.addressMode[0] = cudaAddressModeClamp;
#endif
	}

#endif
}

//////////////////////////////////////////////////////////////////
//Kernels for all methods 
//////////////////////////////////////////////////////////////////

#ifdef PRECOMPUTED_KERNEL_AS_TEXTURE

__device__
Real kernelWeightPrecomputed(const Vector3r &r, const Real radius, cudaTextureObject_t tex)
{
	const float u = r.norm() / radius;
	return tex1D<Real>(tex, u); // TODO: https://devtalk.nvidia.com/default/topic/465851/cuda-programming-and-performance/cuda-texture/post/3310670/#3310670
}

__device__
Vector3r gradKernelWeightPrecomputed(const Vector3r &r, const Real radius, cudaTextureObject_t tex)
{
	const float u = r.norm() / radius;
	return tex1D<Real>(tex, u) * r; // TODO: https://devtalk.nvidia.com/default/topic/465851/cuda-programming-and-performance/cuda-texture/post/3310670/#3310670
}

__device__
Real kernelWeightPrecomputed(const Vector3r &r, const KernelData* const data)
{
	return kernelWeightPrecomputed(r, data->radius, data->texW);
}

__device__
Vector3r gradKernelWeightPrecomputed(const Vector3r &r, const KernelData* const data)
{
	return gradKernelWeightPrecomputed(r, data->radius, data->texGradW);
}

#else

__device__
Real kernelWeightPrecomputed(const Vector3r &r, const KernelData* const data)
{
	Real res = 0.0;
	const Real r2 = r.squaredNorm();
	if (r2 <= data->radius2)
	{
		const Real rl = sqrt(r2);
		//const unsigned int pos = std::min<unsigned int>((unsigned int)(rl * data->invStepSize), PRECOMPUTED_KERNEL_SIZE-2u);
		unsigned int pos = 0;
		if(static_cast<unsigned int>(rl * data->invStepSize) < PRECOMPUTED_KERNEL_SIZE-2u)
			pos = static_cast<unsigned int>(rl * data->invStepSize);
		else
			pos = PRECOMPUTED_KERNEL_SIZE-2u;
		res = static_cast<Real>(0.5)*(data->d_W[pos] + data->d_W[pos+1]);
	}
	return res;
}

__device__
Vector3r gradKernelWeightPrecomputed(const Vector3r &r, const KernelData* const data)
{
	Vector3r res;
	const Real rl = r.norm(); // rl / radius = > 0 - 1, texturSpeicher
	if (rl <= data->radius)
	{
		//const Real rl = sqrt(r2);
		//const unsigned int pos = static_cast<unsigned int>(fminf(static_cast<unsigned int>(rl * data->invStepSize), PRECOMPUTED_KERNEL_SIZE-1u));
		unsigned int pos = 0;
		if(static_cast<unsigned int>(rl * data->invStepSize) < PRECOMPUTED_KERNEL_SIZE-1u)
			pos = static_cast<unsigned int>(rl * data->invStepSize);
		else
			pos = PRECOMPUTED_KERNEL_SIZE-1u;
		res = 0.5*(data->d_gradW[pos] + data->d_gradW[pos + 1]) * r; // ersetzbar
	}
	else
		res.setZero();

	return res;
}

#endif

__device__
Real kernelWeight(const Vector3r& rin, const Real m_radius)
{
	const Real r = sqrt(rin[0] * rin[0] + rin[1] * rin[1] + rin[2] * rin[2]);
	const Real pi = 3.14159265358979323846;

	const Real h3 = m_radius*m_radius*m_radius;
	Real m_k = static_cast<Real>(8.0) / (pi*h3);
	Real m_l = static_cast<Real>(48.0) / (pi*h3);

	Real res = 0.0;
	const Real q = r / m_radius;

	if (q <= 1.0)
	{
		if (q <= 0.5)
		{
			const Real q2 = q*q;
			const Real q3 = q2*q;
			res = m_k * (static_cast<Real>(6.0)*q3 - static_cast<Real>(6.0)*q2 + static_cast<Real>(1.0));
		}
		else
		{
			res = m_k * (static_cast<Real>(2.0)*pow(static_cast<Real>(1.0) - q, 3));
		}
	}
	return res;
}

__device__
Vector3r gradKernelWeight(const Vector3r &rin, const Real m_radius)
{

	const Real pi = 3.14159265358979323846;
	const Real h3 = m_radius*m_radius*m_radius;
	const Real m_l = static_cast<Real>(48.0) / (pi*h3);

	Vector3r res;
	const Real rl = sqrt(rin[0] * rin[0] + rin[1] * rin[1] + rin[2] * rin[2]);
	const Real q = rl / m_radius;
	if ((rl > 1.0e-6) && (q <= 1.0))
	{
		const Vector3r gradq = rin * (static_cast<Real>(1.0) / (rl*m_radius));
		if (q <= 0.5)
		{
			res = m_l*q*((Real) 3.0*q - static_cast<Real>(2.0))*gradq;
		}
		else
		{
			const Real factor = static_cast<Real>(1.0) - q;
			res = m_l*(-factor*factor)*gradq;
		}
	}
	else
		res.setZero();

	return res;
}

// TODO: Marcel
__device__
void addForce(const Vector3r &pos, const Vector3r &f, /* output */ Vector3r* const forcesPerThread, /* output */ Vector3r* const torquesPerThread, 
	const Vector3r* const rigidBodyPositions, const uint* const forcesPerThreadIndices, const uint* const torquesPerThreadIndices, const uint index, const int id)
{
	#ifdef _OPENMP
	int tid = id;
	#else
	int tid = 0;
	#endif
	forcesPerThread[forcesPerThreadIndices[index] + tid] += f;
	torquesPerThread[torquesPerThreadIndices[index] + tid] += (pos - rigidBodyPositions[index]).cross(f);
}


__global__
void computeDensitiesGPU(/*out*/ Real* const densities, const Real* const volumes, const Real* const boundaryVolumes, const uint* const boundaryVolumeIndices, 
	const uint* const fmIndices, const Real* const densities0, const Real W_zero, const KernelData* const kernelData, 
	/*start of forall-parameters*/ const Real3* const* __restrict__ particles, const uint* const* __restrict__ neighbors, const uint*  const* __restrict__ neighborCounts, const uint*  const* __restrict__ neighborOffsets, 
  uint* neighborPointsetIndices, const uint nFluids, const uint nPointSets, const uint fluidModelIndex, const uint numParticles)
{
 	// Boundary: Akinci2012
	const uint i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i >= numParticles)
		return;

	extern __shared__ Real densities_tmp[];
	Real &density = densities_tmp[threadIdx.x];

	density = volumes[fluidModelIndex] * W_zero;
	const Real3 xi = particles[fluidModelIndex][i];

	//////////////////////////////////////////////////////////////////////////
	// Fluid
	//////////////////////////////////////////////////////////////////////////
	forall_fluid_neighborsGPU(
		density += volumes[pid] * kernelWeightPrecomputed(Vector3r(xi.x - xj.x, xi.y - xj.y, xi.z - xj.z), kernelData);
	)

	//////////////////////////////////////////////////////////////////////////
	// Boundary
	//////////////////////////////////////////////////////////////////////////
  forall_boundary_neighborsGPU(
		density += boundaryVolumes[boundaryVolumeIndices[pid - nFluids] + neighborIndex] *  kernelWeightPrecomputed(Vector3r(xi.x - xj.x, xi.y - xj.y, xi.z - xj.z), kernelData);
	)

	density *= densities0[fluidModelIndex];

	densities[fmIndices[fluidModelIndex] + i] = densities_tmp[threadIdx.x];
}


//////////////////////////////////////////////////////////////////
//Kernels for the WCPSH method 
//////////////////////////////////////////////////////////////////

__global__
void clearAccelerationsGPU(Real* masses, Vector3r* accelerations, const Vector3r grav, const uint numActiveParticles)
{
 	int i = blockIdx.x*blockDim.x + threadIdx.x;

	if(i >= numActiveParticles)
		return;

	// Clear accelerations of dynamic particles
	if (masses[i] != 0.0)
	{
		Vector3r &a = accelerations[i];
		a = grav;
	}
}

__global__
void updatePressureGPU(Real* const densities, const uint* const fmIndices, Real* const pressures, const Real* const densities0, const Real m_stiffness, const Real m_exponent,
	const uint fluidModelIndex, const uint numParticles)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;

	if(i >= numParticles)
		return;
	
	Real &density = densities[fmIndices[fluidModelIndex] + i];
	density = max(density, densities0[fluidModelIndex]);
	pressures[fmIndices[fluidModelIndex] + i] = m_stiffness * (pow(density / densities0[fluidModelIndex], m_exponent) - static_cast<Real>(1.0));
}

__global__
void computePressureAccelsGPU( /* output */ Vector3r* const pressureAccels, /* output */ Vector3r* const forcesPerThread, /* output */ Vector3r* const torquesPerThread, const uint* const forcesPerThreadIndices, 
	const uint* const torquesPerThreadIndices, const Real* const densities, const Real* const densities0, const uint* const fmIndices, const Real* const pressures, const Real* const masses, 
	const Vector3r* const rigidBodyPositions, const Real* const volumes, const Real* const boundaryVolumes, const uint* const boundaryVolumeIndices, const bool* const isDynamic, const int tid, const KernelData* kernelData,
	/*start of forall-parameters*/ const Real3* const* __restrict__ particles, const uint* const* __restrict__ neighbors, const uint*  const* __restrict__ neighborCounts, const uint*  const* __restrict__ neighborOffsets, 
  uint* neighborPointsetIndices, const uint nFluids, const uint nPointSets, const uint fluidModelIndex, const uint numParticles)
{
   const uint i = blockIdx.x*blockDim.x + threadIdx.x;

	if(i >= numParticles)
		return;

	extern __shared__ Vector3r pressureAccels_tmp[];

	const Real3 &xi = particles[fluidModelIndex][i];

	const Real density_i = densities[fmIndices[fluidModelIndex] + i];

	pressureAccels_tmp[threadIdx.x] = Vector3r(0, 0, 0);
	Vector3r &ai = pressureAccels_tmp[threadIdx.x];

	const Real dpi = pressures[fmIndices[fluidModelIndex] + i] / (density_i*density_i);
	//////////////////////////////////////////////////////////////////////////
	// Fluid
	//////////////////////////////////////////////////////////////////////////
	forall_fluid_neighborsGPU(
		const Real density_j = densities[fmIndices[pid] + neighborIndex] * densities0[fluidModelIndex] / densities0[pid];
		const Real dpj = pressures[fmIndices[pid] + neighborIndex] / (density_j*density_j);
		ai -= densities0[fluidModelIndex] * volumes[pid] * (dpi + dpj) * gradKernelWeightPrecomputed(Vector3r(xi.x - xj.x, xi.y - xj.y, xi.z - xj.z), kernelData);
	)

	//////////////////////////////////////////////////////////////////////////
	// Boundary
	//////////////////////////////////////////////////////////////////////////
	const Real dpj = pressures[fmIndices[fluidModelIndex] + i] / (densities0[fluidModelIndex] * densities0[fluidModelIndex]);
	forall_boundary_neighborsGPU(
		const Vector3r a = densities0[fluidModelIndex] * boundaryVolumes[fmIndices[pid - nFluids] + neighborIndex] * (dpi + dpj) * gradKernelWeightPrecomputed(Vector3r(xi.x - xj.x, xi.y - xj.y, xi.z - xj.z), kernelData);
		ai -= a;
		if(isDynamic[pid - nFluids])
		{
			addForce(Vector3r(xj.x, xj.y, xj.z), masses[i] * a, forcesPerThread, torquesPerThread, rigidBodyPositions, forcesPerThreadIndices, torquesPerThreadIndices, pid - nFluids, tid);
		}
	)

	pressureAccels[i] = pressureAccels_tmp[threadIdx.x];
}

__global__ 
void updatePosPressureAccelPressureAccel(Vector3r* const positions, Vector3r* const velocities, Vector3r* const accelerations,
	const Vector3r* const pressureAccels, const Real h, const uint numParticles)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;

	if(i >= numParticles)
		return;
	
	accelerations[i] += pressureAccels[i];
	velocities[i] += accelerations[i] * h;
	positions[i] += velocities[i] * h;
	
}


//////////////////////////////////////////////////////////////////
//Kernels for the DFSPH method 
//////////////////////////////////////////////////////////////////

__global__
void multiplyRealWithConstant(/*out*/ Real* const input, const uint* const fmIndices, const Real f, const uint fluidModelIndex, const uint numParticles)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i >= numParticles)
		return;

	input[fmIndices[fluidModelIndex] + i] *= f;
}


__global__ 
void divergenceSolveUpdateFluidVelocities( /*out*/ Vector3r* const fmVelocities, /*out*/ Real* const kappaV, /* output */ Vector3r* const forcesPerThread, /* output */ Vector3r* const torquesPerThread, 
	const uint* const forcesPerThreadIndices, const uint* const torquesPerThreadIndices, const Vector3r* const rigidBodyPositions, const Real* const densitiesAdv, const Real* const factors, 
	const uint* const fmIndices, const Real* const masses, const Real* const fmVolumes, const Real* const boundaryVolumes, const uint* const boundaryVolumeIndices, 
	const Real* const densities0, const bool* const isDynamic, const int tid, const Real h, const Real invH, const KernelData* const kernelData, const Real eps,
	/*start of forall-parameters*/ const Real3* const* __restrict__ particles, const uint* const* __restrict__ neighbors, const uint*  const* __restrict__ neighborCounts, const uint*  const* __restrict__ neighborOffsets, 
	uint* neighborPointsetIndices, const uint nFluids, const uint nPointSets, const uint fluidModelIndex, const uint numParticles)
{
 	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i >= numParticles)
		return;

	extern __shared__ Real3 part[];
	part[threadIdx.x] = particles[fluidModelIndex][i];
	const Real3 &xi = part[threadIdx.x];

	const Real b_i = densitiesAdv[fmIndices[fluidModelIndex] + i];
	const Real ki = b_i * factors[fmIndices[fluidModelIndex] + i];
	kappaV[fmIndices[fluidModelIndex] + i] += ki;

	Vector3r v_i = fmVelocities[fmIndices[fluidModelIndex] + i];

	//////////////////////////////////////////////////////////////////////////
	// Fluid
	//////////////////////////////////////////////////////////////////////////
	forall_fluid_neighborsGPU(
		const Real b_j = densitiesAdv[fmIndices[pid] + neighborIndex];
		const Real kj = b_j * factors[fmIndices[pid] + neighborIndex];

		const Real kSum = ki + densities0[pid] / densities0[fluidModelIndex] * kj;
		if(fabsf(kSum) > eps)
		{
			const Vector3r grad_p_j = -fmVolumes[pid] * gradKernelWeightPrecomputed(Vector3r(xi.x - xj.x, xi.y - xj.y, xi.z - xj.z), kernelData);
			v_i -= h * kSum * grad_p_j; // ki, kj already contain inverse density
		}
	)

	//////////////////////////////////////////////////////////////////////////
	// Boundary
	//////////////////////////////////////////////////////////////////////////
	if(fabsf(ki) > eps)
	{
		forall_boundary_neighborsGPU(
			const Vector3r grad_p_j = -boundaryVolumes[boundaryVolumeIndices[pid - nFluids] + neighborIndex] * gradKernelWeightPrecomputed(Vector3r(xi.x - xj.x, xi.y - xj.y, xi.z - xj.z), kernelData);
			const Vector3r velChange = -h * (Real) 1.0 * ki * grad_p_j;	// kj already contains inverse density
			v_i += velChange;
			addForce(Vector3r(xj.x, xj.y, xj.z), -masses[fmIndices[fluidModelIndex] + i] * velChange * invH, forcesPerThread, torquesPerThread, rigidBodyPositions, forcesPerThreadIndices, torquesPerThreadIndices, pid - nFluids, tid);
		)
	}

	fmVelocities[fmIndices[fluidModelIndex] + i] = v_i;
} 

__global__ 
void divergenceSolveUpdateFluidVelocities( /*out*/ Vector3r* const fmVelocities, /* output */ Vector3r* const forcesPerThread, /* output */ Vector3r* const torquesPerThread, 
	const uint* const forcesPerThreadIndices, const uint* const torquesPerThreadIndices, const Vector3r* const rigidBodyPositions, const Real* const densitiesAdv, const Real* const factors, 
	const uint* const fmIndices, const Real* const masses, const Real* const fmVolumes, const Real* const boundaryVolumes, const uint* const boundaryVolumeIndices, 
	const Real* const densities0, const bool* const isDynamic, const int tid, const Real h, const Real invH, const KernelData* const kernelData, const Real eps,
	/*start of forall-parameters*/ const Real3* const* __restrict__ particles, const uint* const* __restrict__ neighbors, const uint*  const* __restrict__ neighborCounts, const uint*  const* __restrict__ neighborOffsets, 
	uint* neighborPointsetIndices, const uint nFluids, const uint nPointSets, const uint fluidModelIndex, const uint numParticles)
{
 	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i >= numParticles)
		return;

	extern __shared__ Real3 part[];
	part[threadIdx.x] = particles[fluidModelIndex][i];
	const Real3 &xi = part[threadIdx.x];

	const Real b_i = densitiesAdv[fmIndices[fluidModelIndex] + i];
	const Real ki = b_i * factors[fmIndices[fluidModelIndex] + i];

	Vector3r v_i = fmVelocities[fmIndices[fluidModelIndex] + i];

	//////////////////////////////////////////////////////////////////////////
	// Fluid
	//////////////////////////////////////////////////////////////////////////
	forall_fluid_neighborsGPU(
		const Real b_j = densitiesAdv[fmIndices[pid] + neighborIndex];
		const Real kj = b_j * factors[fmIndices[pid] + neighborIndex];

		const Real kSum = ki + densities0[pid] / densities0[fluidModelIndex] * kj;
		if(fabsf(kSum) > eps)
		{
			const Vector3r grad_p_j = -fmVolumes[pid] * gradKernelWeightPrecomputed(Vector3r(xi.x - xj.x, xi.y - xj.y, xi.z - xj.z), kernelData);
			v_i -= h * kSum * grad_p_j; // ki, kj already contain inverse density
		}
	)

	//////////////////////////////////////////////////////////////////////////
	// Boundary
	//////////////////////////////////////////////////////////////////////////
	if(fabsf(ki) > eps)
	{
		forall_boundary_neighborsGPU(
			const Vector3r grad_p_j = -boundaryVolumes[boundaryVolumeIndices[pid - nFluids] + neighborIndex] * gradKernelWeightPrecomputed(Vector3r(xi.x - xj.x, xi.y - xj.y, xi.z - xj.z), kernelData);
			const Vector3r velChange = -h * (Real) 1.0 * ki * grad_p_j;	// kj already contains inverse density
			v_i += velChange;
			addForce(Vector3r(xj.x, xj.y, xj.z), -masses[fmIndices[fluidModelIndex] + i] * velChange * invH, forcesPerThread, torquesPerThread, rigidBodyPositions, forcesPerThreadIndices, torquesPerThreadIndices, pid - nFluids, tid);
		)
	}

	fmVelocities[fmIndices[fluidModelIndex] + i] = v_i;
} 

__global__ 
void pressureSolveUpdateFluidVelocities( /*out*/ Vector3r* const fmVelocities, /*out*/ Real* const kappa, /* output */ Vector3r* const forcesPerThread, /* output */ Vector3r* const torquesPerThread, 
	const uint* const forcesPerThreadIndices, const uint* const torquesPerThreadIndices, const Vector3r* const rigidBodyPositions, const Real* const densitiesAdv, const Real* const factors, 
	const uint* const fmIndices, const Real* const masses, const Real* const fmVolumes, const Real* const boundaryVolumes, const uint* const boundaryVolumeIndices, 
	const Real* const densities0, const bool* const isDynamic, const int tid, const Real h, const Real invH, const KernelData* const kernelData, const Real eps,
	/*start of forall-parameters*/ const Real3* const* __restrict__ particles, const uint* const* __restrict__ neighbors, const uint*  const* __restrict__ neighborCounts, const uint*  const* __restrict__ neighborOffsets, 
  uint* neighborPointsetIndices, const uint nFluids, const uint nPointSets, const uint fluidModelIndex, const uint numParticles)
{
 	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i >= numParticles)
		return;

	extern __shared__ Real3 part[];
	part[threadIdx.x] = particles[fluidModelIndex][i];
	const Real3 &xi = part[threadIdx.x];

	const Real b_i = densitiesAdv[fmIndices[fluidModelIndex] + i] - static_cast<Real>(1.0);
	const Real ki = b_i * factors[fmIndices[fluidModelIndex] + i];

	kappa[fmIndices[fluidModelIndex] + i] += ki;

	Vector3r v_i = fmVelocities[fmIndices[fluidModelIndex] + i];

	//////////////////////////////////////////////////////////////////////////
	// Fluid
	//////////////////////////////////////////////////////////////////////////
	forall_fluid_neighborsGPU(
		const Real b_j = densitiesAdv[fmIndices[pid] + neighborIndex] - static_cast<Real>(1.0);
		const Real kj = b_j * factors[fmIndices[pid] + neighborIndex];

		const Real kSum = ki + densities0[pid] / densities0[fluidModelIndex] * kj;
		if(fabsf(kSum) > eps)
		{
			const Vector3r grad_p_j = -fmVolumes[pid] * gradKernelWeightPrecomputed(Vector3r(xi.x - xj.x, xi.y - xj.y, xi.z - xj.z), kernelData);
			v_i -= h * kSum * grad_p_j; // ki, kj already contain inverse density
		}
	)

	//////////////////////////////////////////////////////////////////////////
	// Boundary
	//////////////////////////////////////////////////////////////////////////
	if(fabsf(ki) > eps)
	{
		forall_boundary_neighborsGPU(
			const Vector3r grad_p_j = -boundaryVolumes[boundaryVolumeIndices[pid - nFluids] + neighborIndex] * gradKernelWeightPrecomputed(Vector3r(xi.x - xj.x, xi.y - xj.y, xi.z - xj.z), kernelData);
			const Vector3r velChange = -h * (Real) 1.0 * ki * grad_p_j;	// kj already contains inverse density
			v_i += velChange;
			addForce(Vector3r(xj.x, xj.y, xj.z), -masses[fmIndices[fluidModelIndex] + i] * velChange * invH, forcesPerThread, torquesPerThread, rigidBodyPositions, forcesPerThreadIndices, torquesPerThreadIndices, pid - nFluids, tid);
		)
	}

	fmVelocities[fmIndices[fluidModelIndex] + i] = v_i;
}

__global__ 
void pressureSolveUpdateFluidVelocities( /*out*/ Vector3r* const fmVelocities, /* output */ Vector3r* const forcesPerThread, /* output */ Vector3r* const torquesPerThread, 
	const uint* const forcesPerThreadIndices, const uint* const torquesPerThreadIndices, const Vector3r* const rigidBodyPositions, const Real* const densitiesAdv, const Real* const factors, 
	const uint* const fmIndices, const Real* const masses, const Real* const fmVolumes, const Real* const boundaryVolumes, const uint* const boundaryVolumeIndices, 
	const Real* const densities0, const bool* const isDynamic, const int tid, const Real h, const Real invH, const KernelData* const kernelData, const Real eps,
	/*start of forall-parameters*/ const Real3* const* __restrict__ particles, const uint* const* __restrict__ neighbors, const uint*  const* __restrict__ neighborCounts, const uint*  const* __restrict__ neighborOffsets, 
  uint* neighborPointsetIndices, const uint nFluids, const uint nPointSets, const uint fluidModelIndex, const uint numParticles)
{
 	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i >= numParticles)
		return;

	extern __shared__ Real3 part[];
	part[threadIdx.x] = particles[fluidModelIndex][i];
	const Real3 &xi = part[threadIdx.x];

	const Real b_i = densitiesAdv[fmIndices[fluidModelIndex] + i] - static_cast<Real>(1.0);
	const Real ki = b_i * factors[fmIndices[fluidModelIndex] + i];

	Vector3r v_i = fmVelocities[fmIndices[fluidModelIndex] + i];

	//////////////////////////////////////////////////////////////////////////
	// Fluid
	//////////////////////////////////////////////////////////////////////////
	forall_fluid_neighborsGPU(
		const Real b_j = densitiesAdv[fmIndices[pid] + neighborIndex] - static_cast<Real>(1.0);
		const Real kj = b_j * factors[fmIndices[pid] + neighborIndex];

		const Real kSum = ki + densities0[pid] / densities0[fluidModelIndex] * kj;
		if(fabsf(kSum) > eps)
		{
			const Vector3r grad_p_j = -fmVolumes[pid] * gradKernelWeightPrecomputed(Vector3r(xi.x - xj.x, xi.y - xj.y, xi.z - xj.z), kernelData);
			v_i -= h * kSum * grad_p_j; // ki, kj already contain inverse density
		}
	)

	//////////////////////////////////////////////////////////////////////////
	// Boundary
	//////////////////////////////////////////////////////////////////////////
	if(fabsf(ki) > eps)
	{
		forall_boundary_neighborsGPU(
			const Vector3r grad_p_j = -boundaryVolumes[boundaryVolumeIndices[pid - nFluids] + neighborIndex] * gradKernelWeightPrecomputed(Vector3r(xi.x - xj.x, xi.y - xj.y, xi.z - xj.z), kernelData);
			const Vector3r velChange = -h * (Real) 1.0 * ki * grad_p_j;	// kj already contains inverse density
			v_i += velChange;
			addForce(Vector3r(xj.x, xj.y, xj.z), -masses[fmIndices[fluidModelIndex] + i] * velChange * invH, forcesPerThread, torquesPerThread, rigidBodyPositions, forcesPerThreadIndices, torquesPerThreadIndices, pid - nFluids, tid);
		)
	}

	fmVelocities[fmIndices[fluidModelIndex] + i] = v_i;
} 


__global__
void computeFactorsAndDensities(/*out*/ Real* const densities, /* out */ Real* factors, const Real* const fmVolumes, const Real* const boundaryVolumes, const uint* const boundaryVolumeIndices, 
	const uint* const fmIndices, const Real* const densities0, const Real W_zero, const Real eps, const KernelData* const kernelData, 
	/*start of forall-parameters*/ const Real3* const* __restrict__ particles, const uint* const* __restrict__ neighbors, const uint*  const* __restrict__ neighborCounts, const uint*  const* __restrict__ neighborOffsets, 
	uint* neighborPointsetIndices, const uint nFluids, const uint nPointSets, const uint fluidModelIndex, const uint numParticles)
{
 	// Boundary: Akinci2012
	const uint i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i >= numParticles)
		return;

	extern __shared__ Real3 part[];
	part[threadIdx.x] = particles[fluidModelIndex][i];
	const Real3 &xi = part[threadIdx.x];

	Real density = fmVolumes[fluidModelIndex] * W_zero;
	Real factor = 0.0;

	Real sum_grad_p_k = 0.0;
	Vector3r grad_p_i;
	grad_p_i.setZero();

	//////////////////////////////////////////////////////////////////////////
	// Fluid
	//////////////////////////////////////////////////////////////////////////
	forall_fluid_neighborsGPU(
		density += fmVolumes[pid] * kernelWeightPrecomputed(Vector3r(xi.x - xj.x, xi.y - xj.y, xi.z - xj.z), kernelData);

		const Vector3r grad_p_j = -fmVolumes[fluidModelIndex] * gradKernelWeightPrecomputed(Vector3r(xi.x - xj.x, xi.y - xj.y, xi.z - xj.z), kernelData);
		sum_grad_p_k += grad_p_j.squaredNorm();
		grad_p_i -= grad_p_j;
	)

	//////////////////////////////////////////////////////////////////////////
	// Boundary
	//////////////////////////////////////////////////////////////////////////
  forall_boundary_neighborsGPU(
		density += boundaryVolumes[boundaryVolumeIndices[pid - nFluids] + neighborIndex] *  kernelWeightPrecomputed(Vector3r(xi.x - xj.x, xi.y - xj.y, xi.z - xj.z), kernelData);

		const Vector3r grad_p_j = -boundaryVolumes[boundaryVolumeIndices[pid - nFluids] + neighborIndex] * gradKernelWeightPrecomputed(Vector3r(xi.x - xj.x, xi.y - xj.y, xi.z - xj.z), kernelData);
		grad_p_i -= grad_p_j;
	)

	density *= densities0[fluidModelIndex];
	densities[fmIndices[fluidModelIndex] + i] = density;

	sum_grad_p_k += grad_p_i.squaredNorm();

	//////////////////////////////////////////////////////////////////////////
	// Compute pressure stiffness denominator
	//////////////////////////////////////////////////////////////////////////
	if (sum_grad_p_k > eps)
		factor = -static_cast<Real>(1.0) / (sum_grad_p_k);
	else
		factor = 0.0;

	factors[fmIndices[fluidModelIndex] + i] = factor;
}

__global__
void divergenceSolveWarmstartComplete( /*out*/ Vector3r* const fmVelocities, const Vector3r* const bmVelocities, /*out*/ Real* const densitiesAdv, /* output */ Vector3r* const forcesPerThread, /* output */ Vector3r* const torquesPerThread, 
	const uint* const forcesPerThreadIndices, const uint* const torquesPerThreadIndices, const Vector3r* const rigidBodyPositions, /* out */ Real* const kappaV,
	const uint* const fmIndices, const Real* const masses, const Real* const fmVolumes, const Real* const boundaryVolumes, const uint* const boundaryVolumeIndices, 
	const Real* const densities0, const bool* const isDynamic, const int tid, const Real h, const KernelData* const kernelData, const Real eps,
	/*start of forall-parameters*/ const Real3* const* __restrict__ particles, const uint* const* __restrict__ neighbors, const uint*  const* __restrict__ neighborCounts, const uint*  const* __restrict__ neighborOffsets, 
  uint* neighborPointsetIndices, const uint nFluids, const uint nPointSets, const uint fluidModelIndex, const uint numParticles)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i >= numParticles || numParticles == 0)
		return;

	extern __shared__ Real3 part[];
	part[threadIdx.x] = particles[fluidModelIndex][i];
	const Real3 &xi = part[threadIdx.x];

	const Real invH = static_cast<Real>(1.0) / h;

	Real ki = static_cast<Real>(0.5) * max( kappaV[fmIndices[fluidModelIndex] + i] * invH, -static_cast<Real>(0.5) * densities0[fluidModelIndex] * densities0[fluidModelIndex]);
	kappaV[fmIndices[fluidModelIndex] + i] = ki;

	Vector3r vi = fmVelocities[fmIndices[fluidModelIndex] + i];

	Real densityAdv = 0.0;
	unsigned int numNeighbors = 0;

	//////////////////////////////////////////////////////////////////////////
	// Fluid
	//////////////////////////////////////////////////////////////////////////
	forall_fluid_neighborsGPU(
		const Real kj = kappaV[fmIndices[pid] + neighborIndex];

		const Real kSum = (ki + densities0[pid] / densities0[fluidModelIndex] * kj);
		if (fabsf(kSum) > eps)
		{
			const Vector3r grad_p_j = -fmVolumes[pid] * gradKernelWeightPrecomputed(Vector3r(xi.x - xj.x, xi.y - xj.y, xi.z - xj.z), kernelData);
			vi -= h * kSum * grad_p_j;					// ki, kj already contain inverse density
		}

		const Vector3r &vj = fmVelocities[fmIndices[pid] + neighborIndex];
		densityAdv += fmVolumes[pid] * (vi - vj).dot(gradKernelWeightPrecomputed(Vector3r(xi.x - xj.x, xi.y - xj.y, xi.z - xj.z), kernelData));
	)

	//////////////////////////////////////////////////////////////////////////
	// Boundary
	//////////////////////////////////////////////////////////////////////////

	forall_boundary_neighborsGPU(
		const Vector3r &vj = bmVelocities[boundaryVolumeIndices[pid - nFluids] + neighborIndex];
		densityAdv += boundaryVolumes[boundaryVolumeIndices[pid - nFluids] + neighborIndex] * (vi - vj).dot(gradKernelWeightPrecomputed(Vector3r(xi.x - xj.x, xi.y - xj.y, xi.z - xj.z), kernelData));
		
		if (fabsf(ki) > eps)
		{
			const Vector3r grad_p_j = -boundaryVolumes[boundaryVolumeIndices[pid - nFluids] + neighborIndex] * gradKernelWeightPrecomputed(Vector3r(xi.x - xj.x, xi.y - xj.y, xi.z - xj.z), kernelData);
			const Vector3r velChange = -h * (Real) 1.0 * ki * grad_p_j;				// kj already contains inverse density
			vi += velChange;
			addForce(Vector3r(xj.x, xj.y, xj.z), -masses[fmIndices[fluidModelIndex] + i] * velChange * invH, forcesPerThread, torquesPerThread, rigidBodyPositions, forcesPerThreadIndices, torquesPerThreadIndices, pid - nFluids, tid);
		}
	)

	fmVelocities[fmIndices[fluidModelIndex] + i] = vi;

	// only correct positive divergence
	densityAdv = max(densityAdv, static_cast<Real>(0.0));

	for (unsigned int pid = 0; pid < nPointSets; pid++)
	{
		const uint neighborsetIndex = neighborPointsetIndices[fluidModelIndex] + pid;
		numNeighbors += neighborCounts[neighborsetIndex][i];
	}
	
	// in case of particle deficiency do not perform a divergence solve
	if (numNeighbors < 20)
		densityAdv = 0.0;

	densitiesAdv[fmIndices[fluidModelIndex] + i] = densityAdv;
}


__global__
void divergenceSolveKernel1(/*out*/ Real* const densitiesAdv, /* out */ Real* const factors, const Real f, const Vector3r* const fmVelocities, const Vector3r* const bmVelocities, const uint* const fmIndices, 
	const Real* const fmVolumes, const Real* const boundaryVolumes, const uint* const boundaryVolumeIndices, const KernelData* const kernelData,
	/*start of forall-parameters*/ const Real3* const* __restrict__ particles, const uint* const* __restrict__ neighbors, const uint*  const* __restrict__ neighborCounts, const uint*  const* __restrict__ neighborOffsets, 
  uint* neighborPointsetIndices, const uint nFluids, const uint nPointSets, const uint fluidModelIndex, const uint numParticles)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i >= numParticles)
		return;

	extern __shared__ Real3 part[];
	part[threadIdx.x] = particles[fluidModelIndex][i];
	const Real3 &xi = part[threadIdx.x];

	const Vector3r vi = fmVelocities[fmIndices[fluidModelIndex] + i];

	Real densityAdv = 0.0;
	unsigned int numNeighbors = 0;

	//////////////////////////////////////////////////////////////////////////
	// Fluid
	//////////////////////////////////////////////////////////////////////////
	forall_fluid_neighborsGPU(
		const Vector3r &vj = fmVelocities[fmIndices[pid] + neighborIndex];
		densityAdv += fmVolumes[pid] * (vi - vj).dot(gradKernelWeightPrecomputed(Vector3r(xi.x - xj.x, xi.y - xj.y, xi.z - xj.z), kernelData));
	)

	//////////////////////////////////////////////////////////////////////////
	// Boundary
	//////////////////////////////////////////////////////////////////////////
	forall_boundary_neighborsGPU(
		const Vector3r &vj = bmVelocities[boundaryVolumeIndices[pid - nFluids] + neighborIndex];
		densityAdv += boundaryVolumes[boundaryVolumeIndices[pid - nFluids] + neighborIndex] * (vi - vj).dot(gradKernelWeightPrecomputed(Vector3r(xi.x - xj.x, xi.y - xj.y, xi.z - xj.z), kernelData));
	)
	
	// only correct positive divergence
	densityAdv = max(densityAdv, static_cast<Real>(0.0));

	for (unsigned int pid = 0; pid < nPointSets; pid++)
	{
		const uint neighborsetIndex = neighborPointsetIndices[fluidModelIndex] + pid;
		numNeighbors += neighborCounts[neighborsetIndex][i];
	}

	// in case of particle deficiency do not perform a divergence solve
	if (numNeighbors < 20)
		densityAdv = 0.0;

	factors[fmIndices[fluidModelIndex] + i] *= f;
	densitiesAdv[fmIndices[fluidModelIndex] + i] = densityAdv;
}

__global__
void divergenceSolveKernel1(/*out*/ Real* const densitiesAdv, /* out */ Real* const kappaV,  /* out */ Real* const factors, const Real f, const Vector3r* const fmVelocities, const Vector3r* const bmVelocities, const uint* const fmIndices, 
	const Real* const fmVolumes, const Real* const boundaryVolumes, const uint* const boundaryVolumeIndices, const KernelData* const kernelData,
	/*start of forall-parameters*/ const Real3* const* __restrict__ particles, const uint* const* __restrict__ neighbors, const uint*  const* __restrict__ neighborCounts, const uint*  const* __restrict__ neighborOffsets, 
  uint* neighborPointsetIndices, const uint nFluids, const uint nPointSets, const uint fluidModelIndex, const uint numParticles)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i >= numParticles)
		return;

	extern __shared__ Real3 part[];
	part[threadIdx.x] = particles[fluidModelIndex][i];
	const Real3 &xi = part[threadIdx.x];

	kappaV[fmIndices[fluidModelIndex] + i] = 0.0;

	const Vector3r vi = fmVelocities[fmIndices[fluidModelIndex] + i];

	Real densityAdv = 0.0;
	unsigned int numNeighbors = 0;

	//////////////////////////////////////////////////////////////////////////
	// Fluid
	//////////////////////////////////////////////////////////////////////////
	forall_fluid_neighborsGPU(
		const Vector3r &vj = fmVelocities[fmIndices[pid] + neighborIndex];
		densityAdv += fmVolumes[pid] * (vi - vj).dot(gradKernelWeightPrecomputed(Vector3r(xi.x - xj.x, xi.y - xj.y, xi.z - xj.z), kernelData));
	)

	//////////////////////////////////////////////////////////////////////////
	// Boundary
	//////////////////////////////////////////////////////////////////////////
	forall_boundary_neighborsGPU(
		const Vector3r &vj = bmVelocities[boundaryVolumeIndices[pid - nFluids] + neighborIndex];
		densityAdv += boundaryVolumes[boundaryVolumeIndices[pid - nFluids] + neighborIndex] * (vi - vj).dot(gradKernelWeightPrecomputed(Vector3r(xi.x - xj.x, xi.y - xj.y, xi.z - xj.z), kernelData));
	)
	
	// only correct positive divergence
	densityAdv = max(densityAdv, static_cast<Real>(0.0));

	for (unsigned int pid = 0; pid < nPointSets; pid++)
	{
		const uint neighborsetIndex = neighborPointsetIndices[fluidModelIndex] + pid;
		numNeighbors += neighborCounts[neighborsetIndex][i];
	}

	// in case of particle deficiency do not perform a divergence solve
	if (numNeighbors < 20)
		densityAdv = 0.0;

	factors[fmIndices[fluidModelIndex] + i] *= f;
	densitiesAdv[fmIndices[fluidModelIndex] + i] = densityAdv;
}


__global__
void divergenceSolveMultiply(/*out*/ Real* const kappaV, /* out */ Real* const factors, const uint* const fmIndices, const Real h, const uint fluidModelIndex, const uint numParticles)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i >= numParticles)
		return;

	kappaV[fmIndices[fluidModelIndex] + i] *= h;
	factors[fmIndices[fluidModelIndex] + i] *= h;
}

__global__
void divergenceSolveKernel2(/*out*/ Real* const densitiesAdv, /* out */ Real* const density_errors, const Vector3r* const fmVelocities, const Vector3r* const bmVelocities, const uint* const fmIndices, 
	const Real* const fmVolumes, const Real* const boundaryVolumes, const uint* const boundaryVolumeIndices, const Real* const densities0, const KernelData* const kernelData,
	/*start of forall-parameters*/ const Real3* const* __restrict__ particles, const uint* const* __restrict__ neighbors, const uint*  const* __restrict__ neighborCounts, const uint*  const* __restrict__ neighborOffsets, 
  uint* neighborPointsetIndices, const uint nFluids, const uint nPointSets, const uint fluidModelIndex, const uint numParticles)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i >= numParticles)
		return;

	extern __shared__ Real3 part[];
	part[threadIdx.x] = particles[fluidModelIndex][i];
	const Real3 &xi = part[threadIdx.x];
	const Vector3r vi = fmVelocities[fmIndices[fluidModelIndex] + i];

	Real densityAdv = 0.0;
	unsigned int numNeighbors = 0;

	//////////////////////////////////////////////////////////////////////////
	// Fluid
	//////////////////////////////////////////////////////////////////////////
	forall_fluid_neighborsGPU(
		const Vector3r &vj = fmVelocities[fmIndices[pid] + neighborIndex];
		densityAdv += fmVolumes[pid] * (vi - vj).dot(gradKernelWeightPrecomputed(Vector3r(xi.x - xj.x, xi.y - xj.y, xi.z - xj.z), kernelData));
	)

	//////////////////////////////////////////////////////////////////////////
	// Boundary
	//////////////////////////////////////////////////////////////////////////
	forall_boundary_neighborsGPU(
		const Vector3r &vj = bmVelocities[boundaryVolumeIndices[pid - nFluids] + neighborIndex];
		densityAdv += boundaryVolumes[boundaryVolumeIndices[pid - nFluids] + neighborIndex] * (vi - vj).dot(gradKernelWeightPrecomputed(Vector3r(xi.x - xj.x, xi.y - xj.y, xi.z - xj.z), kernelData));
	)
	
	// only correct positive divergence
	densityAdv = max(densityAdv, static_cast<Real>(0.0));

	for (unsigned int pid = 0; pid < nPointSets; pid++)
	{
		const uint neighborsetIndex = neighborPointsetIndices[fluidModelIndex] + pid;
		numNeighbors += neighborCounts[neighborsetIndex][i];
	}

	// in case of particle deficiency do not perform a divergence solve
	if (numNeighbors < 20)
		densityAdv = 0.0;

	density_errors[0] += densities0[fluidModelIndex] * densityAdv;
	densitiesAdv[fmIndices[fluidModelIndex] + i] = densityAdv;
}


__global__
void pressureSolveWarmstartComplete(/*out*/ Vector3r* const fmVelocities , /* output */ Vector3r* const forcesPerThread, /* output */ Vector3r* const torquesPerThread, 
	const uint* const forcesPerThreadIndices, const uint* const torquesPerThreadIndices, const Vector3r* const rigidBodyPositions, Real* const kappa, 
	const Real* const densitiesAdv, const Real* const masses, const Real* const fmVolumes, const uint* const fmIndices, const Real* const boundaryVolumes, 
	const uint* const boundaryVolumeIndices, const Real* const densities0, const bool* const isDynamic, const int tid, const Real h, const Real eps, const KernelData* const kernelData,
	/*start of forall-parameters*/ const Real3* const* __restrict__ particles, const uint* const* __restrict__ neighbors, const uint*  const* __restrict__ neighborCounts, const uint*  const* __restrict__ neighborOffsets, 
  uint* neighborPointsetIndices, const uint nFluids, const uint nPointSets, const uint fluidModelIndex, const uint numParticles)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i >= numParticles)
		return;

	if(densitiesAdv[fmIndices[fluidModelIndex] + i] > densities0[fluidModelIndex])
	{
		const Real invH = static_cast<Real>(1.0) / h;
		const Real invH2 = static_cast<Real>(1.0) / (h*h);

		extern __shared__ Real3 part[];
		part[threadIdx.x] = particles[fluidModelIndex][i];
		const Real3 &xi = part[threadIdx.x];

		Vector3r vel = fmVelocities[fmIndices[fluidModelIndex] + i];

		const Real ki = max( kappa[fmIndices[fluidModelIndex] + i] * invH2, -static_cast<Real>(0.5) * densities0[fluidModelIndex] * densities0[fluidModelIndex]);
		kappa[fmIndices[fluidModelIndex] + i] = ki;

		//////////////////////////////////////////////////////////////////////////
		// Fluid
		//////////////////////////////////////////////////////////////////////////
		forall_fluid_neighborsGPU(
			const Real kj = kappa[fmIndices[pid] + neighborIndex];

			const Real kSum = (ki + densities0[pid] / densities0[fluidModelIndex] * kj);
			if (fabsf(kSum) > eps)
			{
				const Vector3r grad_p_j = -fmVolumes[pid] * gradKernelWeightPrecomputed(Vector3r(xi.x - xj.x, xi.y - xj.y, xi.z - xj.z), kernelData);
				vel -= h * kSum * grad_p_j;					// ki, kj already contain inverse density
			}
		)

		//////////////////////////////////////////////////////////////////////////
		// Boundary
		//////////////////////////////////////////////////////////////////////////
		if (fabsf(ki) > eps)
		{
			forall_boundary_neighborsGPU(
				const Vector3r grad_p_j = -boundaryVolumes[boundaryVolumeIndices[pid - nFluids] + neighborIndex] * gradKernelWeightPrecomputed(Vector3r(xi.x - xj.x, xi.y - xj.y, xi.z - xj.z), kernelData);
				const Vector3r velChange = -h * (Real) 1.0 * ki * grad_p_j;				// kj already contains inverse density
				vel += velChange;
				addForce(Vector3r(xj.x, xj.y, xj.z), -masses[fmIndices[fluidModelIndex] + i] * velChange * invH, forcesPerThread, torquesPerThread, rigidBodyPositions, forcesPerThreadIndices, torquesPerThreadIndices, pid - nFluids, tid);
			)
		}

		fmVelocities[fmIndices[fluidModelIndex] + i] = vel;
	}
}

__global__
void pressureSolveKernel1(/*out*/ Real* const densitiesAdv, /* out */ Real* const factors, /* out */ Real* const kappa, const Real* const fmDensities, const Vector3r* const fmVelocities, const Vector3r* const bmVelocities, const uint* const fmIndices, 
	const Real* const fmVolumes, const Real* const boundaryVolumes, const uint* const boundaryVolumeIndices, const Real* const densities0, const Real h, const KernelData* const kernelData,
	/*start of forall-parameters*/ const Real3* const* __restrict__ particles, const uint* const* __restrict__ neighbors, const uint*  const* __restrict__ neighborCounts, const uint*  const* __restrict__ neighborOffsets, 
  uint* neighborPointsetIndices, const uint nFluids, const uint nPointSets, const uint fluidModelIndex, const uint numParticles)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i >= numParticles)
		return;

	extern __shared__ Real3 part[];
	part[threadIdx.x] = particles[fluidModelIndex][i];
	const Real3 &xi = part[threadIdx.x];

	const Real invH2 = static_cast<Real>(1.0) / (h*h);
	factors[fmIndices[fluidModelIndex] + i] *= invH2;

	kappa[fmIndices[fluidModelIndex] + i] = 0.0;

	const Real density = fmDensities[fmIndices[fluidModelIndex] + i];
	const Vector3r vi = fmVelocities[fmIndices[fluidModelIndex] + i];
	Real delta = 0.0;

	//////////////////////////////////////////////////////////////////////////
	// Fluid
	//////////////////////////////////////////////////////////////////////////
	forall_fluid_neighborsGPU(
		const Vector3r &vj = fmVelocities[fmIndices[pid] + neighborIndex];
		delta += fmVolumes[pid] * (vi - vj).dot(gradKernelWeightPrecomputed(Vector3r(xi.x - xj.x, xi.y - xj.y, xi.z - xj.z), kernelData));
	)

	//////////////////////////////////////////////////////////////////////////
	// Boundary
	//////////////////////////////////////////////////////////////////////////
	forall_boundary_neighborsGPU(
		const Vector3r &vj = bmVelocities[boundaryVolumeIndices[pid - nFluids] + neighborIndex];
		delta += boundaryVolumes[boundaryVolumeIndices[pid - nFluids] + neighborIndex] * (vi - vj).dot(gradKernelWeightPrecomputed(Vector3r(xi.x - xj.x, xi.y - xj.y, xi.z - xj.z), kernelData));
	)
	
	Real densityAdv = density / densities0[fluidModelIndex] + h*delta;
	densityAdv = max(densityAdv, static_cast<Real>(1.0));

	densitiesAdv[fmIndices[fluidModelIndex] + i] = densityAdv;
}

__global__
void pressureSolveKernel1(/*out*/ Real* const densitiesAdv, /* out */ Real* const factors, const Real* const fmDensities, const Vector3r* const fmVelocities, const Vector3r* const bmVelocities, const uint* const fmIndices, 
	const Real* const fmVolumes, const Real* const boundaryVolumes, const uint* const boundaryVolumeIndices, const Real* const densities0, const Real h, const KernelData* const kernelData,
	/*start of forall-parameters*/ const Real3* const* __restrict__ particles, const uint* const* __restrict__ neighbors, const uint*  const* __restrict__ neighborCounts, const uint*  const* __restrict__ neighborOffsets, 
  uint* neighborPointsetIndices, const uint nFluids, const uint nPointSets, const uint fluidModelIndex, const uint numParticles)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i >= numParticles)
		return;

	extern __shared__ Real3 part[];
	part[threadIdx.x] = particles[fluidModelIndex][i];
	const Real3 &xi = part[threadIdx.x];

	const Real invH2 = static_cast<Real>(1.0) / (h*h);
	factors[fmIndices[fluidModelIndex] + i] *= invH2;

	const Real density = fmDensities[fmIndices[fluidModelIndex] + i];
	const Vector3r vi = fmVelocities[fmIndices[fluidModelIndex] + i];
	Real delta = 0.0;

	//////////////////////////////////////////////////////////////////////////
	// Fluid
	//////////////////////////////////////////////////////////////////////////
	forall_fluid_neighborsGPU(
		const Vector3r &vj = fmVelocities[fmIndices[pid] + neighborIndex];
		delta += fmVolumes[pid] * (vi - vj).dot(gradKernelWeightPrecomputed(Vector3r(xi.x - xj.x, xi.y - xj.y, xi.z - xj.z), kernelData));
	)

	//////////////////////////////////////////////////////////////////////////
	// Boundary
	//////////////////////////////////////////////////////////////////////////
	forall_boundary_neighborsGPU(
		const Vector3r &vj = bmVelocities[boundaryVolumeIndices[pid - nFluids] + neighborIndex];
		delta += boundaryVolumes[boundaryVolumeIndices[pid - nFluids] + neighborIndex] * (vi - vj).dot(gradKernelWeightPrecomputed(Vector3r(xi.x - xj.x, xi.y - xj.y, xi.z - xj.z), kernelData));
	)
	
	Real densityAdv = density / densities0[fluidModelIndex] + h*delta;
	densityAdv = max(densityAdv, static_cast<Real>(1.0));

	densitiesAdv[fmIndices[fluidModelIndex] + i] = densityAdv;
}

__global__
void pressureSolveKernel2(/*out*/ Real* const densitiesAdv, /*out*/ Real* const density_error, const Real* const fmDensities, const Vector3r* const fmVelocities, const Vector3r* const bmVelocities, const uint* const fmIndices, 
	const Real* const fmVolumes, const Real* const boundaryVolumes, const uint* const boundaryVolumeIndices, const Real* const densities0, const Real h, const KernelData* const kernelData,
	/*start of forall-parameters*/ const Real3* const* __restrict__ particles, const uint* const* __restrict__ neighbors, const uint*  const* __restrict__ neighborCounts, const uint*  const* __restrict__ neighborOffsets, 
  uint* neighborPointsetIndices, const uint nFluids, const uint nPointSets, const uint fluidModelIndex, const uint numParticles)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i >= numParticles)
		return;

	extern __shared__ Real3 part[];
	part[threadIdx.x] = particles[fluidModelIndex][i];
	const Real3 &xi = part[threadIdx.x];

	const Real density = fmDensities[fmIndices[fluidModelIndex] + i];
	const Vector3r vi = fmVelocities[fmIndices[fluidModelIndex] + i];
	Real delta = 0.0;

	//////////////////////////////////////////////////////////////////////////
	// Fluid
	//////////////////////////////////////////////////////////////////////////
	forall_fluid_neighborsGPU(
		const Vector3r &vj = fmVelocities[fmIndices[pid] + neighborIndex];
		delta += fmVolumes[pid] * (vi - vj).dot(gradKernelWeightPrecomputed(Vector3r(xi.x - xj.x, xi.y - xj.y, xi.z - xj.z), kernelData));
	)

	//////////////////////////////////////////////////////////////////////////
	// Boundary
	//////////////////////////////////////////////////////////////////////////
	forall_boundary_neighborsGPU(
		const Vector3r &vj = bmVelocities[boundaryVolumeIndices[pid - nFluids] + neighborIndex];
		delta += boundaryVolumes[boundaryVolumeIndices[pid - nFluids] + neighborIndex] * (vi - vj).dot(gradKernelWeightPrecomputed(Vector3r(xi.x - xj.x, xi.y - xj.y, xi.z - xj.z), kernelData));
	)
	
	Real densityAdv = density / densities0[fluidModelIndex] + h*delta;
	densityAdv = max(densityAdv, static_cast<Real>(1.0));

	density_error[0] += densities0[fluidModelIndex] * densityAdv - densities0[fluidModelIndex];
	densitiesAdv[fmIndices[fluidModelIndex] + i] = densityAdv;
}
