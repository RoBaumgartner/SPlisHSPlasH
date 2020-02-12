#ifndef __WCSPHKernels_h__
#define __WCSPHKernels_h__

#include <cuda_runtime.h>
#include "SPlisHSPlasH/NeighborhoodSearch.h"
#include "SPlisHSPlasH/Common.h"
#include "SPlisHSPlasH/SPHKernels.h"
#include "Types.cuh"

using namespace cuNSearch;

const unsigned int PRECOMPUTED_KERNEL_SIZE = 10000;

//////////////////////////////////////////////////////////////////
// Helper class 
//////////////////////////////////////////////////////////////////

struct KernelData{
	Real *d_W, *d_gradW;
	Real radius, radius2, invStepSize;

	KernelData();
	~KernelData();
};

void updateKernelData(KernelData &data);

//////////////////////////////////////////////////////////////////
//Kernels for all methods 
//////////////////////////////////////////////////////////////////

__device__
Real kernelWeightPrecomputed(const Vector3r &r, const KernelData* const data);

__device__
Vector3r gradKernelWeightPrecomputed(const Vector3r &r, const KernelData* const data);

__device__ 
Real kernelWeight(const Vector3r& rin, const Real m_radius);

__device__
Vector3r gradKernelWeight(const Vector3r &rin, const Real m_radius);

__device__
void addForce(const Vector3r &pos, const Vector3r &f, /* output */ Vector3r* const forcesPerThread, /* output */ Vector3r* const torquesPerThread, 
	const Vector3r* const rigidBodyPositions, const cuNSearch::uint* const forcesPerThreadIndices, const cuNSearch::uint* const torquesPerThreadIndices, const cuNSearch::uint index, const int id);


__global__
void computeDensitiesGPU(/*out*/ Real* const densities, const Real* const volumes, const Real* const boundaryVolumes, const cuNSearch::uint* const boundaryVolumeIndices, 
	const cuNSearch::uint* const fmIndices, const Real* const densities0, const Real W_zero, const KernelData* const kernelData, /*start of forall-parameters*/ Real3** particles, cuNSearch::uint** neighbors, cuNSearch::uint** neighborCounts, cuNSearch::uint** neighborOffsets, 
	cuNSearch::uint* neighborPointsetIndices, const cuNSearch::uint nFluids, const cuNSearch::uint nPointSets, const cuNSearch::uint fluidModelIndex, const cuNSearch::uint numParticles);

//////////////////////////////////////////////////////////////////
//Kernels for WCSPH method 
//////////////////////////////////////////////////////////////////

__global__ 
void clearAccelerationsGPU(Real* masses, Vector3r* accelerations, const Vector3r grav, const unsigned int numActiveParticles);

__global__
void updatePressureGPU(Real* const densities, const cuNSearch::uint* const fmIndices, Real* pressures, const Real* const densities0, const Real m_stiffness, const Real m_exponent,
	const cuNSearch::uint fluidModelIndex, const cuNSearch::uint numParticles);

 __global__
void computePressureAccelsGPU( /* output */ Vector3r* const pressureAccels, /* output */ Vector3r* const forcesPerThread, /* output */ Vector3r* const torquesPerThread, const cuNSearch::uint* const forcesPerThreadIndices, 
	const cuNSearch::uint* const torquesPerThreadIndices, const Real* const densities, const Real* const densities0, const cuNSearch::uint* const fmIndices, const Real* const pressures, const Real* const masses, 
	const Vector3r* const rigidBodyPositions, const Real* const volumes, const Real* const boundaryVolumes, const cuNSearch::uint* const boundaryVolumeIndices, const bool* const isDynamic, const int tid, const KernelData* kernelData,
	/*start of forall-parameters*/ Real3** particles, cuNSearch::uint** neighbors, cuNSearch::uint** neighborCounts, cuNSearch::uint** neighborOffsets, 
  cuNSearch::uint* neighborPointsetIndices, const cuNSearch::uint nFluids, const cuNSearch::uint nPointSets, const cuNSearch::uint fluidModelIndex, const cuNSearch::uint numParticles);

__global__ 
void updatePosPressureAccelPressureAccel(Vector3r* const positions, Vector3r* const velocities, Vector3r* const accelerations,
	const Vector3r* const pressureAccels, const Real h, const cuNSearch::uint numParticles);


//////////////////////////////////////////////////////////////////
//Kernels for the DFSPH method 
//////////////////////////////////////////////////////////////////

__global__ 
void computeDFSPHFactors(/* out */ Real* factors, const Real* const boundaryVolumes, const cuNSearch::uint* const boundaryVolumeIndices, const KernelData* const kernelData, 
	const unsigned int* fmIndices, const Real* fmVolumes, const Real eps,
	/*start of forall-parameters*/ Real3** particles, cuNSearch::uint** neighbors, cuNSearch::uint** neighborCounts, cuNSearch::uint** neighborOffsets, 
  cuNSearch::uint* neighborPointsetIndices, const cuNSearch::uint nFluids, const cuNSearch::uint nPointSets, const cuNSearch::uint fluidModelIndex, const cuNSearch::uint numParticles);

 __global__
void computeDensityChanges(/* out */ Real* const densitiesAdv, const Vector3r* const fmVelocities, const Vector3r* const bmVelocities, const cuNSearch::uint* const fmIndices, 
	const Real* const fmVolumes, const Real* const boundaryVolumes, const cuNSearch::uint* const boundaryVolumeIndices, const KernelData* const kernelData,
	/*start of forall-parameters*/ Real3** particles, cuNSearch::uint** neighbors, cuNSearch::uint** neighborCounts, cuNSearch::uint** neighborOffsets, 
  cuNSearch::uint* neighborPointsetIndices, const cuNSearch::uint nFluids, const cuNSearch::uint nPointSets, const cuNSearch::uint fluidModelIndex, const cuNSearch::uint numParticles);

__global__
void computeDensityAdvs(/*out*/ Real* const densitiesAdv, const Real* const fmDensities, const Vector3r* const fmVelocities, const Vector3r* const bmVelocities, const cuNSearch::uint* const fmIndices, 
	const Real* const fmVolumes, const Real* const boundaryVolumes, const cuNSearch::uint* const boundaryVolumeIndices, const Real* const densities0, const Real h, const KernelData* const kernelData,
	/*start of forall-parameters*/ Real3** particles, cuNSearch::uint** neighbors, cuNSearch::uint** neighborCounts, cuNSearch::uint** neighborOffsets, 
  cuNSearch::uint* neighborPointsetIndices, const cuNSearch::uint nFluids, const cuNSearch::uint nPointSets, const cuNSearch::uint fluidModelIndex, const cuNSearch::uint numParticles);

__global__
void warmstartDivergenceSolveKappaV(/*out*/ Real* const kappaV, const cuNSearch::uint* const fmIndices, const Real* const densities0, const Real invH, const cuNSearch::uint fluidModelIndex, const cuNSearch::uint numParticles);

__global__
void divergenceSolveWarmstart( /*out*/ Vector3r* const fmVelocities, /* output */ Vector3r* const forcesPerThread, /* output */ Vector3r* const torquesPerThread, 
	const cuNSearch::uint* const forcesPerThreadIndices, const cuNSearch::uint* const torquesPerThreadIndices, const Vector3r* const rigidBodyPositions, const Real* const kappaV,
	const cuNSearch::uint* const fmIndices, const Real* const masses, const Real* const fmVolumes, const Real* const boundaryVolumes, const cuNSearch::uint* const boundaryVolumeIndices, 
	const Real* const densities0, const bool* const isDynamic, const int tid, const Real h, const KernelData* const kernelData, const Real eps,
	/*start of forall-parameters*/ Real3** particles, cuNSearch::uint** neighbors, cuNSearch::uint** neighborCounts, cuNSearch::uint** neighborOffsets, 
  cuNSearch::uint* neighborPointsetIndices, const cuNSearch::uint nFluids, const cuNSearch::uint nPointSets, const cuNSearch::uint fluidModelIndex, const cuNSearch::uint numParticles);

__global__
void multiplyRealWithConstant(/*out*/ Real* const input, const cuNSearch::uint* const fmIndices, const Real f, const cuNSearch::uint fluidModelIndex, const cuNSearch::uint numParticles);

__global__
void setRealToZero(/*out*/ Real* const input, const cuNSearch::uint* const fmIndices, const cuNSearch::uint fluidModelIndex, const cuNSearch::uint numParticles);

__global__ 
void divergenceSolveUpdateFluidVelocities( /*out*/ Vector3r* const fmVelocities, /*out*/ Real* const kappaV, /* output */ Vector3r* const forcesPerThread, /* output */ Vector3r* const torquesPerThread, 
	const cuNSearch::uint* const forcesPerThreadIndices, const cuNSearch::uint* const torquesPerThreadIndices, const Vector3r* const rigidBodyPositions, const Real* const densitiesAdv, const Real* const factors, 
	const cuNSearch::uint* const fmIndices, const Real* const masses, const Real* const fmVolumes, const Real* const boundaryVolumes, const cuNSearch::uint* const boundaryVolumeIndices, 
	const Real* const densities0, const bool* const isDynamic, const int tid, const Real h, const Real invH, const KernelData* const kernelData, const Real eps,
	/*start of forall-parameters*/ Real3** particles, cuNSearch::uint** neighbors, cuNSearch::uint** neighborCounts, cuNSearch::uint** neighborOffsets, 
	cuNSearch::uint* neighborPointsetIndices, const cuNSearch::uint nFluids, const cuNSearch::uint nPointSets, const cuNSearch::uint fluidModelIndex, const cuNSearch::uint numParticles);

__global__ 
void divergenceSolveUpdateFluidVelocities( /*out*/ Vector3r* const fmVelocities, /* output */ Vector3r* const forcesPerThread, /* output */ Vector3r* const torquesPerThread, 
	const cuNSearch::uint* const forcesPerThreadIndices, const cuNSearch::uint* const torquesPerThreadIndices, const Vector3r* const rigidBodyPositions, const Real* const densitiesAdv, const Real* const factors, 
	const cuNSearch::uint* const fmIndices, const Real* const masses, const Real* const fmVolumes, const Real* const boundaryVolumes, const cuNSearch::uint* const boundaryVolumeIndices, 
	const Real* const densities0, const bool* const isDynamic, const int tid, const Real h, const Real invH, const KernelData* const kernelData, const Real eps,
	/*start of forall-parameters*/ Real3** particles, cuNSearch::uint** neighbors, cuNSearch::uint** neighborCounts, cuNSearch::uint** neighborOffsets, 
	cuNSearch::uint* neighborPointsetIndices, const cuNSearch::uint nFluids, const cuNSearch::uint nPointSets, const cuNSearch::uint fluidModelIndex, const cuNSearch::uint numParticles);

__global__ 
void pressureSolveUpdateFluidVelocities( /*out*/ Vector3r* const fmVelocities, /*out*/ Vector3r* const forcesPerThread, /*out*/ Vector3r* const torquesPerThread, 
	const cuNSearch::uint* const forcesPerThreadIndices, const cuNSearch::uint* const torquesPerThreadIndices, const Vector3r* const rigidBodyPositions, const Real* const densitiesAdv, const Real* const factors, 
	const cuNSearch::uint* const fmIndices, const Real* const masses, const Real* const fmVolumes, const Real* const boundaryVolumes, const cuNSearch::uint* const boundaryVolumeIndices, 
	const Real* const densities0, const bool* const isDynamic, const int tid, const Real h, const Real invH, const KernelData* const kernelData, const Real eps,
	/*start of forall-parameters*/ Real3** particles, cuNSearch::uint** neighbors, cuNSearch::uint** neighborCounts, cuNSearch::uint** neighborOffsets, 
	cuNSearch::uint* neighborPointsetIndices, const cuNSearch::uint nFluids, const cuNSearch::uint nPointSets, const cuNSearch::uint fluidModelIndex, const cuNSearch::uint numParticles);

__global__ 
void pressureSolveUpdateFluidVelocities( /*out*/ Vector3r* const fmVelocities, /*out*/ Real* const kappa, /*out*/ Vector3r* const forcesPerThread, /*out*/ Vector3r* const torquesPerThread, 
	const cuNSearch::uint* const forcesPerThreadIndices, const cuNSearch::uint* const torquesPerThreadIndices, const Vector3r* const rigidBodyPositions, const Real* const densitiesAdv, const Real* const factors, 
	const cuNSearch::uint* const fmIndices, const Real* const masses, const Real* const fmVolumes, const Real* const boundaryVolumes, const cuNSearch::uint* const boundaryVolumeIndices, 
	const Real* const densities0, const bool* const isDynamic, const int tid, const Real h, const Real invH, const KernelData* const kernelData, const Real eps,
	/*start of forall-parameters*/ Real3** particles, cuNSearch::uint** neighbors, cuNSearch::uint** neighborCounts, cuNSearch::uint** neighborOffsets, 
  cuNSearch::uint* neighborPointsetIndices, const cuNSearch::uint nFluids, const cuNSearch::uint nPointSets, const cuNSearch::uint fluidModelIndex, const cuNSearch::uint numParticles);

__global__
void updateDensityErrorDivergence(/* out */ Real* const density_errors, const Real* const densitiesAdv, const Real* const densities0, const cuNSearch::uint* const fmIndices,
	const cuNSearch::uint fluidModelIndex, const cuNSearch::uint numParticles);

__global__
void warmstartPressureSolveKappa(/*out*/ Real* kappa, const cuNSearch::uint* const fmIndices, const Real* const densities0, const Real invH2, const cuNSearch::uint fluidModelIndex, const cuNSearch::uint numParticles);

__global__
void pressureSolveWarmstart(/*out*/ Vector3r* const fmVelocities , /* output */ Vector3r* const forcesPerThread, /* output */ Vector3r* const torquesPerThread, 
	const cuNSearch::uint* const forcesPerThreadIndices, const cuNSearch::uint* const torquesPerThreadIndices, const Vector3r* const rigidBodyPositions,const Real* const kappa, 
	const Real* const densitiesAdv, const Real* const masses, const Real* const fmVolumes, const cuNSearch::uint* const fmIndices, const Real* const boundaryVolumes, 
	const cuNSearch::uint* const boundaryVolumeIndices, const Real* const densities0, const bool* const isDynamic, const int tid, const Real h, const Real eps, const KernelData* const kernelData,
	/*start of forall-parameters*/ Real3** particles, cuNSearch::uint** neighbors, cuNSearch::uint** neighborCounts, cuNSearch::uint** neighborOffsets, 
  cuNSearch::uint* neighborPointsetIndices, const cuNSearch::uint nFluids, const cuNSearch::uint nPointSets, const cuNSearch::uint fluidModelIndex, const cuNSearch::uint numParticles);

__global__
void updateDensityErrorPressureSolve(/*out*/ Real* const density_error, const Real* const densitiesAdv, const Real* const densities0, const cuNSearch::uint* const fmIndices,
	const cuNSearch::uint fluidModelIndex, const cuNSearch::uint numParticles);

#endif