#include "TimeStepDFSPHGPU.h"
#include "SPlisHSPlasH/TimeManager.h"
#include "SPlisHSPlasH/SPHKernels.h"
#include "SimulationDataDFSPH.h"
#include <iostream>
#include "Utilities/Timing.h"
#include "Utilities/Counting.h"
#include "SPlisHSPlasH/Simulation.h"
#include "SPlisHSPlasH/BoundaryModel_Akinci2012.h"
#include "SPlisHSPlasH/BoundaryModel_Koschier2017.h"
#include "SPlisHSPlasH/BoundaryModel_Bender2019.h"

#include "PointSetImplementation.cuh"


using namespace SPH;
using namespace std;
using namespace GenParam;
using namespace cuNSearch;

#define USE_CORRECTED_FORMULATION

int TimeStepDFSPHGPU::SOLVER_ITERATIONS_V = -1;
int TimeStepDFSPHGPU::MAX_ITERATIONS_V = -1;
int TimeStepDFSPHGPU::MAX_ERROR_V = -1;
int TimeStepDFSPHGPU::USE_DIVERGENCE_SOLVER = -1;


TimeStepDFSPHGPU::TimeStepDFSPHGPU() :
	TimeStep(),
	m_simulationData()
{
	m_simulationData.init();
	m_counter = 0;
	m_iterationsV = 0;
	m_enableDivergenceSolver = true;
	m_maxIterationsV = 100;
	m_maxErrorV = 0.1;

	CudaHelper::CudaMalloc(&d_kernelData, 1);

	Simulation *sim = Simulation::getCurrent();
	const unsigned int nModels = sim->numberOfFluidModels();
	for (unsigned int fluidModelIndex = 0; fluidModelIndex < nModels; fluidModelIndex++)
	{
		FluidModel *model = sim->getFluidModel(fluidModelIndex);
		model->addField({ "factor", FieldType::Scalar, [this, fluidModelIndex](const unsigned int i) -> Real* { return &m_simulationData.getFactor(fluidModelIndex, i); } });
		model->addField({ "advected density", FieldType::Scalar, [this, fluidModelIndex](const unsigned int i) -> Real* { return &m_simulationData.getDensityAdv(fluidModelIndex, i); } });
		model->addField({ "kappa", FieldType::Scalar, [this, fluidModelIndex](const unsigned int i) -> Real* { return &m_simulationData.getKappa(fluidModelIndex, i); }, true });
		model->addField({ "kappa_v", FieldType::Scalar, [this, fluidModelIndex](const unsigned int i) -> Real* { return &m_simulationData.getKappaV(fluidModelIndex, i); }, true });
	}
}

TimeStepDFSPHGPU::~TimeStepDFSPHGPU(void)
{
	CudaHelper::CudaFree(d_kernelData);
	CudaHelper::CudaFree(d_neighbors);
	CudaHelper::CudaFree(d_neighborCounts);
	CudaHelper::CudaFree(d_neighborOffsets);
	CudaHelper::CudaFree(d_neighborPointsetIndices); 

#ifdef USE_WARMSTART
	CudaHelper::CudaFree(d_kappa);
#endif

#ifdef USE_WARMSTART_V
	CudaHelper::CudaFree(d_kappaV);
#endif

	CudaHelper::CudaFree(d_density_error);

	Simulation *sim = Simulation::getCurrent();
	const unsigned int nModels = sim->numberOfFluidModels();
	for (unsigned int fluidModelIndex = 0; fluidModelIndex < nModels; fluidModelIndex++)
	{
		FluidModel *model = sim->getFluidModel(fluidModelIndex);
		model->removeFieldByName("factor");
		model->removeFieldByName("advected density");
		model->removeFieldByName("kappa");
		model->removeFieldByName("kappa_v");
	}
}

void TimeStepDFSPHGPU::initParameters()
{
	TimeStep::initParameters();

	SOLVER_ITERATIONS_V = createNumericParameter("iterationsV", "Iterations (divergence)", &m_iterationsV);
	setGroup(SOLVER_ITERATIONS_V, "DFSPH");
	setDescription(SOLVER_ITERATIONS_V, "Iterations required by the divergence solver.");
	getParameter(SOLVER_ITERATIONS_V)->setReadOnly(true);

	MAX_ITERATIONS_V = createNumericParameter("maxIterationsV", "Max. iterations (divergence)", &m_maxIterationsV);
	setGroup(MAX_ITERATIONS_V, "DFSPH");
	setDescription(MAX_ITERATIONS_V, "Maximal number of iterations of the divergence solver.");
	static_cast<NumericParameter<unsigned int>*>(getParameter(MAX_ITERATIONS_V))->setMinValue(1);

	MAX_ERROR_V = createNumericParameter("maxErrorV", "Max. divergence error(%)", &m_maxErrorV);
	setGroup(MAX_ERROR_V, "DFSPH");
	setDescription(MAX_ERROR_V, "Maximal divergence error (%).");
	static_cast<RealParameter*>(getParameter(MAX_ERROR_V))->setMinValue(1e-6);

	USE_DIVERGENCE_SOLVER = createBoolParameter("enableDivergenceSolver", "Enable divergence solver", &m_enableDivergenceSolver);
	setGroup(USE_DIVERGENCE_SOLVER, "DFSPH");
	setDescription(USE_DIVERGENCE_SOLVER, "Turn divergence solver on/off.");
}

void TimeStepDFSPHGPU::initCUDA()
{ // sim init in static boundary simulator
	Simulation *sim = Simulation::getCurrent();
	const unsigned int nModels = sim->numberOfFluidModels();
	const unsigned int nPointSets = sim->numberOfPointSets();
	unsigned int sumOfAllFluidParticles = 0;

	std::vector<cuNSearch::PointSet> &pointSets = sim->getCurrent()->getPointSets();
	d_particles.resize(pointSets.size());
	for(int pid = 0 ; pid < pointSets.size() ; ++pid)
	{
		d_particles[pid] = CudaHelper::GetPointer(pointSets[pid].getPointSetImplementation()->getParticles());
	}

	d_volumes.resize(nModels);
	d_densities0.resize(nModels);
	for(unsigned int fluidModelIndex = 0; fluidModelIndex < nModels; fluidModelIndex++)
	{
		FluidModel *fm = sim->getFluidModel(fluidModelIndex);
		sumOfAllFluidParticles += fm->numberOfParticles();
		d_volumes[fluidModelIndex] = fm->getVolume(0);
		d_densities0[fluidModelIndex] = fm->getDensity0();
	}

	d_fmIndices.resize(nModels);

	d_rigidBodyPositions.resize(nPointSets - nModels);
	d_isDynamic.resize(nPointSets - nModels);
	d_forcesPerThreadIndices.resize(nPointSets - nModels); // TODO: Marcel
	d_torquesPerThreadIndices.resize(nPointSets - nModels); // TODO: Marcel

	d_boundaryVolumeIndices.resize(nPointSets - nModels);

	// allocate everything to its maximum potential size 
	CudaHelper::CudaMalloc(&d_neighborPointsetIndices, nPointSets);
	CudaHelper::CudaMalloc(&d_neighbors, nPointSets * nPointSets);
	CudaHelper::CudaMalloc(&d_neighborCounts, nPointSets * nPointSets);
	CudaHelper::CudaMalloc(&d_neighborOffsets, nPointSets * nPointSets);

	CudaHelper::CudaMalloc(&d_fmDensities, sumOfAllFluidParticles);	
	CudaHelper::CudaMalloc(&d_densitiesAdv, sumOfAllFluidParticles);
	CudaHelper::CudaMalloc(&d_factors, sumOfAllFluidParticles);

	CudaHelper::CudaMalloc(&d_density_error, 1);

#ifdef USE_WARMSTART	
	CudaHelper::CudaMalloc(&d_kappa, sumOfAllFluidParticles);
#endif

#ifdef USE_WARMSTART_V	
	CudaHelper::CudaMalloc(&d_kappaV, sumOfAllFluidParticles);
#endif

	d_masses.resize(sumOfAllFluidParticles);
	d_fmVelocities.resize(sumOfAllFluidParticles);

	// TODO: Marcel
	int sumForcesPerThread = 0;
	int sumTorquesPerThread = 0;
	int sumBoundaryParticles = 0;
	for (unsigned int boundaryModelIndex = nModels; boundaryModelIndex < nPointSets; boundaryModelIndex++)
	{
		BoundaryModel_Akinci2012 *bm_neighbor = static_cast<BoundaryModel_Akinci2012*>(sim->getBoundaryModelFromPointSet(boundaryModelIndex));	

		sumForcesPerThread += bm_neighbor->getForcesPerThread().size();
		sumTorquesPerThread += bm_neighbor->getTorquesPerThread().size();
		sumBoundaryParticles += bm_neighbor->getVelocities().size();
	}

	d_forcesPerThread.resize(sumForcesPerThread);
	d_torquesPerThread.resize(sumTorquesPerThread);
	d_bmVelocities.resize(sumBoundaryParticles);
	d_boundaryVolumes.resize(sumBoundaryParticles);
}

void TimeStepDFSPHGPU::step()
{
	Simulation *sim = Simulation::getCurrent();
	TimeManager *tm = TimeManager::getCurrent();
	const Real h = tm->getTimeStepSize();
	const unsigned int nModels = sim->numberOfFluidModels();
	const unsigned int nPointSets = sim->numberOfPointSets();

	performNeighborhoodSearch();

	if(!isInitialized)
	{
		initCUDA(); // TODO: shift this in init or constructor
	}

	prepareData();

	// re-compute the precomputed kernel if necessary
	if( sim->getSupportRadius() != PrecomputedKernel<CubicKernel, PRECOMPUTED_KERNEL_SIZE>::getRadius() || !isInitialized)
	{
		PrecomputedKernel<CubicKernel, PRECOMPUTED_KERNEL_SIZE>::setRadius(sim->getSupportRadius());
		updateKernelData(kernelData);
		CudaHelper::MemcpyHostToDevice(&kernelData, d_kernelData, 1);
		
		isInitialized = true;
	}

	START_TIMING("compute the densities");
	unsigned int sumActiveParticles = 0;
	for (unsigned int fluidModelIndex = 0; fluidModelIndex < nModels; fluidModelIndex++)
	{
		FluidModel *model = sim->getFluidModel(fluidModelIndex);
		const unsigned int numParticles = model->numActiveParticles();
		std::vector<cuNSearch::PointSet> &pointSets = sim->getCurrent()->getPointSets();
		PointSetImplementation *impl = pointSets[fluidModelIndex].getPointSetImplementation();
		const Real W_zero = sim->W_zero();

		computeDensitiesGPU<<<impl->getNumberOfBlocks(), impl->getThreadsPerBlock(), impl->getThreadsPerBlock() * sizeof(Real)>>>( d_fmDensities, CudaHelper::GetPointer(d_volumes), CudaHelper::GetPointer(d_boundaryVolumes), 
			CudaHelper::GetPointer(d_boundaryVolumeIndices), CudaHelper::GetPointer(d_fmIndices), CudaHelper::GetPointer(d_densities0), W_zero, d_kernelData, 
			CudaHelper::GetPointer(d_particles), d_neighbors, d_neighborCounts, d_neighborOffsets, d_neighborPointsetIndices, nModels, 
			nPointSets, fluidModelIndex, numParticles);

		CudaHelper::CheckLastError();	

		CudaHelper::MemcpyDeviceToHost(d_fmDensities + sumActiveParticles, &(model->getDensity(0)), sumParticles);
		sumActiveParticles += numParticles;
	}
	STOP_TIMING_AVG;

	START_TIMING("computeDFSPHFactor");
	for (unsigned int fluidModelIndex = 0; fluidModelIndex < nModels; fluidModelIndex++)
	{
		FluidModel *model = sim->getFluidModel(fluidModelIndex);
		const unsigned int numParticles = model->numActiveParticles();
		std::vector<cuNSearch::PointSet> &pointSets = sim->getCurrent()->getPointSets();
		PointSetImplementation *impl = pointSets[fluidModelIndex].getPointSetImplementation();

		computeDFSPHFactors<<<impl->getNumberOfBlocks(), impl->getThreadsPerBlock()>>>( d_factors, CudaHelper::GetPointer(d_boundaryVolumes), CudaHelper::GetPointer(d_boundaryVolumeIndices), d_kernelData, 
			CudaHelper::GetPointer(d_fmIndices), CudaHelper::GetPointer(d_volumes), m_eps, CudaHelper::GetPointer(d_particles), 
			d_neighbors, d_neighborCounts, d_neighborOffsets, d_neighborPointsetIndices, nModels, 
			nPointSets, fluidModelIndex, numParticles);

		CudaHelper::CheckLastError();
	}

	STOP_TIMING_AVG;

	if (m_enableDivergenceSolver)
	{
		sumActiveParticles = 0;
		for(unsigned int pid = 0; pid < nModels; pid++)
		{
			FluidModel *fm_neighbor = sim->getFluidModel(pid);
			const unsigned int numActiveParticles = fm_neighbor->numActiveParticles();

			d_fmVelocities.insert(d_fmVelocities.begin() + sumActiveParticles, fm_neighbor->getVelocities().begin(), fm_neighbor->getVelocities().begin() + numActiveParticles);
			sumActiveParticles += numActiveParticles;
		}

		START_TIMING("divergenceSolve");
		divergenceSolve();
		STOP_TIMING_AVG

		sumActiveParticles = 0;
		for(unsigned int fluidModelIndex = 0; fluidModelIndex < sim->numberOfFluidModels(); fluidModelIndex++)
		{
			FluidModel *fm_neighbor = sim->getFluidModel(fluidModelIndex);
			CudaHelper::MemcpyDeviceToHost( CudaHelper::GetPointer(d_fmVelocities) + sumActiveParticles, &(fm_neighbor->getVelocity(0)), sim->getFluidModel(fluidModelIndex)->numActiveParticles());
			sumActiveParticles += fm_neighbor->numActiveParticles();
		}
	}

	else
		m_iterationsV = 0;

	// Compute accelerations: a(t)
	for (unsigned int fluidModelIndex = 0; fluidModelIndex < nModels; fluidModelIndex++)
		clearAccelerations(fluidModelIndex);

	sim->computeNonPressureForces();

	sim->updateTimeStepSize();

	// compute new velocities only considering non-pressure forces
	for (unsigned int m = 0; m < nModels; m++)
	{
		FluidModel *fm = sim->getFluidModel(m);
		const unsigned int numParticles = fm->numActiveParticles();
		#pragma omp parallel default(shared)
		{
			#pragma omp for schedule(static)  
			for (int i = 0; i < (int)numParticles; i++)
			{
				if (fm->getParticleState(i) == ParticleState::Active)
				{
					Vector3r &vel = fm->getVelocity(i);
					vel += h * fm->getAcceleration(i);
				}
			}
		}
	}

	// put velocities on GPU again
	sumActiveParticles = 0;
	for(unsigned int pid = 0; pid < nModels; pid++)
	{
		FluidModel *fm_neighbor = sim->getFluidModel(pid);
		const unsigned int numActiveParticles = fm_neighbor->numActiveParticles();
		d_fmVelocities.insert(d_fmVelocities.begin() + sumActiveParticles, fm_neighbor->getVelocities().begin(), fm_neighbor->getVelocities().begin() + numActiveParticles);
		sumActiveParticles += numActiveParticles;
	}

	START_TIMING("pressureSolve");
	pressureSolve();
	STOP_TIMING_AVG;

	START_TIMING("Copy data back from GPU");
	getDataBack();
	STOP_TIMING_AVG;

	// compute final positions
	for (unsigned int m = 0; m < nModels; m++)
	{
		FluidModel *fm = sim->getFluidModel(m);
		const unsigned int numParticles = fm->numActiveParticles();
		#pragma omp parallel default(shared)
		{
			#pragma omp for schedule(static)  
			for (int i = 0; i < (int)numParticles; i++)
			{
				if (fm->getParticleState(i) == ParticleState::Active)
				{
					Vector3r &xi = fm->getPosition(i);
					const Vector3r &vi = fm->getVelocity(i);
					xi += h * vi;
				}
			}
		}
	}

	sim->emitParticles();
	sim->animateParticles();

	// Compute new time	
	tm->setTime (tm->getTime () + h);
}

#ifdef USE_WARMSTART
void TimeStepDFSPHGPU::warmstartPressureSolve(const unsigned int fluidModelIndex)
{
	const Real h = TimeManager::getCurrent()->getTimeStepSize();
	const Real h2 = h*h;
	const Real invH = static_cast<Real>(1.0) / h;
	const Real invH2 = static_cast<Real>(1.0) / h2;
	Simulation *sim = Simulation::getCurrent();
	FluidModel *model = sim->getFluidModel(fluidModelIndex);
	const Real density0 = model->getDensity0();
	const int numParticles = (int)model->numActiveParticles();
	const unsigned int nPointSets = sim->numberOfPointSets();
	std::vector<cuNSearch::PointSet> &pointSets = sim->getCurrent()->getPointSets();
	PointSetImplementation *impl = pointSets[fluidModelIndex].getPointSetImplementation();
	if (numParticles == 0)
		return;

	const unsigned int nFluids = sim->numberOfFluidModels();
	const unsigned int nBoundaries = sim->numberOfBoundaryModels();

	//////////////////////////////////////////////////////////////////////////
	// Divide by h^2, the time step size has been removed in 
	// the last step to make the stiffness value independent 
	// of the time step size
	//////////////////////////////////////////////////////////////////////////
	warmstartPressureSolveKappa<<<impl->getNumberOfBlocks(), impl->getThreadsPerBlock()>>>(d_kappa, CudaHelper::GetPointer(d_fmIndices), CudaHelper::GetPointer(d_densities0), invH2, fluidModelIndex, numParticles);

	CudaHelper::CheckLastError();

	pressureSolveWarmstart<<<impl->getNumberOfBlocks(), impl->getThreadsPerBlock()>>>( CudaHelper::GetPointer(d_fmVelocities), CudaHelper::GetPointer(d_forcesPerThread), CudaHelper::GetPointer(d_torquesPerThread), 
		CudaHelper::GetPointer(d_forcesPerThreadIndices), CudaHelper::GetPointer(d_torquesPerThreadIndices), CudaHelper::GetPointer(d_rigidBodyPositions), d_kappa, 
		d_densitiesAdv, CudaHelper::GetPointer(d_masses), CudaHelper::GetPointer(d_volumes), CudaHelper::GetPointer(d_fmIndices), CudaHelper::GetPointer(d_boundaryVolumes), 
		CudaHelper::GetPointer(d_boundaryVolumeIndices), CudaHelper::GetPointer(d_densities0), CudaHelper::GetPointer(d_isDynamic), omp_get_thread_num(), h, m_eps, d_kernelData,
		CudaHelper::GetPointer(d_particles), d_neighbors, d_neighborCounts, d_neighborOffsets, d_neighborPointsetIndices, nFluids, 
		nPointSets, fluidModelIndex, numParticles);	

	CudaHelper::CheckLastError();
}
#endif

void TimeStepDFSPHGPU::pressureSolve()
{
	const Real h = TimeManager::getCurrent()->getTimeStepSize();
	const Real h2 = h*h;
	const Real invH = static_cast<Real>(1.0) / h;
	const Real invH2 = static_cast<Real>(1.0) / h2;
	Simulation *sim = Simulation::getCurrent();
	const unsigned int nFluids = sim->numberOfFluidModels();
	const unsigned int nPointSets = sim->numberOfPointSets();
	unsigned int sumActiveParticles = 0;

#ifdef USE_WARMSTART	
	sumActiveParticles = 0;
	for (unsigned int fluidModelIndex = 0; fluidModelIndex < nFluids; fluidModelIndex++)
	{
		CudaHelper::MemcpyHostToDevice( &(m_simulationData.getKappa(fluidModelIndex, 0)), d_kappa + sumActiveParticles, sim->getFluidModel(fluidModelIndex)->numActiveParticles());
		sumActiveParticles += sim->getFluidModel(fluidModelIndex)->numActiveParticles();
	}

	for (unsigned int fluidModelIndex = 0; fluidModelIndex < nFluids; fluidModelIndex++)
		warmstartPressureSolve(fluidModelIndex);
#endif

	//////////////////////////////////////////////////////////////////////////
	// Compute rho_adv
	//////////////////////////////////////////////////////////////////////////

	for (unsigned int fluidModelIndex = 0; fluidModelIndex < nFluids; fluidModelIndex++)
	{
		FluidModel *model = sim->getFluidModel(fluidModelIndex);
		const int numParticles = (int)model->numActiveParticles();
		std::vector<cuNSearch::PointSet> &pointSets = sim->getCurrent()->getPointSets();
		PointSetImplementation *impl = pointSets[fluidModelIndex].getPointSetImplementation();

		computeDensityAdvs<<<impl->getNumberOfBlocks(), impl->getThreadsPerBlock()>>>(d_densitiesAdv, d_fmDensities, CudaHelper::GetPointer(d_fmVelocities), CudaHelper::GetPointer(d_bmVelocities),
			CudaHelper::GetPointer(d_fmIndices), CudaHelper::GetPointer(d_volumes), CudaHelper::GetPointer(d_boundaryVolumes), CudaHelper::GetPointer(d_boundaryVolumeIndices),
			CudaHelper::GetPointer(d_densities0), h, d_kernelData, CudaHelper::GetPointer(d_particles), d_neighbors, d_neighborCounts, d_neighborOffsets, d_neighborPointsetIndices, nFluids, 
			nPointSets, fluidModelIndex, numParticles);

		CudaHelper::CheckLastError();
		CudaHelper::DeviceSynchronize();
		
		multiplyRealWithConstant<<<impl->getNumberOfBlocks(), impl->getThreadsPerBlock()>>>( d_factors, CudaHelper::GetPointer(d_fmIndices), invH2, fluidModelIndex, numParticles);

		CudaHelper::CheckLastError();

	#ifdef USE_WARMSTART
		setRealToZero<<<impl->getNumberOfBlocks(), impl->getThreadsPerBlock()>>>(d_kappa, CudaHelper::GetPointer(d_fmIndices), fluidModelIndex, numParticles);
		CudaHelper::CheckLastError();
	#endif
	}

	m_iterations = 0;

	//////////////////////////////////////////////////////////////////////////
	// Start solver
	//////////////////////////////////////////////////////////////////////////
	
	Real avg_density_err = 0.0;
	bool chk = false;

	
	while ((!chk || (m_iterations < m_minIterations)) && (m_iterations < m_maxIterations))
	{
		chk = true;
		for (unsigned int i = 0; i < nFluids; i++)
		{
			FluidModel *model = sim->getFluidModel(i);
			const Real density0 = model->getDensity0();

			avg_density_err = 0.0;
			pressureSolveIteration(i, avg_density_err);

			// Maximal allowed density fluctuation
			const Real eta = m_maxError * static_cast<Real>(0.01) * density0;  // maxError is given in percent
			chk = chk && (avg_density_err <= eta);
		}

		m_iterations++;
	}

	INCREASE_COUNTER("DFSPH - iterations", static_cast<Real>(m_iterations));

#ifdef USE_WARMSTART
	//////////////////////////////////////////////////////////////////////////
	// Multiply by h^2, the time step size has to be removed 
	// to make the stiffness value independent 
	// of the time step size
	//////////////////////////////////////////////////////////////////////////
	for (unsigned int fluidModelIndex = 0; fluidModelIndex < nFluids; fluidModelIndex++)
	{
		FluidModel *model = sim->getFluidModel(fluidModelIndex);
		std::vector<cuNSearch::PointSet> &pointSets = sim->getCurrent()->getPointSets();
		PointSetImplementation *impl = pointSets[fluidModelIndex].getPointSetImplementation();
		const int numParticles = (int)model->numActiveParticles();
		
		multiplyRealWithConstant<<<impl->getNumberOfBlocks(), impl->getThreadsPerBlock()>>>( d_kappa, CudaHelper::GetPointer(d_fmIndices), h2, fluidModelIndex, numParticles);
		CudaHelper::CheckLastError();
	}

	sumActiveParticles = 0;
	for(unsigned int fluidModelIndex = 0; fluidModelIndex < nFluids; fluidModelIndex++)
	{
		CudaHelper::MemcpyDeviceToHost(d_kappa + sumActiveParticles, &(m_simulationData.getKappa(fluidModelIndex, 0)), sim->getFluidModel(fluidModelIndex)->numActiveParticles());
		sumActiveParticles += sim->getFluidModel(fluidModelIndex)->numActiveParticles();
	}
#endif
}

void TimeStepDFSPHGPU::pressureSolveIteration(const unsigned int fluidModelIndex, Real &avg_density_err)
{
	Simulation *sim = Simulation::getCurrent();
	FluidModel *model = sim->getFluidModel(fluidModelIndex);
	const Real density0 = model->getDensity0();
	const int numParticles = (int)model->numActiveParticles();
	const unsigned int nPointSets = sim->numberOfPointSets();
	std::vector<cuNSearch::PointSet> &pointSets = sim->getCurrent()->getPointSets();
	PointSetImplementation *impl = pointSets[fluidModelIndex].getPointSetImplementation();
	if (numParticles == 0)
		return;

	const unsigned int nFluids = sim->numberOfFluidModels();
	const Real h = TimeManager::getCurrent()->getTimeStepSize();
	const Real invH = static_cast<Real>(1.0) / h;
	//Real density_error = 0.0;

		Real density_error = 0.0;
		CudaHelper::MemcpyHostToDevice( &density_error, d_density_error, 1);

#ifdef USE_WARMSTART
	pressureSolveUpdateFluidVelocities<<<impl->getNumberOfBlocks(), impl->getThreadsPerBlock()>>>( CudaHelper::GetPointer(d_fmVelocities), d_kappa, CudaHelper::GetPointer(d_forcesPerThread), 
		CudaHelper::GetPointer(d_torquesPerThread), CudaHelper::GetPointer(d_forcesPerThreadIndices), CudaHelper::GetPointer(d_torquesPerThreadIndices), 
		CudaHelper::GetPointer(d_rigidBodyPositions), d_densitiesAdv, d_factors, CudaHelper::GetPointer(d_fmIndices), CudaHelper::GetPointer(d_masses), 
		CudaHelper::GetPointer(d_volumes), CudaHelper::GetPointer(d_boundaryVolumes), CudaHelper::GetPointer(d_boundaryVolumeIndices), 
		CudaHelper::GetPointer(d_densities0), CudaHelper::GetPointer(d_isDynamic), omp_get_thread_num(), h, invH, d_kernelData, m_eps, 
		CudaHelper::GetPointer(d_particles), d_neighbors, d_neighborCounts, d_neighborOffsets, d_neighborPointsetIndices, nFluids, 
		nPointSets, fluidModelIndex, numParticles);
#else
	pressureSolveUpdateFluidVelocities<<<impl->getNumberOfBlocks(), impl->getThreadsPerBlock()>>>( CudaHelper::GetPointer(d_fmVelocities), CudaHelper::GetPointer(d_forcesPerThread), 
		CudaHelper::GetPointer(d_torquesPerThread), CudaHelper::GetPointer(d_forcesPerThreadIndices), CudaHelper::GetPointer(d_torquesPerThreadIndices), 
		CudaHelper::GetPointer(d_rigidBodyPositions), d_densitiesAdv, d_factors, CudaHelper::GetPointer(d_fmIndices), CudaHelper::GetPointer(d_masses), 
		CudaHelper::GetPointer(d_volumes), CudaHelper::GetPointer(d_boundaryVolumes), CudaHelper::GetPointer(d_boundaryVolumeIndices), 
		CudaHelper::GetPointer(d_densities0), CudaHelper::GetPointer(d_isDynamic), omp_get_thread_num(), h, invH, d_kernelData, m_eps, 
		CudaHelper::GetPointer(d_particles), d_neighbors, d_neighborCounts, d_neighborOffsets, d_neighborPointsetIndices, nFluids, 
		nPointSets, fluidModelIndex, numParticles);
#endif

	CudaHelper::CheckLastError();

	computeDensityAdvs<<<impl->getNumberOfBlocks(), impl->getThreadsPerBlock()>>>(d_densitiesAdv, d_fmDensities, CudaHelper::GetPointer(d_fmVelocities), CudaHelper::GetPointer(d_bmVelocities),
		CudaHelper::GetPointer(d_fmIndices), CudaHelper::GetPointer(d_volumes), CudaHelper::GetPointer(d_boundaryVolumes), CudaHelper::GetPointer(d_boundaryVolumeIndices),
		CudaHelper::GetPointer(d_densities0), h, d_kernelData, CudaHelper::GetPointer(d_particles), d_neighbors, d_neighborCounts, 
		d_neighborOffsets, d_neighborPointsetIndices, nFluids, nPointSets, fluidModelIndex, numParticles);

	CudaHelper::CheckLastError();

	updateDensityErrorPressureSolve<<<impl->getNumberOfBlocks(), impl->getThreadsPerBlock()>>>( d_density_error, d_densitiesAdv, CudaHelper::GetPointer(d_densities0), 
		CudaHelper::GetPointer(d_fmIndices), fluidModelIndex, numParticles);

	CudaHelper::CheckLastError();

	CudaHelper::MemcpyDeviceToHost(d_density_error, &density_error, 1);
	avg_density_err = density_error / numParticles;
}

#ifdef USE_WARMSTART_V
void TimeStepDFSPHGPU::warmstartDivergenceSolve(const unsigned int fluidModelIndex)
{
	const Real h = TimeManager::getCurrent()->getTimeStepSize();
	const Real invH = static_cast<Real>(1.0) / h;
	Simulation *sim = Simulation::getCurrent();
	FluidModel *model = sim->getFluidModel(fluidModelIndex);
	const Real density0 = model->getDensity0();
	const int numParticles = (int)model->numActiveParticles();
	const unsigned int nPointSets = sim->numberOfPointSets();
	std::vector<cuNSearch::PointSet> &pointSets = sim->getCurrent()->getPointSets();
	PointSetImplementation *impl = pointSets[fluidModelIndex].getPointSetImplementation();
	if (numParticles == 0)
		return;

	const unsigned int nFluids = sim->numberOfFluidModels();
	const unsigned int nBoundaries = sim->numberOfBoundaryModels();

	warmstartDivergenceSolveKappaV<<<impl->getNumberOfBlocks(), impl->getThreadsPerBlock()>>>(d_kappaV, CudaHelper::GetPointer(d_fmIndices), CudaHelper::GetPointer(d_densities0), invH, fluidModelIndex, numParticles);

	CudaHelper::CheckLastError();

	computeDensityChanges<<<impl->getNumberOfBlocks(), impl->getThreadsPerBlock()>>>(d_densitiesAdv, CudaHelper::GetPointer(d_fmVelocities), CudaHelper::GetPointer(d_bmVelocities),
		CudaHelper::GetPointer(d_fmIndices), CudaHelper::GetPointer(d_volumes), CudaHelper::GetPointer(d_boundaryVolumes), CudaHelper::GetPointer(d_boundaryVolumeIndices), 
		d_kernelData, CudaHelper::GetPointer(d_particles), d_neighbors, d_neighborCounts, d_neighborOffsets, d_neighborPointsetIndices, nFluids, 
		nPointSets, fluidModelIndex, numParticles);

	CudaHelper::CheckLastError();

	divergenceSolveWarmstart<<<impl->getNumberOfBlocks(), impl->getThreadsPerBlock()>>>( CudaHelper::GetPointer(d_fmVelocities), CudaHelper::GetPointer(d_forcesPerThread), 
		CudaHelper::GetPointer(d_torquesPerThread), CudaHelper::GetPointer(d_forcesPerThreadIndices), CudaHelper::GetPointer(d_torquesPerThreadIndices), 
		CudaHelper::GetPointer(d_rigidBodyPositions), d_kappaV, CudaHelper::GetPointer(d_fmIndices), CudaHelper::GetPointer(d_masses), 
		CudaHelper::GetPointer(d_volumes), CudaHelper::GetPointer(d_boundaryVolumes), CudaHelper::GetPointer(d_boundaryVolumeIndices), 
		CudaHelper::GetPointer(d_densities0), CudaHelper::GetPointer(d_isDynamic), omp_get_thread_num(), h, d_kernelData, m_eps, 
		CudaHelper::GetPointer(d_particles), d_neighbors, d_neighborCounts, d_neighborOffsets, d_neighborPointsetIndices, nFluids, 
		nPointSets, fluidModelIndex, numParticles);

	CudaHelper::CheckLastError();
}
#endif

void TimeStepDFSPHGPU::divergenceSolve()
{
	//////////////////////////////////////////////////////////////////////////
	// Init parameters
	//////////////////////////////////////////////////////////////////////////

	const Real h = TimeManager::getCurrent()->getTimeStepSize();
	const Real invH = static_cast<Real>(1.0) / h;
	Simulation *sim = Simulation::getCurrent();
	const unsigned int maxIter = m_maxIterationsV;
	const Real maxError = m_maxErrorV;
	const unsigned int nFluids = sim->numberOfFluidModels();
	const unsigned int nPointSets = sim->numberOfPointSets();
	unsigned int sumActiveParticles = 0; // helper for data transfers

#ifdef USE_WARMSTART_V	
	sumActiveParticles = 0;
	for (unsigned int fluidModelIndex = 0; fluidModelIndex < nFluids; fluidModelIndex++)
	{
		CudaHelper::MemcpyHostToDevice( &(m_simulationData.getKappaV(fluidModelIndex, 0)), d_kappaV + sumActiveParticles, sim->getFluidModel(fluidModelIndex)->numActiveParticles());
		sumActiveParticles += sim->getFluidModel(fluidModelIndex)->numActiveParticles();
	}

	for(unsigned int fluidModelIndex =0; fluidModelIndex < nFluids; fluidModelIndex++)
		warmstartDivergenceSolve(fluidModelIndex);
#endif

	//////////////////////////////////////////////////////////////////////////
	// Compute velocity of density change
	//////////////////////////////////////////////////////////////////////////
	for (unsigned int fluidModelIndex = 0; fluidModelIndex < nFluids; fluidModelIndex++)
	{
		FluidModel *model = sim->getFluidModel(fluidModelIndex);
		std::vector<cuNSearch::PointSet> &pointSets = sim->getCurrent()->getPointSets();
		PointSetImplementation *impl = pointSets[fluidModelIndex].getPointSetImplementation();
		const int numParticles = (int)model->numActiveParticles();

		computeDensityChanges<<<impl->getNumberOfBlocks(), impl->getThreadsPerBlock()>>>(d_densitiesAdv, CudaHelper::GetPointer(d_fmVelocities), CudaHelper::GetPointer(d_bmVelocities),
			CudaHelper::GetPointer(d_fmIndices), CudaHelper::GetPointer(d_volumes), CudaHelper::GetPointer(d_boundaryVolumes), CudaHelper::GetPointer(d_boundaryVolumeIndices), 
			d_kernelData, CudaHelper::GetPointer(d_particles), d_neighbors, d_neighborCounts, d_neighborOffsets, d_neighborPointsetIndices, nFluids, 
			nPointSets, fluidModelIndex, numParticles);

		CudaHelper::CheckLastError();

		multiplyRealWithConstant<<<impl->getNumberOfBlocks(), impl->getThreadsPerBlock()>>>( d_factors, CudaHelper::GetPointer(d_fmIndices), invH, fluidModelIndex, numParticles);

		CudaHelper::CheckLastError();

#ifdef USE_WARMSTART_V
		setRealToZero<<<impl->getNumberOfBlocks(), impl->getThreadsPerBlock()>>>(d_kappaV, CudaHelper::GetPointer(d_fmIndices), fluidModelIndex, numParticles);
		CudaHelper::CheckLastError();
#endif
	}

	m_iterationsV = 0;

	//////////////////////////////////////////////////////////////////////////
	// Start solver
	//////////////////////////////////////////////////////////////////////////
	
	Real avg_density_err = 0.0;
	bool chk = false;

	while ((!chk || (m_iterationsV < 1)) && (m_iterationsV < maxIter))
	{
		chk = true;
		for (unsigned int fluidModelIndex = 0; fluidModelIndex < nFluids; fluidModelIndex++)
		{
			FluidModel *model = sim->getFluidModel(fluidModelIndex);
			const Real density0 = model->getDensity0();

			avg_density_err = 0.0;
			divergenceSolveIteration(fluidModelIndex, avg_density_err);
			
			// Maximal allowed density fluctuation
			// use maximal density error divided by time step size
			const Real eta = (static_cast<Real>(1.0) / h) * maxError * static_cast<Real>(0.01) * density0;  // maxError is given in percent
			chk = chk && (avg_density_err <= eta);
		}

		m_iterationsV++;
	}

	INCREASE_COUNTER("DFSPH - iterationsV", static_cast<Real>(m_iterationsV));

	//////////////////////////////////////////////////////////////////////////
	// Multiply by h, the time step size has to be removed 
	// to make the stiffness value independent 
	// of the time step size
	//////////////////////////////////////////////////////////////////////////
	for (unsigned int fluidModelIndex = 0; fluidModelIndex < nFluids; fluidModelIndex++)
	{
		FluidModel *model = sim->getFluidModel(fluidModelIndex);
		std::vector<cuNSearch::PointSet> &pointSets = sim->getCurrent()->getPointSets();
		PointSetImplementation *impl = pointSets[fluidModelIndex].getPointSetImplementation();
		const int numParticles = (int)model->numActiveParticles();

#ifdef USE_WARMSTART_V
		multiplyRealWithConstant<<<impl->getNumberOfBlocks(), impl->getThreadsPerBlock()>>>( d_kappaV, CudaHelper::GetPointer(d_fmIndices), h, fluidModelIndex, numParticles);
		CudaHelper::CheckLastError();
#endif

		multiplyRealWithConstant<<<impl->getNumberOfBlocks(), impl->getThreadsPerBlock()>>>( d_factors, CudaHelper::GetPointer(d_fmIndices), h, fluidModelIndex, numParticles);
		CudaHelper::CheckLastError();
	}

#ifdef USE_WARMSTART_V
	sumActiveParticles = 0;
	for(unsigned int fluidModelIndex = 0; fluidModelIndex < nFluids; fluidModelIndex++)
	{
		CudaHelper::MemcpyDeviceToHost(d_kappaV + sumActiveParticles, &(m_simulationData.getKappaV(fluidModelIndex, 0)), sim->getFluidModel(fluidModelIndex)->numActiveParticles());
		sumActiveParticles += sim->getFluidModel(fluidModelIndex)->numActiveParticles();
	}
#endif
}

void TimeStepDFSPHGPU::divergenceSolveIteration(const unsigned int fluidModelIndex, Real &avg_density_err)
{
	Simulation *sim = Simulation::getCurrent();
	FluidModel *model = sim->getFluidModel(fluidModelIndex);
	const Real density0 = model->getDensity0();
	const int numParticles = (int)model->numActiveParticles();
	const unsigned int nPointSets = sim->numberOfPointSets();
	std::vector<cuNSearch::PointSet> &pointSets = sim->getCurrent()->getPointSets();
	PointSetImplementation *impl = pointSets[fluidModelIndex].getPointSetImplementation();
	if (numParticles == 0)
		return;

	const unsigned int nFluids = sim->numberOfFluidModels();
	const unsigned int nBoundaries = sim->numberOfBoundaryModels();
	const Real h = TimeManager::getCurrent()->getTimeStepSize();
	const Real invH = static_cast<Real>(1.0) / h;
	Real density_error = 0.0;
/* 	Real density_error = 0.0, *d_density_error;

	CudaHelper::CudaMalloc(&d_density_error, 1);
	CudaHelper::MemcpyHostToDevice( &density_error, d_density_error, 1); */

	//////////////////////////////////////////////////////////////////////////
	// Perform Jacobi iteration over all blocks
	//////////////////////////////////////////////////////////////////////////	
#ifdef USE_WARMSTART_V
	divergenceSolveUpdateFluidVelocities<<<impl->getNumberOfBlocks(), impl->getThreadsPerBlock()>>>( CudaHelper::GetPointer(d_fmVelocities), d_kappaV, CudaHelper::GetPointer(d_forcesPerThread), 
		CudaHelper::GetPointer(d_torquesPerThread), CudaHelper::GetPointer(d_forcesPerThreadIndices), CudaHelper::GetPointer(d_torquesPerThreadIndices), 
		CudaHelper::GetPointer(d_rigidBodyPositions), d_densitiesAdv, d_factors, CudaHelper::GetPointer(d_fmIndices), CudaHelper::GetPointer(d_masses), 
		CudaHelper::GetPointer(d_volumes), CudaHelper::GetPointer(d_boundaryVolumes), CudaHelper::GetPointer(d_boundaryVolumeIndices), 
		CudaHelper::GetPointer(d_densities0), CudaHelper::GetPointer(d_isDynamic), omp_get_thread_num(), h, invH, d_kernelData, m_eps, 
		CudaHelper::GetPointer(d_particles), d_neighbors, d_neighborCounts, d_neighborOffsets, d_neighborPointsetIndices, nFluids, 
		nPointSets, fluidModelIndex, numParticles);
#else
	divergenceSolveUpdateFluidVelocities<<<impl->getNumberOfBlocks(), impl->getThreadsPerBlock()>>>( CudaHelper::GetPointer(d_fmVelocities), CudaHelper::GetPointer(d_forcesPerThread), 
		CudaHelper::GetPointer(d_torquesPerThread), CudaHelper::GetPointer(d_forcesPerThreadIndices), CudaHelper::GetPointer(d_torquesPerThreadIndices), 
		CudaHelper::GetPointer(d_rigidBodyPositions), d_densitiesAdv, d_factors, CudaHelper::GetPointer(d_fmIndices), CudaHelper::GetPointer(d_masses), 
		CudaHelper::GetPointer(d_volumes), CudaHelper::GetPointer(d_boundaryVolumes), CudaHelper::GetPointer(d_boundaryVolumeIndices), 
		CudaHelper::GetPointer(d_densities0), CudaHelper::GetPointer(d_isDynamic), omp_get_thread_num(), h, invH, d_kernelData, m_eps, 
		CudaHelper::GetPointer(d_particles), d_neighbors, d_neighborCounts, d_neighborOffsets, d_neighborPointsetIndices, nFluids, 
		nPointSets, fluidModelIndex, numParticles);
#endif

	CudaHelper::CheckLastError();

	computeDensityChanges<<<impl->getNumberOfBlocks(), impl->getThreadsPerBlock()>>>(d_densitiesAdv, CudaHelper::GetPointer(d_fmVelocities), CudaHelper::GetPointer(d_bmVelocities),
		CudaHelper::GetPointer(d_fmIndices), CudaHelper::GetPointer(d_volumes), CudaHelper::GetPointer(d_boundaryVolumes), CudaHelper::GetPointer(d_boundaryVolumeIndices), 
		d_kernelData, CudaHelper::GetPointer(d_particles), d_neighbors, d_neighborCounts, d_neighborOffsets, d_neighborPointsetIndices, nFluids, 
		nPointSets, fluidModelIndex, numParticles);

	CudaHelper::CheckLastError();

	unsigned int sumActiveParticles = 0;
	for(unsigned int fluidModelIndex = 0; fluidModelIndex < nFluids; fluidModelIndex++)
	{
		CudaHelper::MemcpyDeviceToHost( d_densitiesAdv + sumActiveParticles, &(m_simulationData.getDensityAdv(fluidModelIndex, 0)), sim->getFluidModel(fluidModelIndex)->numActiveParticles());
		sumActiveParticles += sim->getFluidModel(fluidModelIndex)->numActiveParticles();
	}

	#pragma omp parallel default(shared)
	{
		#pragma omp for reduction(+:density_error) schedule(static) 
		for (int i = 0; i < (int)numParticles; i++)
		{
			density_error += density0 * m_simulationData.getDensityAdv(fluidModelIndex, i);
		}
	}

	sumActiveParticles = 0;
	for(unsigned int fluidModelIndex = 0; fluidModelIndex < nFluids; fluidModelIndex++)
	{
		CudaHelper::MemcpyHostToDevice( &(m_simulationData.getDensityAdv(fluidModelIndex, 0)), d_densitiesAdv + sumActiveParticles, sim->getFluidModel(fluidModelIndex)->numActiveParticles());
		sumActiveParticles += sim->getFluidModel(fluidModelIndex)->numActiveParticles();
	}

/* 	updateDensityErrorDivergence<<<impl->getNumberOfBlocks(), impl->getThreadsPerBlock()>>>( d_density_error, d_densitiesAdv, CudaHelper::GetPointer(d_densities0), 
		CudaHelper::GetPointer(d_fmIndices), fluidModelIndex, numParticles);

	CudaHelper::CheckLastError();
	CudaHelper::DeviceSynchronize();

	CudaHelper::MemcpyDeviceToHost(d_density_error, &density_error, 1);
	CudaHelper::CudaFree(d_density_error); */
	
	avg_density_err = density_error/numParticles;
}


void TimeStepDFSPHGPU::prepareData()
{
	Simulation *sim = Simulation::getCurrent();
	const unsigned int nPointSets = sim->numberOfPointSets();
	const unsigned int nFluids = sim->numberOfFluidModels();

	//////////////////////////////////////////////////////////////////////////////
	// Common data
	//////////////////////////////////////////////////////////////////////////////

	std::vector<cuNSearch::PointSet> &pointSets = sim->getCurrent()->getPointSets();
	std::vector<unsigned int> neighborPointsetIndices_tmp(nPointSets);

	unsigned int neighborsetCount = 0;
	for(int i = 0 ; i < nPointSets ; ++i)
	{
		neighborPointsetIndices_tmp[i] = neighborsetCount;
		neighborsetCount += pointSets[i].n_neighborsets();	
	}

	CudaHelper::MemcpyHostToDevice(neighborPointsetIndices_tmp.data(), d_neighborPointsetIndices, nPointSets);

	// flattened out the structures for efficiency
	for(int i = 0 ; i < nPointSets ; ++i)
	{
		const unsigned int nNeighborsets = pointSets[i].n_neighborsets();

		std::vector<uint*> neighbors_tmp(nNeighborsets);
		std::vector<uint*> neighborCounts_tmp(nNeighborsets);
		std::vector<uint*> neighborOffsets_tmp(nNeighborsets);

		for(int j = 0; j < nNeighborsets; j++)
		{
			neighbors_tmp[j] = pointSets[i].neighbor_indices(j);
			neighborCounts_tmp[j] = pointSets[i].neighbor_counts(j);
			neighborOffsets_tmp[j] = pointSets[i].neighbor_offsets(j);
		}

		CudaHelper::MemcpyHostToDevice(neighbors_tmp.data(), d_neighbors + neighborPointsetIndices_tmp[i], nNeighborsets);
		CudaHelper::MemcpyHostToDevice(neighborCounts_tmp.data(), d_neighborCounts + neighborPointsetIndices_tmp[i], nNeighborsets);
		CudaHelper::MemcpyHostToDevice(neighborOffsets_tmp.data(), d_neighborOffsets + neighborPointsetIndices_tmp[i], nNeighborsets);
	}

	// for computeDensities and computePressureAccels
	unsigned int sumBoundaryVolumes = 0;
	for(unsigned int pid = nFluids; pid < nPointSets; pid++)
	{
		BoundaryModel_Akinci2012 *bm_neighbor = static_cast<BoundaryModel_Akinci2012*>(sim->getBoundaryModelFromPointSet(pid));
		d_boundaryVolumes.insert(d_boundaryVolumes.begin() + sumBoundaryVolumes, bm_neighbor->getVolumes().begin(), bm_neighbor->getVolumes().end());

		d_boundaryVolumeIndices[pid - nFluids] = sumBoundaryVolumes;
		sumBoundaryVolumes += bm_neighbor->getVolumes().size();	
	}

	sumParticles = 0;
	for(unsigned int fluidModelIndex = 0; fluidModelIndex < nFluids; fluidModelIndex++)
	{
		FluidModel *model = sim->getFluidModel(fluidModelIndex);
		const unsigned int numActiveParticles = model->numActiveParticles();

		d_masses.insert(d_masses.begin() + sumParticles, model->getMasses().begin(), model->getMasses().begin() + numActiveParticles);
		d_fmIndices[fluidModelIndex] = sumParticles;
		sumParticles += numActiveParticles;
	}

	////////////////////////////////////////////////////////////////////////////
	// DFPSH specific
	////////////////////////////////////////////////////////////////////////////
	
	// TODO: Marcel
	int sumForcesPerThread = 0;
	int sumTorquesPerThread = 0;
	int sumBoundaryParticles = 0;
	for (unsigned int boundaryModelIndex = nFluids; boundaryModelIndex < nPointSets; boundaryModelIndex++)
	{
		BoundaryModel_Akinci2012 *bm_neighbor = static_cast<BoundaryModel_Akinci2012*>(sim->getBoundaryModelFromPointSet(boundaryModelIndex));	
		d_forcesPerThread.insert(d_forcesPerThread.begin() + sumForcesPerThread, bm_neighbor->getForcesPerThread().begin(), bm_neighbor->getForcesPerThread().end());
		d_torquesPerThread.insert(d_torquesPerThread.begin() + sumTorquesPerThread, bm_neighbor->getTorquesPerThread().begin(), bm_neighbor->getTorquesPerThread().end());
		
		d_forcesPerThreadIndices[boundaryModelIndex - nFluids] = sumForcesPerThread;
		d_torquesPerThreadIndices[boundaryModelIndex - nFluids] = sumTorquesPerThread;
		
		d_rigidBodyPositions[boundaryModelIndex - nFluids] = bm_neighbor->getRigidBodyPosition();
		d_isDynamic[boundaryModelIndex - nFluids] = bm_neighbor->isDynamic();
		d_bmVelocities.insert(d_bmVelocities.begin() + sumBoundaryParticles, bm_neighbor->getVelocities().begin(), bm_neighbor->getVelocities().end());

		sumForcesPerThread += bm_neighbor->getForcesPerThread().size();
		sumTorquesPerThread += bm_neighbor->getTorquesPerThread().size();
		sumBoundaryParticles += bm_neighbor->getVelocities().size();
	}
}
	
void TimeStepDFSPHGPU::getDataBack()
{
	Simulation *sim = Simulation::getCurrent();
	const unsigned int nPointSets = sim->numberOfPointSets();
	const unsigned int nFluids = sim->numberOfFluidModels();
	unsigned int sumActiveParticles = 0;

	for(unsigned int fluidModelIndex = 0; fluidModelIndex < nFluids; fluidModelIndex++)
	{
		FluidModel *fm_neighbor = sim->getFluidModel(fluidModelIndex);
	//	CudaHelper::MemcpyDeviceToHost( d_factors + sumActiveParticles, &(m_simulationData.getFactor(fluidModelIndex, 0)), sim->getFluidModel(fluidModelIndex)->numActiveParticles());
	//	CudaHelper::MemcpyDeviceToHost( d_densitiesAdv + sumActiveParticles, &(m_simulationData.getDensityAdv(fluidModelIndex, 0)), sim->getFluidModel(fluidModelIndex)->numActiveParticles());
		CudaHelper::MemcpyDeviceToHost( CudaHelper::GetPointer(d_fmVelocities) + sumActiveParticles, &(fm_neighbor->getVelocity(0)), sim->getFluidModel(fluidModelIndex)->numActiveParticles());
		sumActiveParticles += sim->getFluidModel(fluidModelIndex)->numActiveParticles();
	}
	
	// TODO: Marcel
	unsigned int sumForcesPerThread = 0;
	unsigned int sumTorquesPerThread = 0;
	sumActiveParticles = 0;
	for (unsigned int boundaryModelIndex = nFluids; boundaryModelIndex < sim->numberOfPointSets(); boundaryModelIndex++)
	{
		BoundaryModel_Akinci2012 *bm_neighbor = static_cast<BoundaryModel_Akinci2012*>(sim->getBoundaryModelFromPointSet(boundaryModelIndex));
	
		CudaHelper::MemcpyDeviceToHost( CudaHelper::GetPointer(d_forcesPerThread) + sumForcesPerThread, &(bm_neighbor->getForcesPerThread()[0]), bm_neighbor->getForcesPerThread().size());
		CudaHelper::MemcpyDeviceToHost( CudaHelper::GetPointer(d_torquesPerThread) + sumTorquesPerThread, &(bm_neighbor->getTorquesPerThread()[0]), bm_neighbor->getTorquesPerThread().size());
	
		sumForcesPerThread += bm_neighbor->getForcesPerThread().size();
		sumTorquesPerThread += bm_neighbor->getTorquesPerThread().size();
	}
}

void TimeStepDFSPHGPU::reset()
{
	TimeStep::reset();
	m_simulationData.reset();
	m_counter = 0;
	m_iterations = 0;
	m_iterationsV = 0;
}

void TimeStepDFSPHGPU::performNeighborhoodSearch()
{
	if (Simulation::getCurrent()->zSortEnabled())
	{
		if (m_counter % 500 == 0)
		{
			Simulation::getCurrent()->performNeighborhoodSearchSort();
			m_simulationData.performNeighborhoodSearchSort();
		}
		m_counter++;
	}

	Simulation::getCurrent()->performNeighborhoodSearch();
}

void TimeStepDFSPHGPU::emittedParticles(FluidModel *model, const unsigned int startIndex)
{
	m_simulationData.emittedParticles(model, startIndex);
}

void TimeStepDFSPHGPU::resize()
{
	m_simulationData.init();
}
