#ifndef MULTIGRIDSOLVER_H
#define MULTIGRIDSOLVER_H

#include <queue>
#include <vector>

#include "utility.h"

/* Data structure for Priority Queue */
struct VertexPair {
	int		vId;
	double	distance;
	bool	operator> (const VertexPair& ref) const { return distance > ref.distance; }
	bool	operator< (const VertexPair& ref) const { return distance < ref.distance; }
};

/* Enum to set solver type */
enum Sampling {
	FASTDISK = 0,
	RANDOM = 1,
	MIS = 2
};

enum Weighting {
	BARYCENTRIC = 0,
	UNIFORM = 1,
	INVDIST = 2
};

namespace GravoMG {
	class MultigridSolver
	{
	public:
		MultigridSolver(Eigen::MatrixXd& V, Eigen::MatrixXi& neigh, Eigen::SparseMatrix<double>& M);
		~MultigridSolver();
		/* Hierarchy-related methods */
		void buildHierarchy();
		void constructProlongation();
		double computeAverageEdgeLength(const Eigen::MatrixXd& pos, const Eigen::MatrixXi& neigh);
		double inTriangle(const Eigen::RowVector3d& p, const std::vector<int>& tri, const Eigen::RowVector3d& triNormal, const Eigen::MatrixXd& pos, Eigen::RowVector3d& bary, std::map<int, float>& insideEdge);
		std::vector<double> uniformWeights(const int& n_points);
		std::vector<double> inverseDistanceWeights(const Eigen::MatrixXd& pos, const Eigen::RowVector3d& source, const std::vector<int>& edges);
		std::vector<int> maximumDeltaIndependentSet(const Eigen::MatrixXd& pos, const Eigen::MatrixXi& edges, const double& radius);
		std::vector<int> maximumDeltaIndependentSet(const Eigen::MatrixXd& pos, const Eigen::MatrixXi& edges, const double& radius, Eigen::VectorXd& D, std::vector<size_t>& nearestSourceK);
		std::vector<int> fastDiskSample(const Eigen::MatrixXd& pos, const Eigen::MatrixXi& edges, const double& radius, Eigen::VectorXd& D, std::vector<size_t>& nearestSourceK);
		void constructDijkstraWithCluster(const Eigen::MatrixXd& points, const std::vector<int>& source, const Eigen::MatrixXi& neigh, int k, Eigen::VectorXd& D, std::vector<size_t>& nearestSourceK);

		// Multigrid solvers	
		// Using Gauss-Seidel Smoother (our main process)
		double multiGridVCycleGS(Eigen::SparseMatrix<double>& A, Eigen::MatrixXd& b, Eigen::MatrixXd& x, int k, bool isDebug = true);
		double multiGridFCycleGS(Eigen::SparseMatrix<double>& A, Eigen::MatrixXd& b, Eigen::MatrixXd& x, int k, bool isDebug = true);
		double multiGridWCycleGS(Eigen::SparseMatrix<double>& A, Eigen::MatrixXd& b, Eigen::MatrixXd& x, int k, bool isDebug = true);
		void GaussSeidelSmoother(Eigen::SparseMatrix<double>& LHS, Eigen::MatrixXd& rhs,
			Eigen::MatrixXd& x, int maxIter, double tol, bool isDebug = true);

		double residualCheck(const Eigen::SparseMatrix<double>& A, const Eigen::MatrixXd& b, const Eigen::MatrixXd& x, int type);

		/* Core solver function */
		void solve(Eigen::SparseMatrix<double>& LHS, Eigen::MatrixXd& rhs, Eigen::MatrixXd& x);

		/* Position-related variables */
		Eigen::MatrixXd						V;							//!< Coordinates of each vertex
		Eigen::MatrixXd						V0;							//!< Original coordinates of each vertex
		Eigen::MatrixXd						normals;
		Eigen::SparseMatrix<double>			M;							//!<mass matrix to use>
		Eigen::SparseMatrix<double>			Minv;							//!<inverse mass matrix to use>
		std::vector<Eigen::MatrixXd> 		levelV;
		std::vector<Eigen::MatrixXi> 		levelE;
		std::vector<std::vector<bool>>		prolongationFallback;
		std::vector<std::vector<std::vector<int>>>		allTriangles;
		std::vector<Eigen::MatrixXd>		levelN;
		
		/* Hierarchy data*/
		std::vector<size_t>								DoF;							//!< Degrees of freedom per level
		std::vector<Eigen::SparseMatrix<double>>		U;								//!< Prolongation operator
		std::vector<Eigen::SparseMatrix<double>>		UNeigh;					
		std::vector<Eigen::SparseMatrix<double>>		Abar, Sbar, Mbar;				//!< Reduced system matrices
		std::vector<Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower | Eigen::Upper>> cgSolvers;
		std::vector<size_t>								sampleIDHierarchy;
		std::vector<Eigen::MatrixXi>					neighHierarchy;
		std::vector<std::vector<size_t>>				nearestSource;                  // the shortest-distanced source from a given point
		std::vector<Eigen::VectorXd>					geodDistance;                   // Geodesic distance on each level (based on the number of samples on each level)
		int												geodesicType = 0;                 // Type of the hierarchy disk: 0 => HSIM-type || 1 => GVD-type (non-overlap) || 2 => combinatorial (only points in kNN)
		int												prolongationFunction = 0;			// Type of mapping from geodesic distance to prolongation operator value. 0 => constant, 1 => linear, 2 => 5th order polynomial,  3 => 2nd order polynomial
		std::vector<std::vector<int>>					samples;						
		std::vector<size_t>								PointsToSampleMap;
		std::vector<std::vector<size_t>>				PointsToSampleMaps;             
		int									cycleType = 0;								//0: V-cycle, 1: F-cycle, 2: W-cycle
		int									factorizationType = 0;						//0: LLT,	1: LDLT  
		bool                                isSmootherGaussSeidel;
		bool                                isGVDTwoRings;
		bool								isClusterExpanded;
		bool                                isPartitionOfUnity;
		bool								isUsingLocalMax;
		bool								nested = false;								// Whether to use a nested hierarchy or move the coarse points
		int									knnNeigh;
		int									stoppingCriteria = 0;						//!< Types of norm for residual check => 0: rel. norm (||Ax-b||/||b||)		1: L2 M^-1 (||Ax-b||_{M-1}/||b||_{M-1})		2: L2 M (||Ax-b||{M}/||b||_{M})		3: Abs (Ax-b).norm()
		int									maxIter = 50;
		int									lowBound = 1000;
		double								ratio = 8;									// The fraction of points to keep in each level: 1 / ratio
		Sampling							samplingStrategy = FASTDISK;				// Which sampling strategy to use. Options are: FASTDISK, POISSONDISK, FPS, RANDOM, MIS
		Weighting							weightingScheme = BARYCENTRIC;				// Which weighting scheme to use. Options are: BARYCENTRIC, UNIFORM, INVDIST

		Eigen::MatrixXi						neigh;										//!< Neighboring informaiton of the lowest level

		/* Solver-related data */
		int preIters;
		int postIters;
		double accuracy = 5e-4;
		Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> coarsestSolver;
		bool POUtruncated = true;

		/* Dummy data*/
		Eigen::MatrixXd						sampledPoints;
		Eigen::MatrixXd						Vplanar;
		Eigen::MatrixXi						Fplanar;
		int hierarchyType;							//0: non-nested, 1:GVDE
		std::vector<Eigen::MatrixXd>		newCentroidHierarchy;

		/* Logging and timing */
		std::map<std::string, double> hierarchyTiming;
		std::map<std::string, double> solverTiming;
		std::vector<std::tuple<double, double>> convergence;
		bool verbose = true;
		bool debug = false;

		/* Ablation parameters */
		bool								ablation = false;
		int									ablationNumPoints = 3;
		bool								ablationRandom = false;

	private:

	};


}

#endif // !MULTIGRIDSOLVER_H
