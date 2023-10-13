

// #include <climits>
// #include <vector>
// #include <math.h>
// #include <stdio.h>
// #include <string.h>
// #include <ctype.h>
// #include <time.h>
// #include <sys/time.h>
// #include <iostream>
// #include <unistd.h>

// #if _OPENMP
// # include <omp.h>
// #endif

// #include "lulesh.h"

/* Work Routines */
class Domain {

   public:

   // Constructor
   Domain(int numRanks, int colLoc,
          int rowLoc, int planeLoc,
          int nx, int tp, int nr, int balance, int cost);

   // Destructor
   ~Domain();

   //
   // ALLOCATION
   //

   void AllocateNodePersistent(int numNode) // Node-centered
   {
      m_x.resize(numNode);  // coordinates
      m_y.resize(numNode);
      m_z.resize(numNode);

      m_xd.resize(numNode); // velocities
      m_yd.resize(numNode);
      m_zd.resize(numNode);

      m_xdd.resize(numNode); // accelerations
      m_ydd.resize(numNode);
      m_zdd.resize(numNode);

      m_fx.resize(numNode);  // forces
      m_fy.resize(numNode);
      m_fz.resize(numNode);

      m_nodalMass.resize(numNode);  // mass
   }

   void AllocateElemPersistent(int numElem) // Elem-centered
   {
      m_nodelist.resize(8*numElem);

      // elem connectivities through face
      m_lxim.resize(numElem);
      m_lxip.resize(numElem);
      m_letam.resize(numElem);
      m_letap.resize(numElem);
      m_lzetam.resize(numElem);
      m_lzetap.resize(numElem);

      m_elemBC.resize(numElem);

      m_e.resize(numElem);
      m_p.resize(numElem);

      m_q.resize(numElem);
      m_ql.resize(numElem);
      m_qq.resize(numElem);

      m_v.resize(numElem);

      m_volo.resize(numElem);
      m_delv.resize(numElem);
      m_vdov.resize(numElem);

      m_arealg.resize(numElem);

      m_ss.resize(numElem);

      m_elemMass.resize(numElem);

      m_vnew.resize(numElem) ;
   }

   void AllocateGradients(int numElem, int allElem)
   {
      // Position gradients
      m_delx_xi   = Allocate<double>(numElem) ;
      m_delx_eta  = Allocate<double>(numElem) ;
      m_delx_zeta = Allocate<double>(numElem) ;

      // Velocity gradients
      m_delv_xi   = Allocate<double>(allElem) ;
      m_delv_eta  = Allocate<double>(allElem);
      m_delv_zeta = Allocate<double>(allElem) ;
   }

   void DeallocateGradients()
   {
      Release(&m_delx_zeta);
      Release(&m_delx_eta) ;
      Release(&m_delx_xi)  ;

      Release(&m_delv_zeta);
      Release(&m_delv_eta) ;
      Release(&m_delv_xi)  ;
   }

   void AllocateStrains(int numElem)
   {
      m_dxx = Allocate<double>(numElem) ;
      m_dyy = Allocate<double>(numElem) ;
      m_dzz = Allocate<double>(numElem) ;
   }

   void DeallocateStrains()
   {
      Release(&m_dzz) ;
      Release(&m_dyy) ;
      Release(&m_dxx) ;
   }
   
   //
   // ACCESSORS
   //

   // Node-centered

   // Nodal coordinates
   double& x(int idx)    { return m_x[idx] ; }
   double& y(int idx)    { return m_y[idx] ; }
   double& z(int idx)    { return m_z[idx] ; }

   // Nodal velocities
   double& xd(int idx)   { return m_xd[idx] ; }
   double& yd(int idx)   { return m_yd[idx] ; }
   double& zd(int idx)   { return m_zd[idx] ; }

   // Nodal accelerations
   double& xdd(int idx)  { return m_xdd[idx] ; }
   double& ydd(int idx)  { return m_ydd[idx] ; }
   double& zdd(int idx)  { return m_zdd[idx] ; }

   // Nodal forces
   double& fx(int idx)   { return m_fx[idx] ; }
   double& fy(int idx)   { return m_fy[idx] ; }
   double& fz(int idx)   { return m_fz[idx] ; }

   // Nodal mass
   double& nodalMass(int idx) { return m_nodalMass[idx] ; }

   // Nodes on symmertry planes
   int symmX(int idx) { return m_symmX[idx] ; }
   int symmY(int idx) { return m_symmY[idx] ; }
   int symmZ(int idx) { return m_symmZ[idx] ; }
   bool symmXempty()          { return m_symmX.empty(); }
   bool symmYempty()          { return m_symmY.empty(); }
   bool symmZempty()          { return m_symmZ.empty(); }

   //
   // Element-centered
   //
   int&  regElemSize(int idx) { return m_regElemSize[idx] ; }
   int&  regNumList(int idx) { return m_regNumList[idx] ; }
   int*  regNumList()            { return &m_regNumList[0] ; }
   int*  regElemlist(int r)    { return m_regElemlist[r] ; }
   int&  regElemlist(int r, int idx) { return m_regElemlist[r][idx] ; }

   int*  nodelist(int idx)    { return &m_nodelist[int(8)*idx] ; }

   // elem connectivities through face
   int&  lxim(int idx) { return m_lxim[idx] ; }
   int&  lxip(int idx) { return m_lxip[idx] ; }
   int&  letam(int idx) { return m_letam[idx] ; }
   int&  letap(int idx) { return m_letap[idx] ; }
   int&  lzetam(int idx) { return m_lzetam[idx] ; }
   int&  lzetap(int idx) { return m_lzetap[idx] ; }

   // elem face symm/free-surface flag
   int&  elemBC(int idx) { return m_elemBC[idx] ; }

   // Principal strains - temporary
   double& dxx(int idx)  { return m_dxx[idx] ; }
   double& dyy(int idx)  { return m_dyy[idx] ; }
   double& dzz(int idx)  { return m_dzz[idx] ; }

   // New relative volume - temporary
   double& vnew(int idx)  { return m_vnew[idx] ; }

   // Velocity gradient - temporary
   double& delv_xi(int idx)    { return m_delv_xi[idx] ; }
   double& delv_eta(int idx)   { return m_delv_eta[idx] ; }
   double& delv_zeta(int idx)  { return m_delv_zeta[idx] ; }

   // Position gradient - temporary
   double& delx_xi(int idx)    { return m_delx_xi[idx] ; }
   double& delx_eta(int idx)   { return m_delx_eta[idx] ; }
   double& delx_zeta(int idx)  { return m_delx_zeta[idx] ; }

   // Energy
   double& e(int idx)          { return m_e[idx] ; }

   // Pressure
   double& p(int idx)          { return m_p[idx] ; }

   // Artificial viscosity
   double& q(int idx)          { return m_q[idx] ; }

   // Linear term for q
   double& ql(int idx)         { return m_ql[idx] ; }
   // Quadratic term for q
   double& qq(int idx)         { return m_qq[idx] ; }

   // Relative volume
   double& v(int idx)          { return m_v[idx] ; }
   double& delv(int idx)       { return m_delv[idx] ; }

   // Reference volume
   double& volo(int idx)       { return m_volo[idx] ; }

   // volume derivative over volume
   double& vdov(int idx)       { return m_vdov[idx] ; }

   // Element characteristic length
   double& arealg(int idx)     { return m_arealg[idx] ; }

   // Sound speed
   double& ss(int idx)         { return m_ss[idx] ; }

   // Element mass
   double& elemMass(int idx)  { return m_elemMass[idx] ; }

   int nodeElemCount(int idx)
   { return m_nodeElemStart[idx+1] - m_nodeElemStart[idx] ; }

   int *nodeElemCornerList(int idx)
   { return &m_nodeElemCornerList[m_nodeElemStart[idx]] ; }

   // Parameters 

   // Cutoffs
   double u_cut()               { return m_u_cut ; }
   double e_cut()               { return m_e_cut ; }
   double p_cut()               { return m_p_cut ; }
   double q_cut()               { return m_q_cut ; }
   double v_cut()               { return m_v_cut ; }

   // Other constants (usually are settable via input file in real codes)
   double hgcoef()              { return m_hgcoef ; }
   double qstop()               { return m_qstop ; }
   double monoq_max_slope()     { return m_monoq_max_slope ; }
   double monoq_limiter_mult()  { return m_monoq_limiter_mult ; }
   double ss4o3()               { return m_ss4o3 ; }
   double qlc_monoq()           { return m_qlc_monoq ; }
   double qqc_monoq()           { return m_qqc_monoq ; }
   double qqc()                 { return m_qqc ; }

   double eosvmax()             { return m_eosvmax ; }
   double eosvmin()             { return m_eosvmin ; }
   double pmin()                { return m_pmin ; }
   double emin()                { return m_emin ; }
   double dvovmax()             { return m_dvovmax ; }
   double refdens()             { return m_refdens ; }

   // Timestep controls, etc...
   double& time()                 { return m_time ; }
   double& deltatime()            { return m_deltatime ; }
   double& deltatimemultlb()      { return m_deltatimemultlb ; }
   double& deltatimemultub()      { return m_deltatimemultub ; }
   double& stoptime()             { return m_stoptime ; }
   double& dtcourant()            { return m_dtcourant ; }
   double& dthydro()              { return m_dthydro ; }
   double& dtmax()                { return m_dtmax ; }
   double& dtfixed()              { return m_dtfixed ; }

   int&  cycle()                { return m_cycle ; }
   int&  numRanks()           { return m_numRanks ; }

   int&  colLoc()             { return m_colLoc ; }
   int&  rowLoc()             { return m_rowLoc ; }
   int&  planeLoc()           { return m_planeLoc ; }
   int&  tp()                 { return m_tp ; }

   int&  sizeX()              { return m_sizeX ; }
   int&  sizeY()              { return m_sizeY ; }
   int&  sizeZ()              { return m_sizeZ ; }
   int&  numReg()             { return m_numReg ; }
   int&  cost()             { return m_cost ; }
   int&  numElem()            { return m_numElem ; }
   int&  numNode()            { return m_numNode ; }
   
   int&  maxPlaneSize()       { return m_maxPlaneSize ; }
   int&  maxEdgeSize()        { return m_maxEdgeSize ; }
   
   //
   // MPI-Related additional data
   //

#if USE_MPI   
   // Communication Work space 
   double *commDataSend ;
   double *commDataRecv ;
   
   // Maximum number of block neighbors 
   MPI_Request recvRequest[26] ; // 6 faces + 12 edges + 8 corners 
   MPI_Request sendRequest[26] ; // 6 faces + 12 edges + 8 corners 
#endif

  private:

   void BuildMesh(int nx, int edgeNodes, int edgeElems);
   void SetupThreadSupportStructures();
   void CreateRegionIndexSets(int nreg, int balance);
   void SetupCommBuffers(int edgeNodes);
   void SetupSymmetryPlanes(int edgeNodes);
   void SetupElementConnectivities(int edgeElems);
   void SetupBoundaryConditions(int edgeElems);

   //
   // IMPLEMENTATION
   //

   /* Node-centered */
   std::vector<double> m_x ;  /* coordinates */
   std::vector<double> m_y ;
   std::vector<double> m_z ;

   std::vector<double> m_xd ; /* velocities */
   std::vector<double> m_yd ;
   std::vector<double> m_zd ;

   std::vector<double> m_xdd ; /* accelerations */
   std::vector<double> m_ydd ;
   std::vector<double> m_zdd ;

   std::vector<double> m_fx ;  /* forces */
   std::vector<double> m_fy ;
   std::vector<double> m_fz ;

   std::vector<double> m_nodalMass ;  /* mass */

   std::vector<int> m_symmX ;  /* symmetry plane nodesets */
   std::vector<int> m_symmY ;
   std::vector<int> m_symmZ ;

   // Element-centered

   // Region information
   int    m_numReg ;
   int    m_cost; //imbalance cost
   int *m_regElemSize ;   // Size of region sets
   int *m_regNumList ;    // Region number per domain element
   int **m_regElemlist ;  // region indexset 

   std::vector<int>  m_nodelist ;     /* elemToNode connectivity */

   std::vector<int>  m_lxim ;  /* element connectivity across each face */
   std::vector<int>  m_lxip ;
   std::vector<int>  m_letam ;
   std::vector<int>  m_letap ;
   std::vector<int>  m_lzetam ;
   std::vector<int>  m_lzetap ;

   std::vector<int>    m_elemBC ;  /* symmetry/free-surface flags for each elem face */

   double             *m_dxx ;  /* principal strains -- temporary */
   double             *m_dyy ;
   double             *m_dzz ;

   double             *m_delv_xi ;    /* velocity gradient -- temporary */
   double             *m_delv_eta ;
   double             *m_delv_zeta ;

   double             *m_delx_xi ;    /* coordinate gradient -- temporary */
   double             *m_delx_eta ;
   double             *m_delx_zeta ;
   
   std::vector<double> m_e ;   /* energy */

   std::vector<double> m_p ;   /* pressure */
   std::vector<double> m_q ;   /* q */
   std::vector<double> m_ql ;  /* linear term for q */
   std::vector<double> m_qq ;  /* quadratic term for q */

   std::vector<double> m_v ;     /* relative volume */
   std::vector<double> m_volo ;  /* reference volume */
   std::vector<double> m_vnew ;  /* new relative volume -- temporary */
   std::vector<double> m_delv ;  /* m_vnew - m_v */
   std::vector<double> m_vdov ;  /* volume derivative over volume */

   std::vector<double> m_arealg ;  /* characteristic length of an element */
   
   std::vector<double> m_ss ;      /* "sound speed" */

   std::vector<double> m_elemMass ;  /* mass */

   // Cutoffs (treat as constants)
   double  m_e_cut ;             // energy tolerance 
   double  m_p_cut ;             // pressure tolerance 
   double  m_q_cut ;             // q tolerance 
   double  m_v_cut ;             // relative volume tolerance 
   double  m_u_cut ;             // velocity tolerance 

   // Other constants (usually setable, but hardcoded in this proxy app)

   double  m_hgcoef ;            // hourglass control 
   double  m_ss4o3 ;
   double  m_qstop ;             // excessive q indicator 
   double  m_monoq_max_slope ;
   double  m_monoq_limiter_mult ;
   double  m_qlc_monoq ;         // linear term coef for q 
   double  m_qqc_monoq ;         // quadratic term coef for q 
   double  m_qqc ;
   double  m_eosvmax ;
   double  m_eosvmin ;
   double  m_pmin ;              // pressure floor 
   double  m_emin ;              // energy floor 
   double  m_dvovmax ;           // maximum allowable volume change 
   double  m_refdens ;           // reference density 

   // Variables to keep track of timestep, simulation time, and cycle
   double  m_dtcourant ;         // courant constraint 
   double  m_dthydro ;           // volume change constraint 
   int   m_cycle ;             // iteration count for simulation 
   double  m_dtfixed ;           // fixed time increment 
   double  m_time ;              // current time 
   double  m_deltatime ;         // variable time increment 
   double  m_deltatimemultlb ;
   double  m_deltatimemultub ;
   double  m_dtmax ;             // maximum allowable time increment 
   double  m_stoptime ;          // end time for simulation 


   int   m_numRanks ;

   int m_colLoc ;
   int m_rowLoc ;
   int m_planeLoc ;
   int m_tp ;

   int m_sizeX ;
   int m_sizeY ;
   int m_sizeZ ;
   int m_numElem ;
   int m_numNode ;

   int m_maxPlaneSize ;
   int m_maxEdgeSize ;

   // OMP hack 
   int *m_nodeElemStart ;
   int *m_nodeElemCornerList ;

   // Used in setup
   int m_rowMin, m_rowMax;
   int m_colMin, m_colMax;
   int m_planeMin, m_planeMax ;

} ;

static inline
void TimeIncrement(Domain& domain)
{
   double targetdt = domain.stoptime() - domain.time() ;

   if ((domain.dtfixed() <= double(0.0)) && (domain.cycle() != int(0))) {
      double ratio ;
      double olddt = domain.deltatime() ;

      /* This will require a reduction in parallel */
      double gnewdt = double(1.0e+20) ;
      double newdt ;
      if (domain.dtcourant() < gnewdt) {
         gnewdt = domain.dtcourant() / double(2.0) ;
      }
      if (domain.dthydro() < gnewdt) {
         gnewdt = domain.dthydro() * double(2.0) / double(3.0) ;
      }

#if USE_MPI      
      MPI_Allreduce(&gnewdt, &newdt, 1,
                    ((sizeof(double) == 4) ? MPI_FLOAT : MPI_DOUBLE),
                    MPI_MIN, MPI_COMM_WORLD) ;
#else
      newdt = gnewdt;
#endif
      
      ratio = newdt / olddt ;
      if (ratio >= double(1.0)) {
         if (ratio < domain.deltatimemultlb()) {
            newdt = olddt ;
         }
         else if (ratio > domain.deltatimemultub()) {
            newdt = olddt*domain.deltatimemultub() ;
         }
      }

      if (newdt > domain.dtmax()) {
         newdt = domain.dtmax() ;
      }
      domain.deltatime() = newdt ;
   }

   /* TRY TO PREVENT VERY SMALL SCALING ON THE NEXT CYCLE */
   if ((targetdt > domain.deltatime()) &&
       (targetdt < (double(4.0) * domain.deltatime() / double(3.0))) ) {
      targetdt = double(2.0) * domain.deltatime() / double(3.0) ;
   }

   if (targetdt < domain.deltatime()) {
      domain.deltatime() = targetdt ;
   }

   domain.time() += domain.deltatime() ;

   ++domain.cycle() ;
}

/******************************************/

static inline
void CollectDomainNodesToElemNodes(Domain &domain,
                                   int* elemToNode,
                                   double elemX[8],
                                   double elemY[8],
                                   double elemZ[8])
{
   int nd0i = elemToNode[0] ;
   int nd1i = elemToNode[1] ;
   int nd2i = elemToNode[2] ;
   int nd3i = elemToNode[3] ;
   int nd4i = elemToNode[4] ;
   int nd5i = elemToNode[5] ;
   int nd6i = elemToNode[6] ;
   int nd7i = elemToNode[7] ;

   elemX[0] = domain.x(nd0i);
   elemX[1] = domain.x(nd1i);
   elemX[2] = domain.x(nd2i);
   elemX[3] = domain.x(nd3i);
   elemX[4] = domain.x(nd4i);
   elemX[5] = domain.x(nd5i);
   elemX[6] = domain.x(nd6i);
   elemX[7] = domain.x(nd7i);

   elemY[0] = domain.y(nd0i);
   elemY[1] = domain.y(nd1i);
   elemY[2] = domain.y(nd2i);
   elemY[3] = domain.y(nd3i);
   elemY[4] = domain.y(nd4i);
   elemY[5] = domain.y(nd5i);
   elemY[6] = domain.y(nd6i);
   elemY[7] = domain.y(nd7i);

   elemZ[0] = domain.z(nd0i);
   elemZ[1] = domain.z(nd1i);
   elemZ[2] = domain.z(nd2i);
   elemZ[3] = domain.z(nd3i);
   elemZ[4] = domain.z(nd4i);
   elemZ[5] = domain.z(nd5i);
   elemZ[6] = domain.z(nd6i);
   elemZ[7] = domain.z(nd7i);

}

/******************************************/

static inline
void InitStressTermsForElems(Domain &domain,
                             double *sigxx, double *sigyy, double *sigzz,
                             int numElem)
{
   //
   // pull in the stresses appropriate to the hydro integration
   //

#pragma omp parallel for firstprivate(numElem)
   for (int i = 0 ; i < numElem ; ++i){
      sigxx[i] = sigyy[i] = sigzz[i] =  - domain.p(i) - domain.q(i) ;
   }
}

/******************************************/

static inline
void CalcElemShapeFunctionDerivatives( double x[],
                                       double y[],
                                       double z[],
                                       double b[][8],
                                       double* volume )
{
  double x0 = x[0] ;   
  double x1 = x[1] ;
  double x2 = x[2] ;   
  double x3 = x[3] ;
  double x4 = x[4] ;   
  double x5 = x[5] ;
  double x6 = x[6] ;   
  double x7 = x[7] ;

  double y0 = y[0] ;   
  double y1 = y[1] ;
  double y2 = y[2] ;   
  double y3 = y[3] ;
  double y4 = y[4] ;   
  double y5 = y[5] ;
  double y6 = y[6] ;   
  double y7 = y[7] ;

  double z0 = z[0] ;   
  double z1 = z[1] ;
  double z2 = z[2] ;   
  double z3 = z[3] ;
  double z4 = z[4] ;   
  double z5 = z[5] ;
  double z6 = z[6] ;   
  double z7 = z[7] ;

  double fjxxi; 
  double fjxet; 
  double fjxze;
  double fjyxi; 
  double fjyet; 
  double fjyze;
  double fjzxi; 
  double fjzet; 
  double fjzze;
  double cjxxi; 
  double cjxet; 
  double cjxze;
  double cjyxi; 
  double cjyet; 
  double cjyze;
  double cjzxi; 
  double cjzet; 
  double cjzze;

  fjxxi = double(.125) * ( (x6-x0) + (x5-x3) - (x7-x1) - (x4-x2) );
  fjxet = double(.125) * ( (x6-x0) - (x5-x3) + (x7-x1) - (x4-x2) );
  fjxze = double(.125) * ( (x6-x0) + (x5-x3) + (x7-x1) + (x4-x2) );

  fjyxi = double(.125) * ( (y6-y0) + (y5-y3) - (y7-y1) - (y4-y2) );
  fjyet = double(.125) * ( (y6-y0) - (y5-y3) + (y7-y1) - (y4-y2) );
  fjyze = double(.125) * ( (y6-y0) + (y5-y3) + (y7-y1) + (y4-y2) );

  fjzxi = double(.125) * ( (z6-z0) + (z5-z3) - (z7-z1) - (z4-z2) );
  fjzet = double(.125) * ( (z6-z0) - (z5-z3) + (z7-z1) - (z4-z2) );
  fjzze = double(.125) * ( (z6-z0) + (z5-z3) + (z7-z1) + (z4-z2) );

  /* compute cofactors */
  cjxxi =    (fjyet * fjzze) - (fjzet * fjyze);
  cjxet =  - (fjyxi * fjzze) + (fjzxi * fjyze);
  cjxze =    (fjyxi * fjzet) - (fjzxi * fjyet);

  cjyxi =  - (fjxet * fjzze) + (fjzet * fjxze);
  cjyet =    (fjxxi * fjzze) - (fjzxi * fjxze);
  cjyze =  - (fjxxi * fjzet) + (fjzxi * fjxet);

  cjzxi =    (fjxet * fjyze) - (fjyet * fjxze);
  cjzet =  - (fjxxi * fjyze) + (fjyxi * fjxze);
  cjzze =    (fjxxi * fjyet) - (fjyxi * fjxet);

  /* calculate partials :
     this need only be done for l = 0,1,2,3   since , by symmetry ,
     (6,7,4,5) = - (0,1,2,3) .
  */
  b[0][0] =   -  cjxxi  -  cjxet  -  cjxze;
  b[0][1] =      cjxxi  -  cjxet  -  cjxze;
  b[0][2] =      cjxxi  +  cjxet  -  cjxze;
  b[0][3] =   -  cjxxi  +  cjxet  -  cjxze;
  b[0][4] = -b[0][2];
  b[0][5] = -b[0][3];
  b[0][6] = -b[0][0];
  b[0][7] = -b[0][1];

  b[1][0] =   -  cjyxi  -  cjyet  -  cjyze;
  b[1][1] =      cjyxi  -  cjyet  -  cjyze;
  b[1][2] =      cjyxi  +  cjyet  -  cjyze;
  b[1][3] =   -  cjyxi  +  cjyet  -  cjyze;
  b[1][4] = -b[1][2];
  b[1][5] = -b[1][3];
  b[1][6] = -b[1][0];
  b[1][7] = -b[1][1];

  b[2][0] =   -  cjzxi  -  cjzet  -  cjzze;
  b[2][1] =      cjzxi  -  cjzet  -  cjzze;
  b[2][2] =      cjzxi  +  cjzet  -  cjzze;
  b[2][3] =   -  cjzxi  +  cjzet  -  cjzze;
  b[2][4] = -b[2][2];
  b[2][5] = -b[2][3];
  b[2][6] = -b[2][0];
  b[2][7] = -b[2][1];

  /* calculate jacobian determinant (volume) */
  *volume = double(8.) * ( fjxet * cjxet + fjyet * cjyet + fjzet * cjzet);
}

/******************************************/

static inline
void SumElemFaceNormal(double *normalX0, double *normalY0, double *normalZ0,
                       double *normalX1, double *normalY1, double *normalZ1,
                       double *normalX2, double *normalY2, double *normalZ2,
                       double *normalX3, double *normalY3, double *normalZ3,
                       double x0, double y0, double z0,
                       double x1, double y1, double z1,
                       double x2, double y2, double z2,
                       double x3, double y3, double z3)
{
   double bisectX0 = double(0.5) * (x3 + x2 - x1 - x0);
   double bisectY0 = double(0.5) * (y3 + y2 - y1 - y0);
   double bisectZ0 = double(0.5) * (z3 + z2 - z1 - z0);
   double bisectX1 = double(0.5) * (x2 + x1 - x3 - x0);
   double bisectY1 = double(0.5) * (y2 + y1 - y3 - y0);
   double bisectZ1 = double(0.5) * (z2 + z1 - z3 - z0);
   double areaX = double(0.25) * (bisectY0 * bisectZ1 - bisectZ0 * bisectY1);
   double areaY = double(0.25) * (bisectZ0 * bisectX1 - bisectX0 * bisectZ1);
   double areaZ = double(0.25) * (bisectX0 * bisectY1 - bisectY0 * bisectX1);

   *normalX0 += areaX;
   *normalX1 += areaX;
   *normalX2 += areaX;
   *normalX3 += areaX;

   *normalY0 += areaY;
   *normalY1 += areaY;
   *normalY2 += areaY;
   *normalY3 += areaY;

   *normalZ0 += areaZ;
   *normalZ1 += areaZ;
   *normalZ2 += areaZ;
   *normalZ3 += areaZ;
}

/******************************************/

static inline
void CalcElemNodeNormals(double pfx[8],
                         double pfy[8],
                         double pfz[8],
                         double x[8],
                         double y[8],
                         double z[8])
{
   for (int i = 0 ; i < 8 ; ++i) {
      pfx[i] = double(0.0);
      pfy[i] = double(0.0);
      pfz[i] = double(0.0);
   }
   /* evaluate face one: nodes 0, 1, 2, 3 */
   SumElemFaceNormal(&pfx[0], &pfy[0], &pfz[0],
                  &pfx[1], &pfy[1], &pfz[1],
                  &pfx[2], &pfy[2], &pfz[2],
                  &pfx[3], &pfy[3], &pfz[3],
                  x[0], y[0], z[0], x[1], y[1], z[1],
                  x[2], y[2], z[2], x[3], y[3], z[3]);
   /* evaluate face two: nodes 0, 4, 5, 1 */
   SumElemFaceNormal(&pfx[0], &pfy[0], &pfz[0],
                  &pfx[4], &pfy[4], &pfz[4],
                  &pfx[5], &pfy[5], &pfz[5],
                  &pfx[1], &pfy[1], &pfz[1],
                  x[0], y[0], z[0], x[4], y[4], z[4],
                  x[5], y[5], z[5], x[1], y[1], z[1]);
   /* evaluate face three: nodes 1, 5, 6, 2 */
   SumElemFaceNormal(&pfx[1], &pfy[1], &pfz[1],
                  &pfx[5], &pfy[5], &pfz[5],
                  &pfx[6], &pfy[6], &pfz[6],
                  &pfx[2], &pfy[2], &pfz[2],
                  x[1], y[1], z[1], x[5], y[5], z[5],
                  x[6], y[6], z[6], x[2], y[2], z[2]);
   /* evaluate face four: nodes 2, 6, 7, 3 */
   SumElemFaceNormal(&pfx[2], &pfy[2], &pfz[2],
                  &pfx[6], &pfy[6], &pfz[6],
                  &pfx[7], &pfy[7], &pfz[7],
                  &pfx[3], &pfy[3], &pfz[3],
                  x[2], y[2], z[2], x[6], y[6], z[6],
                  x[7], y[7], z[7], x[3], y[3], z[3]);
   /* evaluate face five: nodes 3, 7, 4, 0 */
   SumElemFaceNormal(&pfx[3], &pfy[3], &pfz[3],
                  &pfx[7], &pfy[7], &pfz[7],
                  &pfx[4], &pfy[4], &pfz[4],
                  &pfx[0], &pfy[0], &pfz[0],
                  x[3], y[3], z[3], x[7], y[7], z[7],
                  x[4], y[4], z[4], x[0], y[0], z[0]);
   /* evaluate face six: nodes 4, 7, 6, 5 */
   SumElemFaceNormal(&pfx[4], &pfy[4], &pfz[4],
                  &pfx[7], &pfy[7], &pfz[7],
                  &pfx[6], &pfy[6], &pfz[6],
                  &pfx[5], &pfy[5], &pfz[5],
                  x[4], y[4], z[4], x[7], y[7], z[7],
                  x[6], y[6], z[6], x[5], y[5], z[5]);
}

/******************************************/

static inline
void SumElemStressesToNodeForces( double B[][8],
                                  double stress_xx,
                                  double stress_yy,
                                  double stress_zz,
                                  double fx[], double fy[], double fz[] )
{
   for(int i = 0; i < 8; i++) {
      fx[i] = -( stress_xx * B[0][i] );
      fy[i] = -( stress_yy * B[1][i]  );
      fz[i] = -( stress_zz * B[2][i] );
   }
}

/******************************************/

static inline
void IntegrateStressForElems( Domain &domain,
                              double *sigxx, double *sigyy, double *sigzz,
                              double *determ, int numElem, int numNode)
{
#if _OPENMP
   int numthreads = omp_get_max_threads();
#else
   int numthreads = 1;
#endif

   int numElem8 = numElem * 8 ;
   double *fx_elem;
   double *fy_elem;
   double *fz_elem;
   double fx_local[8] ;
   double fy_local[8] ;
   double fz_local[8] ;


  if (numthreads > 1) {
     fx_elem = Allocate<double>(numElem8) ;
     fy_elem = Allocate<double>(numElem8) ;
     fz_elem = Allocate<double>(numElem8) ;
  }
  // loop over all elements

#pragma omp parallel for firstprivate(numElem)
  for( int k=0 ; k<numElem ; ++k )
  {
    int* elemToNode = domain.nodelist(k);
    double B[3][8] ;// shape function derivatives
    double x_local[8] ;
    double y_local[8] ;
    double z_local[8] ;

    // get nodal coordinates from global arrays and copy into local arrays.
    CollectDomainNodesToElemNodes(domain, elemToNode, x_local, y_local, z_local);

    // Volume calculation involves extra work for numerical consistency
    CalcElemShapeFunctionDerivatives(x_local, y_local, z_local,
                                         B, &determ[k]);

    CalcElemNodeNormals( B[0] , B[1], B[2],
                          x_local, y_local, z_local );

    if (numthreads > 1) {
       // Eliminate thread writing conflicts at the nodes by giving
       // each element its own copy to write to
       SumElemStressesToNodeForces( B, sigxx[k], sigyy[k], sigzz[k],
                                    &fx_elem[k*8],
                                    &fy_elem[k*8],
                                    &fz_elem[k*8] ) ;
    }
    else {
       SumElemStressesToNodeForces( B, sigxx[k], sigyy[k], sigzz[k],
                                    fx_local, fy_local, fz_local ) ;

       // copy nodal force contributions to global force arrray.
       for( int lnode=0 ; lnode<8 ; ++lnode ) {
          int gnode = elemToNode[lnode];
          domain.fx(gnode) += fx_local[lnode];
          domain.fy(gnode) += fy_local[lnode];
          domain.fz(gnode) += fz_local[lnode];
       }
    }
  }

  if (numthreads > 1) {
     // If threaded, then we need to copy the data out of the temporary
     // arrays used above into the final forces field
#pragma omp parallel for firstprivate(numNode)
     for( int gnode=0 ; gnode<numNode ; ++gnode )
     {
        int count = domain.nodeElemCount(gnode) ;
        int *cornerList = domain.nodeElemCornerList(gnode) ;
        double fx_tmp = double(0.0) ;
        double fy_tmp = double(0.0) ;
        double fz_tmp = double(0.0) ;
        for (int i=0 ; i < count ; ++i) {
           int ielem = cornerList[i] ;
           fx_tmp += fx_elem[ielem] ;
           fy_tmp += fy_elem[ielem] ;
           fz_tmp += fz_elem[ielem] ;
        }
        domain.fx(gnode) = fx_tmp ;
        domain.fy(gnode) = fy_tmp ;
        domain.fz(gnode) = fz_tmp ;
     }
     Release(&fz_elem) ;
     Release(&fy_elem) ;
     Release(&fx_elem) ;
  }
}

/******************************************/

static inline
void VoluDer(double x0, double x1, double x2,
             double x3, double x4, double x5,
             double y0, double y1, double y2,
             double y3, double y4, double y5,
             double z0, double z1, double z2,
             double z3, double z4, double z5,
             double* dvdx, double* dvdy, double* dvdz)
{
   double twelfth = double(1.0) / double(12.0) ;

   *dvdx =
      (y1 + y2) * (z0 + z1) - (y0 + y1) * (z1 + z2) +
      (y0 + y4) * (z3 + z4) - (y3 + y4) * (z0 + z4) -
      (y2 + y5) * (z3 + z5) + (y3 + y5) * (z2 + z5);
   *dvdy =
      - (x1 + x2) * (z0 + z1) + (x0 + x1) * (z1 + z2) -
      (x0 + x4) * (z3 + z4) + (x3 + x4) * (z0 + z4) +
      (x2 + x5) * (z3 + z5) - (x3 + x5) * (z2 + z5);

   *dvdz =
      - (y1 + y2) * (x0 + x1) + (y0 + y1) * (x1 + x2) -
      (y0 + y4) * (x3 + x4) + (y3 + y4) * (x0 + x4) +
      (y2 + y5) * (x3 + x5) - (y3 + y5) * (x2 + x5);

   *dvdx *= twelfth;
   *dvdy *= twelfth;
   *dvdz *= twelfth;
}

/******************************************/

static inline
void CalcElemVolumeDerivative(double dvdx[8],
                              double dvdy[8],
                              double dvdz[8],
                              double x[8],
                              double y[8],
                              double z[8])
{
   VoluDer(x[1], x[2], x[3], x[4], x[5], x[7],
           y[1], y[2], y[3], y[4], y[5], y[7],
           z[1], z[2], z[3], z[4], z[5], z[7],
           &dvdx[0], &dvdy[0], &dvdz[0]);
   VoluDer(x[0], x[1], x[2], x[7], x[4], x[6],
           y[0], y[1], y[2], y[7], y[4], y[6],
           z[0], z[1], z[2], z[7], z[4], z[6],
           &dvdx[3], &dvdy[3], &dvdz[3]);
   VoluDer(x[3], x[0], x[1], x[6], x[7], x[5],
           y[3], y[0], y[1], y[6], y[7], y[5],
           z[3], z[0], z[1], z[6], z[7], z[5],
           &dvdx[2], &dvdy[2], &dvdz[2]);
   VoluDer(x[2], x[3], x[0], x[5], x[6], x[4],
           y[2], y[3], y[0], y[5], y[6], y[4],
           z[2], z[3], z[0], z[5], z[6], z[4],
           &dvdx[1], &dvdy[1], &dvdz[1]);
   VoluDer(x[7], x[6], x[5], x[0], x[3], x[1],
           y[7], y[6], y[5], y[0], y[3], y[1],
           z[7], z[6], z[5], z[0], z[3], z[1],
           &dvdx[4], &dvdy[4], &dvdz[4]);
   VoluDer(x[4], x[7], x[6], x[1], x[0], x[2],
           y[4], y[7], y[6], y[1], y[0], y[2],
           z[4], z[7], z[6], z[1], z[0], z[2],
           &dvdx[5], &dvdy[5], &dvdz[5]);
   VoluDer(x[5], x[4], x[7], x[2], x[1], x[3],
           y[5], y[4], y[7], y[2], y[1], y[3],
           z[5], z[4], z[7], z[2], z[1], z[3],
           &dvdx[6], &dvdy[6], &dvdz[6]);
   VoluDer(x[6], x[5], x[4], x[3], x[2], x[0],
           y[6], y[5], y[4], y[3], y[2], y[0],
           z[6], z[5], z[4], z[3], z[2], z[0],
           &dvdx[7], &dvdy[7], &dvdz[7]);
}

/******************************************/

static inline
void CalcElemFBHourglassForce(double *xd, double *yd, double *zd,  double hourgam[][4],
                              double coefficient,
                              double *hgfx, double *hgfy, double *hgfz )
{
   double hxx[4];
   for(int i = 0; i < 4; i++) {
      hxx[i] = hourgam[0][i] * xd[0] + hourgam[1][i] * xd[1] +
               hourgam[2][i] * xd[2] + hourgam[3][i] * xd[3] +
               hourgam[4][i] * xd[4] + hourgam[5][i] * xd[5] +
               hourgam[6][i] * xd[6] + hourgam[7][i] * xd[7];
   }
   for(int i = 0; i < 8; i++) {
      hgfx[i] = coefficient *
                (hourgam[i][0] * hxx[0] + hourgam[i][1] * hxx[1] +
                 hourgam[i][2] * hxx[2] + hourgam[i][3] * hxx[3]);
   }
   for(int i = 0; i < 4; i++) {
      hxx[i] = hourgam[0][i] * yd[0] + hourgam[1][i] * yd[1] +
               hourgam[2][i] * yd[2] + hourgam[3][i] * yd[3] +
               hourgam[4][i] * yd[4] + hourgam[5][i] * yd[5] +
               hourgam[6][i] * yd[6] + hourgam[7][i] * yd[7];
   }
   for(int i = 0; i < 8; i++) {
      hgfy[i] = coefficient *
                (hourgam[i][0] * hxx[0] + hourgam[i][1] * hxx[1] +
                 hourgam[i][2] * hxx[2] + hourgam[i][3] * hxx[3]);
   }
   for(int i = 0; i < 4; i++) {
      hxx[i] = hourgam[0][i] * zd[0] + hourgam[1][i] * zd[1] +
               hourgam[2][i] * zd[2] + hourgam[3][i] * zd[3] +
               hourgam[4][i] * zd[4] + hourgam[5][i] * zd[5] +
               hourgam[6][i] * zd[6] + hourgam[7][i] * zd[7];
   }
   for(int i = 0; i < 8; i++) {
      hgfz[i] = coefficient *
                (hourgam[i][0] * hxx[0] + hourgam[i][1] * hxx[1] +
                 hourgam[i][2] * hxx[2] + hourgam[i][3] * hxx[3]);
   }
}

/******************************************/

static inline
void CalcFBHourglassForceForElems( Domain &domain,
                                   double *determ,
                                   double *x8n, double *y8n, double *z8n,
                                   double *dvdx, double *dvdy, double *dvdz,
                                   double hourg, int numElem,
                                   int numNode)
{

#if _OPENMP
   int numthreads = omp_get_max_threads();
#else
   int numthreads = 1;
#endif
   /*************************************************
    *
    *     FUNCTION: Calculates the Flanagan-Belytschko anti-hourglass
    *               force.
    *
    *************************************************/
  
   int numElem8 = numElem * 8 ;

   double *fx_elem; 
   double *fy_elem; 
   double *fz_elem; 

   if(numthreads > 1) {
      fx_elem = Allocate<double>(numElem8) ;
      fy_elem = Allocate<double>(numElem8) ;
      fz_elem = Allocate<double>(numElem8) ;
   }

   double  gamma[4][8];

   gamma[0][0] = double( 1.);
   gamma[0][1] = double( 1.);
   gamma[0][2] = double(-1.);
   gamma[0][3] = double(-1.);
   gamma[0][4] = double(-1.);
   gamma[0][5] = double(-1.);
   gamma[0][6] = double( 1.);
   gamma[0][7] = double( 1.);
   gamma[1][0] = double( 1.);
   gamma[1][1] = double(-1.);
   gamma[1][2] = double(-1.);
   gamma[1][3] = double( 1.);
   gamma[1][4] = double(-1.);
   gamma[1][5] = double( 1.);
   gamma[1][6] = double( 1.);
   gamma[1][7] = double(-1.);
   gamma[2][0] = double( 1.);
   gamma[2][1] = double(-1.);
   gamma[2][2] = double( 1.);
   gamma[2][3] = double(-1.);
   gamma[2][4] = double( 1.);
   gamma[2][5] = double(-1.);
   gamma[2][6] = double( 1.);
   gamma[2][7] = double(-1.);
   gamma[3][0] = double(-1.);
   gamma[3][1] = double( 1.);
   gamma[3][2] = double(-1.);
   gamma[3][3] = double( 1.);
   gamma[3][4] = double( 1.);
   gamma[3][5] = double(-1.);
   gamma[3][6] = double( 1.);
   gamma[3][7] = double(-1.);

/*************************************************/
/*    compute the hourglass modes */


#pragma omp parallel for firstprivate(numElem, hourg)
   for(int i2=0;i2<numElem;++i2){
      double *fx_local;
      double *fy_local;
      double *fz_local;
      double hgfx[8];
      double hgfy[8];
      double hgfz[8] ;

      double coefficient;

      double hourgam[8][4];
      double xd1[8];
      double yd1[8];
      double zd1[8];

      int *elemToNode = domain.nodelist(i2);
      int i3=8*i2;
      double volinv=double(1.0)/determ[i2];
      double ss1;
      double mass1;
      double volume13;
      for(int i1=0;i1<4;++i1){

         double hourmodx =
            x8n[i3] * gamma[i1][0] + x8n[i3+1] * gamma[i1][1] +
            x8n[i3+2] * gamma[i1][2] + x8n[i3+3] * gamma[i1][3] +
            x8n[i3+4] * gamma[i1][4] + x8n[i3+5] * gamma[i1][5] +
            x8n[i3+6] * gamma[i1][6] + x8n[i3+7] * gamma[i1][7];

         double hourmody =
            y8n[i3] * gamma[i1][0] + y8n[i3+1] * gamma[i1][1] +
            y8n[i3+2] * gamma[i1][2] + y8n[i3+3] * gamma[i1][3] +
            y8n[i3+4] * gamma[i1][4] + y8n[i3+5] * gamma[i1][5] +
            y8n[i3+6] * gamma[i1][6] + y8n[i3+7] * gamma[i1][7];

         double hourmodz =
            z8n[i3] * gamma[i1][0] + z8n[i3+1] * gamma[i1][1] +
            z8n[i3+2] * gamma[i1][2] + z8n[i3+3] * gamma[i1][3] +
            z8n[i3+4] * gamma[i1][4] + z8n[i3+5] * gamma[i1][5] +
            z8n[i3+6] * gamma[i1][6] + z8n[i3+7] * gamma[i1][7];

         hourgam[0][i1] = gamma[i1][0] -  volinv*(dvdx[i3  ] * hourmodx +
                                                  dvdy[i3  ] * hourmody +
                                                  dvdz[i3  ] * hourmodz );

         hourgam[1][i1] = gamma[i1][1] -  volinv*(dvdx[i3+1] * hourmodx +
                                                  dvdy[i3+1] * hourmody +
                                                  dvdz[i3+1] * hourmodz );

         hourgam[2][i1] = gamma[i1][2] -  volinv*(dvdx[i3+2] * hourmodx +
                                                  dvdy[i3+2] * hourmody +
                                                  dvdz[i3+2] * hourmodz );

         hourgam[3][i1] = gamma[i1][3] -  volinv*(dvdx[i3+3] * hourmodx +
                                                  dvdy[i3+3] * hourmody +
                                                  dvdz[i3+3] * hourmodz );

         hourgam[4][i1] = gamma[i1][4] -  volinv*(dvdx[i3+4] * hourmodx +
                                                  dvdy[i3+4] * hourmody +
                                                  dvdz[i3+4] * hourmodz );

         hourgam[5][i1] = gamma[i1][5] -  volinv*(dvdx[i3+5] * hourmodx +
                                                  dvdy[i3+5] * hourmody +
                                                  dvdz[i3+5] * hourmodz );

         hourgam[6][i1] = gamma[i1][6] -  volinv*(dvdx[i3+6] * hourmodx +
                                                  dvdy[i3+6] * hourmody +
                                                  dvdz[i3+6] * hourmodz );

         hourgam[7][i1] = gamma[i1][7] -  volinv*(dvdx[i3+7] * hourmodx +
                                                  dvdy[i3+7] * hourmody +
                                                  dvdz[i3+7] * hourmodz );

      }

      /* compute forces */
      /* store forces into h arrays (force arrays) */

      ss1=domain.ss(i2);
      mass1=domain.elemMass(i2);
      volume13=CBRT(determ[i2]);

      int n0si2 = elemToNode[0];
      int n1si2 = elemToNode[1];
      int n2si2 = elemToNode[2];
      int n3si2 = elemToNode[3];
      int n4si2 = elemToNode[4];
      int n5si2 = elemToNode[5];
      int n6si2 = elemToNode[6];
      int n7si2 = elemToNode[7];

      xd1[0] = domain.xd(n0si2);
      xd1[1] = domain.xd(n1si2);
      xd1[2] = domain.xd(n2si2);
      xd1[3] = domain.xd(n3si2);
      xd1[4] = domain.xd(n4si2);
      xd1[5] = domain.xd(n5si2);
      xd1[6] = domain.xd(n6si2);
      xd1[7] = domain.xd(n7si2);

      yd1[0] = domain.yd(n0si2);
      yd1[1] = domain.yd(n1si2);
      yd1[2] = domain.yd(n2si2);
      yd1[3] = domain.yd(n3si2);
      yd1[4] = domain.yd(n4si2);
      yd1[5] = domain.yd(n5si2);
      yd1[6] = domain.yd(n6si2);
      yd1[7] = domain.yd(n7si2);

      zd1[0] = domain.zd(n0si2);
      zd1[1] = domain.zd(n1si2);
      zd1[2] = domain.zd(n2si2);
      zd1[3] = domain.zd(n3si2);
      zd1[4] = domain.zd(n4si2);
      zd1[5] = domain.zd(n5si2);
      zd1[6] = domain.zd(n6si2);
      zd1[7] = domain.zd(n7si2);

      coefficient = - hourg * double(0.01) * ss1 * mass1 / volume13;

      CalcElemFBHourglassForce(xd1,yd1,zd1,
                      hourgam,
                      coefficient, hgfx, hgfy, hgfz);

      // With the threaded version, we write into local arrays per elem
      // so we don't have to worry about race conditions
      if (numthreads > 1) {
         fx_local = &fx_elem[i3] ;
         fx_local[0] = hgfx[0];
         fx_local[1] = hgfx[1];
         fx_local[2] = hgfx[2];
         fx_local[3] = hgfx[3];
         fx_local[4] = hgfx[4];
         fx_local[5] = hgfx[5];
         fx_local[6] = hgfx[6];
         fx_local[7] = hgfx[7];

         fy_local = &fy_elem[i3] ;
         fy_local[0] = hgfy[0];
         fy_local[1] = hgfy[1];
         fy_local[2] = hgfy[2];
         fy_local[3] = hgfy[3];
         fy_local[4] = hgfy[4];
         fy_local[5] = hgfy[5];
         fy_local[6] = hgfy[6];
         fy_local[7] = hgfy[7];

         fz_local = &fz_elem[i3] ;
         fz_local[0] = hgfz[0];
         fz_local[1] = hgfz[1];
         fz_local[2] = hgfz[2];
         fz_local[3] = hgfz[3];
         fz_local[4] = hgfz[4];
         fz_local[5] = hgfz[5];
         fz_local[6] = hgfz[6];
         fz_local[7] = hgfz[7];
      }
      else {
         domain.fx(n0si2) += hgfx[0];
         domain.fy(n0si2) += hgfy[0];
         domain.fz(n0si2) += hgfz[0];

         domain.fx(n1si2) += hgfx[1];
         domain.fy(n1si2) += hgfy[1];
         domain.fz(n1si2) += hgfz[1];

         domain.fx(n2si2) += hgfx[2];
         domain.fy(n2si2) += hgfy[2];
         domain.fz(n2si2) += hgfz[2];

         domain.fx(n3si2) += hgfx[3];
         domain.fy(n3si2) += hgfy[3];
         domain.fz(n3si2) += hgfz[3];

         domain.fx(n4si2) += hgfx[4];
         domain.fy(n4si2) += hgfy[4];
         domain.fz(n4si2) += hgfz[4];

         domain.fx(n5si2) += hgfx[5];
         domain.fy(n5si2) += hgfy[5];
         domain.fz(n5si2) += hgfz[5];

         domain.fx(n6si2) += hgfx[6];
         domain.fy(n6si2) += hgfy[6];
         domain.fz(n6si2) += hgfz[6];

         domain.fx(n7si2) += hgfx[7];
         domain.fy(n7si2) += hgfy[7];
         domain.fz(n7si2) += hgfz[7];
      }
   }

   if (numthreads > 1) {
     // Collect the data from the local arrays into the final force arrays
#pragma omp parallel for firstprivate(numNode)
      for( int gnode=0 ; gnode<numNode ; ++gnode )
      {
         int count = domain.nodeElemCount(gnode) ;
         int *cornerList = domain.nodeElemCornerList(gnode) ;
         double fx_tmp = double(0.0) ;
         double fy_tmp = double(0.0) ;
         double fz_tmp = double(0.0) ;
         for (int i=0 ; i < count ; ++i) {
            int ielem = cornerList[i] ;
            fx_tmp += fx_elem[ielem] ;
            fy_tmp += fy_elem[ielem] ;
            fz_tmp += fz_elem[ielem] ;
         }
         domain.fx(gnode) += fx_tmp ;
         domain.fy(gnode) += fy_tmp ;
         domain.fz(gnode) += fz_tmp ;
      }
      Release(&fz_elem) ;
      Release(&fy_elem) ;
      Release(&fx_elem) ;
   }
}

/******************************************/

static inline
void CalcHourglassControlForElems(Domain& domain,
                                  double determ[], double hgcoef)
{
   int numElem = domain.numElem() ;
   int numElem8 = numElem * 8 ;
   double *dvdx = Allocate<double>(numElem8) ;
   double *dvdy = Allocate<double>(numElem8) ;
   double *dvdz = Allocate<double>(numElem8) ;
   double *x8n  = Allocate<double>(numElem8) ;
   double *y8n  = Allocate<double>(numElem8) ;
   double *z8n  = Allocate<double>(numElem8) ;

   /* start loop over elements */
#pragma omp parallel for firstprivate(numElem)
   for (int i=0 ; i<numElem ; ++i){
      double  x1[8],  y1[8],  z1[8] ;
      double pfx[8], pfy[8], pfz[8] ;

      int* elemToNode = domain.nodelist(i);
      CollectDomainNodesToElemNodes(domain, elemToNode, x1, y1, z1);

      CalcElemVolumeDerivative(pfx, pfy, pfz, x1, y1, z1);

      /* load into temporary storage for FB Hour Glass control */
      for(int ii=0;ii<8;++ii){
         int jj=8*i+ii;

         dvdx[jj] = pfx[ii];
         dvdy[jj] = pfy[ii];
         dvdz[jj] = pfz[ii];
         x8n[jj]  = x1[ii];
         y8n[jj]  = y1[ii];
         z8n[jj]  = z1[ii];
      }

      determ[i] = domain.volo(i) * domain.v(i);

      /* Do a check for negative volumes */
      if ( domain.v(i) <= double(0.0) ) {
#if USE_MPI         
         MPI_Abort(MPI_COMM_WORLD, VolumeError) ;
#else
         exit(VolumeError);
#endif
      }
   }

   if ( hgcoef > double(0.) ) {
      CalcFBHourglassForceForElems( domain,
                                    determ, x8n, y8n, z8n, dvdx, dvdy, dvdz,
                                    hgcoef, numElem, domain.numNode()) ;
   }

   Release(&z8n) ;
   Release(&y8n) ;
   Release(&x8n) ;
   Release(&dvdz) ;
   Release(&dvdy) ;
   Release(&dvdx) ;

   return ;
}

/******************************************/

static inline
void CalcVolumeForceForElems(Domain& domain)
{
   int numElem = domain.numElem() ;
   if (numElem != 0) {
      double  hgcoef = domain.hgcoef() ;
      double *sigxx  = Allocate<double>(numElem) ;
      double *sigyy  = Allocate<double>(numElem) ;
      double *sigzz  = Allocate<double>(numElem) ;
      double *determ = Allocate<double>(numElem) ;

      /* Sum contributions to total stress tensor */
      InitStressTermsForElems(domain, sigxx, sigyy, sigzz, numElem);

      // call elemlib stress integration loop to produce nodal forces from
      // material stresses.
      IntegrateStressForElems( domain,
                               sigxx, sigyy, sigzz, determ, numElem,
                               domain.numNode()) ;

      // check for negative element volume
#pragma omp parallel for firstprivate(numElem)
      for ( int k=0 ; k<numElem ; ++k ) {
         if (determ[k] <= double(0.0)) {
#if USE_MPI            
            MPI_Abort(MPI_COMM_WORLD, VolumeError) ;
#else
            exit(VolumeError);
#endif
         }
      }

      CalcHourglassControlForElems(domain, determ, hgcoef) ;

      Release(&determ) ;
      Release(&sigzz) ;
      Release(&sigyy) ;
      Release(&sigxx) ;
   }
}

/******************************************/

static inline void CalcForceForNodes(Domain& domain)
{
  int numNode = domain.numNode() ;

#if USE_MPI  
  CommRecv(domain, MSG_COMM_SBN, 3,
           domain.sizeX() + 1, domain.sizeY() + 1, domain.sizeZ() + 1,
           true, false) ;
#endif  

#pragma omp parallel for firstprivate(numNode)
  for (int i=0; i<numNode; ++i) {
     domain.fx(i) = double(0.0) ;
     domain.fy(i) = double(0.0) ;
     domain.fz(i) = double(0.0) ;
  }

  /* Calcforce calls partial, force, hourq */
  CalcVolumeForceForElems(domain) ;

#if USE_MPI  
  Domain_member fieldData[3] ;
  fieldData[0] = &Domain::fx ;
  fieldData[1] = &Domain::fy ;
  fieldData[2] = &Domain::fz ;
  
  CommSend(domain, MSG_COMM_SBN, 3, fieldData,
           domain.sizeX() + 1, domain.sizeY() + 1, domain.sizeZ() +  1,
           true, false) ;
  CommSBN(domain, 3, fieldData) ;
#endif  
}

/******************************************/

static inline
void CalcAccelerationForNodes(Domain &domain, int numNode)
{
   
#pragma omp parallel for firstprivate(numNode)
   for (int i = 0; i < numNode; ++i) {
      domain.xdd(i) = domain.fx(i) / domain.nodalMass(i);
      domain.ydd(i) = domain.fy(i) / domain.nodalMass(i);
      domain.zdd(i) = domain.fz(i) / domain.nodalMass(i);
   }
}

/******************************************/

static inline
void ApplyAccelerationBoundaryConditionsForNodes(Domain& domain)
{
   int size = domain.sizeX();
   int numNodeBC = (size+1)*(size+1) ;

#pragma omp parallel
   {
      if (!domain.symmXempty() != 0) {
#pragma omp for nowait firstprivate(numNodeBC)
         for(int i=0 ; i<numNodeBC ; ++i)
            domain.xdd(domain.symmX(i)) = double(0.0) ;
      }

      if (!domain.symmYempty() != 0) {
#pragma omp for nowait firstprivate(numNodeBC)
         for(int i=0 ; i<numNodeBC ; ++i)
            domain.ydd(domain.symmY(i)) = double(0.0) ;
      }

      if (!domain.symmZempty() != 0) {
#pragma omp for nowait firstprivate(numNodeBC)
         for(int i=0 ; i<numNodeBC ; ++i)
            domain.zdd(domain.symmZ(i)) = double(0.0) ;
      }
   }
}

/******************************************/

static inline
void CalcVelocityForNodes(Domain &domain, double dt, double u_cut,
                          int numNode)
{

#pragma omp parallel for firstprivate(numNode)
   for ( int i = 0 ; i < numNode ; ++i )
   {
     double xdtmp;
     double ydtmp;
     double zdtmp;

     xdtmp = domain.xd(i) + domain.xdd(i) * dt ;
     if( FABS(xdtmp) < u_cut ) xdtmp = double(0.0);
     domain.xd(i) = xdtmp ;

     ydtmp = domain.yd(i) + domain.ydd(i) * dt ;
     if( FABS(ydtmp) < u_cut ) ydtmp = double(0.0);
     domain.yd(i) = ydtmp ;

     zdtmp = domain.zd(i) + domain.zdd(i) * dt ;
     if( FABS(zdtmp) < u_cut ) zdtmp = double(0.0);
     domain.zd(i) = zdtmp ;
   }
}

/******************************************/

static inline
void CalcPositionForNodes(Domain &domain, double dt, int numNode)
{
#pragma omp parallel for firstprivate(numNode)
   for ( int i = 0 ; i < numNode ; ++i )
   {
     domain.x(i) += domain.xd(i) * dt ;
     domain.y(i) += domain.yd(i) * dt ;
     domain.z(i) += domain.zd(i) * dt ;
   }
}

/******************************************/

static inline
void LagrangeNodal(Domain& domain)
{
#ifdef SEDOV_SYNC_POS_VEL_EARLY
   Domain_member fieldData[6] ;
#endif

   double delt = domain.deltatime() ;
   double u_cut = domain.u_cut() ;

  /* time of boundary condition evaluation is beginning of step for force and
   * acceleration boundary conditions. */
  CalcForceForNodes(domain);

#if USE_MPI  
#ifdef SEDOV_SYNC_POS_VEL_EARLY
   CommRecv(domain, MSG_SYNC_POS_VEL, 6,
            domain.sizeX() + 1, domain.sizeY() + 1, domain.sizeZ() + 1,
            false, false) ;
#endif
#endif
   
   CalcAccelerationForNodes(domain, domain.numNode());
   
   ApplyAccelerationBoundaryConditionsForNodes(domain);

   CalcVelocityForNodes( domain, delt, u_cut, domain.numNode()) ;

   CalcPositionForNodes( domain, delt, domain.numNode() );
#if USE_MPI
#ifdef SEDOV_SYNC_POS_VEL_EARLY
  fieldData[0] = &Domain::x ;
  fieldData[1] = &Domain::y ;
  fieldData[2] = &Domain::z ;
  fieldData[3] = &Domain::xd ;
  fieldData[4] = &Domain::yd ;
  fieldData[5] = &Domain::zd ;

   CommSend(domain, MSG_SYNC_POS_VEL, 6, fieldData,
            domain.sizeX() + 1, domain.sizeY() + 1, domain.sizeZ() + 1,
            false, false) ;
   CommSyncPosVel(domain) ;
#endif
#endif
   
  return;
}

/******************************************/

static inline
double CalcElemVolume( double x0, double x1,
               double x2, double x3,
               double x4, double x5,
               double x6, double x7,
               double y0, double y1,
               double y2, double y3,
               double y4, double y5,
               double y6, double y7,
               double z0, double z1,
               double z2, double z3,
               double z4, double z5,
               double z6, double z7 )
{
  double twelveth = double(1.0)/double(12.0);

  double dx61 = x6 - x1;
  double dy61 = y6 - y1;
  double dz61 = z6 - z1;

  double dx70 = x7 - x0;
  double dy70 = y7 - y0;
  double dz70 = z7 - z0;

  double dx63 = x6 - x3;
  double dy63 = y6 - y3;
  double dz63 = z6 - z3;

  double dx20 = x2 - x0;
  double dy20 = y2 - y0;
  double dz20 = z2 - z0;

  double dx50 = x5 - x0;
  double dy50 = y5 - y0;
  double dz50 = z5 - z0;

  double dx64 = x6 - x4;
  double dy64 = y6 - y4;
  double dz64 = z6 - z4;

  double dx31 = x3 - x1;
  double dy31 = y3 - y1;
  double dz31 = z3 - z1;

  double dx72 = x7 - x2;
  double dy72 = y7 - y2;
  double dz72 = z7 - z2;

  double dx43 = x4 - x3;
  double dy43 = y4 - y3;
  double dz43 = z4 - z3;

  double dx57 = x5 - x7;
  double dy57 = y5 - y7;
  double dz57 = z5 - z7;

  double dx14 = x1 - x4;
  double dy14 = y1 - y4;
  double dz14 = z1 - z4;

  double dx25 = x2 - x5;
  double dy25 = y2 - y5;
  double dz25 = z2 - z5;

#define TRIPLE_PRODUCT(x1, y1, z1, x2, y2, z2, x3, y3, z3) \
   ((x1)*((y2)*(z3) - (z2)*(y3)) + (x2)*((z1)*(y3) - (y1)*(z3)) + (x3)*((y1)*(z2) - (z1)*(y2)))

  double volume =
    TRIPLE_PRODUCT(dx31 + dx72, dx63, dx20,
       dy31 + dy72, dy63, dy20,
       dz31 + dz72, dz63, dz20) +
    TRIPLE_PRODUCT(dx43 + dx57, dx64, dx70,
       dy43 + dy57, dy64, dy70,
       dz43 + dz57, dz64, dz70) +
    TRIPLE_PRODUCT(dx14 + dx25, dx61, dx50,
       dy14 + dy25, dy61, dy50,
       dz14 + dz25, dz61, dz50);

#undef TRIPLE_PRODUCT

  volume *= twelveth;

  return volume ;
}

/******************************************/

//inline
double CalcElemVolume( double x[8], double y[8], double z[8] )
{
return CalcElemVolume( x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7],
                       y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7],
                       z[0], z[1], z[2], z[3], z[4], z[5], z[6], z[7]);
}

/******************************************/

static inline
double AreaFace( double x0, double x1,
                 double x2, double x3,
                 double y0, double y1,
                 double y2, double y3,
                 double z0, double z1,
                 double z2, double z3)
{
   double fx = (x2 - x0) - (x3 - x1);
   double fy = (y2 - y0) - (y3 - y1);
   double fz = (z2 - z0) - (z3 - z1);
   double gx = (x2 - x0) + (x3 - x1);
   double gy = (y2 - y0) + (y3 - y1);
   double gz = (z2 - z0) + (z3 - z1);
   double area =
      (fx * fx + fy * fy + fz * fz) *
      (gx * gx + gy * gy + gz * gz) -
      (fx * gx + fy * gy + fz * gz) *
      (fx * gx + fy * gy + fz * gz);
   return area ;
}

/******************************************/

static inline
double CalcElemCharacteristicLength( double x[8],
                                     double y[8],
                                     double z[8],
                                     double volume)
{
   double a = double(0.0);
   double charLength = double(0.0);

   a = AreaFace(x[0],x[1],x[2],x[3],
                y[0],y[1],y[2],y[3],
                z[0],z[1],z[2],z[3]) ;
   charLength = std::max(a,charLength) ;

   a = AreaFace(x[4],x[5],x[6],x[7],
                y[4],y[5],y[6],y[7],
                z[4],z[5],z[6],z[7]) ;
   charLength = std::max(a,charLength) ;

   a = AreaFace(x[0],x[1],x[5],x[4],
                y[0],y[1],y[5],y[4],
                z[0],z[1],z[5],z[4]) ;
   charLength = std::max(a,charLength) ;

   a = AreaFace(x[1],x[2],x[6],x[5],
                y[1],y[2],y[6],y[5],
                z[1],z[2],z[6],z[5]) ;
   charLength = std::max(a,charLength) ;

   a = AreaFace(x[2],x[3],x[7],x[6],
                y[2],y[3],y[7],y[6],
                z[2],z[3],z[7],z[6]) ;
   charLength = std::max(a,charLength) ;

   a = AreaFace(x[3],x[0],x[4],x[7],
                y[3],y[0],y[4],y[7],
                z[3],z[0],z[4],z[7]) ;
   charLength = std::max(a,charLength) ;

   charLength = double(4.0) * volume / SQRT(charLength);

   return charLength;
}

/******************************************/

static inline
void CalcElemVelocityGradient( double* xvel,
                                double* yvel,
                                double* zvel,
                                double b[][8],
                                double detJ,
                                double* d )
{
  double inv_detJ = double(1.0) / detJ ;
  double dyddx; 
  double dxddy; 
  double dzddx; 
  double dxddz; 
  double dzddy; 
  double dyddz;
  double* pfx = b[0];
  double* pfy = b[1];
  double* pfz = b[2];

  d[0] = inv_detJ * ( pfx[0] * (xvel[0]-xvel[6])
                     + pfx[1] * (xvel[1]-xvel[7])
                     + pfx[2] * (xvel[2]-xvel[4])
                     + pfx[3] * (xvel[3]-xvel[5]) );

  d[1] = inv_detJ * ( pfy[0] * (yvel[0]-yvel[6])
                     + pfy[1] * (yvel[1]-yvel[7])
                     + pfy[2] * (yvel[2]-yvel[4])
                     + pfy[3] * (yvel[3]-yvel[5]) );

  d[2] = inv_detJ * ( pfz[0] * (zvel[0]-zvel[6])
                     + pfz[1] * (zvel[1]-zvel[7])
                     + pfz[2] * (zvel[2]-zvel[4])
                     + pfz[3] * (zvel[3]-zvel[5]) );

  dyddx  = inv_detJ * ( pfx[0] * (yvel[0]-yvel[6])
                      + pfx[1] * (yvel[1]-yvel[7])
                      + pfx[2] * (yvel[2]-yvel[4])
                      + pfx[3] * (yvel[3]-yvel[5]) );

  dxddy  = inv_detJ * ( pfy[0] * (xvel[0]-xvel[6])
                      + pfy[1] * (xvel[1]-xvel[7])
                      + pfy[2] * (xvel[2]-xvel[4])
                      + pfy[3] * (xvel[3]-xvel[5]) );

  dzddx  = inv_detJ * ( pfx[0] * (zvel[0]-zvel[6])
                      + pfx[1] * (zvel[1]-zvel[7])
                      + pfx[2] * (zvel[2]-zvel[4])
                      + pfx[3] * (zvel[3]-zvel[5]) );

  dxddz  = inv_detJ * ( pfz[0] * (xvel[0]-xvel[6])
                      + pfz[1] * (xvel[1]-xvel[7])
                      + pfz[2] * (xvel[2]-xvel[4])
                      + pfz[3] * (xvel[3]-xvel[5]) );

  dzddy  = inv_detJ * ( pfy[0] * (zvel[0]-zvel[6])
                      + pfy[1] * (zvel[1]-zvel[7])
                      + pfy[2] * (zvel[2]-zvel[4])
                      + pfy[3] * (zvel[3]-zvel[5]) );

  dyddz  = inv_detJ * ( pfz[0] * (yvel[0]-yvel[6])
                      + pfz[1] * (yvel[1]-yvel[7])
                      + pfz[2] * (yvel[2]-yvel[4])
                      + pfz[3] * (yvel[3]-yvel[5]) );
  d[5]  = double( .5) * ( dxddy + dyddx );
  d[4]  = double( .5) * ( dxddz + dzddx );
  d[3]  = double( .5) * ( dzddy + dyddz );
}

/******************************************/

//static inline
void CalcKinematicsForElems( Domain &domain,
                             double deltaTime, int numElem )
{

  // loop over all elements
#pragma omp parallel for firstprivate(numElem, deltaTime)
  for( int k=0 ; k<numElem ; ++k )
  {
    double B[3][8] ; /** shape function derivatives */
    double D[6] ;
    double x_local[8] ;
    double y_local[8] ;
    double z_local[8] ;
    double xd_local[8] ;
    double yd_local[8] ;
    double zd_local[8] ;
    double detJ = double(0.0) ;

    double volume ;
    double relativeVolume ;
    int* elemToNode = domain.nodelist(k) ;

    // get nodal coordinates from global arrays and copy into local arrays.
    CollectDomainNodesToElemNodes(domain, elemToNode, x_local, y_local, z_local);

    // volume calculations
    volume = CalcElemVolume(x_local, y_local, z_local );
    relativeVolume = volume / domain.volo(k) ;
    domain.vnew(k) = relativeVolume ;
    domain.delv(k) = relativeVolume - domain.v(k) ;

    // set characteristic length
    domain.arealg(k) = CalcElemCharacteristicLength(x_local, y_local, z_local,
                                             volume);

    // get nodal velocities from global array and copy into local arrays.
    for( int lnode=0 ; lnode<8 ; ++lnode )
    {
      int gnode = elemToNode[lnode];
      xd_local[lnode] = domain.xd(gnode);
      yd_local[lnode] = domain.yd(gnode);
      zd_local[lnode] = domain.zd(gnode);
    }

    double dt2 = double(0.5) * deltaTime;
    for ( int j=0 ; j<8 ; ++j )
    {
       x_local[j] -= dt2 * xd_local[j];
       y_local[j] -= dt2 * yd_local[j];
       z_local[j] -= dt2 * zd_local[j];
    }

    CalcElemShapeFunctionDerivatives( x_local, y_local, z_local,
                                      B, &detJ );

    CalcElemVelocityGradient( xd_local, yd_local, zd_local,
                               B, detJ, D );

    // put velocity gradient quantities into their global arrays.
    domain.dxx(k) = D[0];
    domain.dyy(k) = D[1];
    domain.dzz(k) = D[2];
  }
}

/******************************************/

static inline
void CalcLagrangeElements(Domain& domain)
{
   int numElem = domain.numElem() ;
   if (numElem > 0) {
      double deltatime = domain.deltatime() ;

      domain.AllocateStrains(numElem);

      CalcKinematicsForElems(domain, deltatime, numElem) ;

      // element loop to do some stuff not included in the elemlib function.
#pragma omp parallel for firstprivate(numElem)
      for ( int k=0 ; k<numElem ; ++k )
      {
         // calc strain rate and apply as constraint (only done in FB element)
         double vdov = domain.dxx(k) + domain.dyy(k) + domain.dzz(k) ;
         double vdovthird = vdov/double(3.0) ;

         // make the rate of deformation tensor deviatoric
         domain.vdov(k) = vdov ;
         domain.dxx(k) -= vdovthird ;
         domain.dyy(k) -= vdovthird ;
         domain.dzz(k) -= vdovthird ;

        // See if any volumes are negative, and take appropriate action.
         if (domain.vnew(k) <= double(0.0))
        {
#if USE_MPI           
           MPI_Abort(MPI_COMM_WORLD, VolumeError) ;
#else
           exit(VolumeError);
#endif
        }
      }
      domain.DeallocateStrains();
   }
}

/******************************************/

static inline
void CalcMonotonicQGradientsForElems(Domain& domain)
{
   int numElem = domain.numElem();

#pragma omp parallel for firstprivate(numElem)
   for (int i = 0 ; i < numElem ; ++i ) {
      double ptiny = double(1.e-36) ;
      double ax;
      double ay;
      double az;
      double dxv;
      double dyv;
      double dzv;

      int *elemToNode = domain.nodelist(i);
      int n0 = elemToNode[0] ;
      int n1 = elemToNode[1] ;
      int n2 = elemToNode[2] ;
      int n3 = elemToNode[3] ;
      int n4 = elemToNode[4] ;
      int n5 = elemToNode[5] ;
      int n6 = elemToNode[6] ;
      int n7 = elemToNode[7] ;

      double x0 = domain.x(n0) ;
      double x1 = domain.x(n1) ;
      double x2 = domain.x(n2) ;
      double x3 = domain.x(n3) ;
      double x4 = domain.x(n4) ;
      double x5 = domain.x(n5) ;
      double x6 = domain.x(n6) ;
      double x7 = domain.x(n7) ;

      double y0 = domain.y(n0) ;
      double y1 = domain.y(n1) ;
      double y2 = domain.y(n2) ;
      double y3 = domain.y(n3) ;
      double y4 = domain.y(n4) ;
      double y5 = domain.y(n5) ;
      double y6 = domain.y(n6) ;
      double y7 = domain.y(n7) ;

      double z0 = domain.z(n0) ;
      double z1 = domain.z(n1) ;
      double z2 = domain.z(n2) ;
      double z3 = domain.z(n3) ;
      double z4 = domain.z(n4) ;
      double z5 = domain.z(n5) ;
      double z6 = domain.z(n6) ;
      double z7 = domain.z(n7) ;

      double xv0 = domain.xd(n0) ;
      double xv1 = domain.xd(n1) ;
      double xv2 = domain.xd(n2) ;
      double xv3 = domain.xd(n3) ;
      double xv4 = domain.xd(n4) ;
      double xv5 = domain.xd(n5) ;
      double xv6 = domain.xd(n6) ;
      double xv7 = domain.xd(n7) ;

      double yv0 = domain.yd(n0) ;
      double yv1 = domain.yd(n1) ;
      double yv2 = domain.yd(n2) ;
      double yv3 = domain.yd(n3) ;
      double yv4 = domain.yd(n4) ;
      double yv5 = domain.yd(n5) ;
      double yv6 = domain.yd(n6) ;
      double yv7 = domain.yd(n7) ;

      double zv0 = domain.zd(n0) ;
      double zv1 = domain.zd(n1) ;
      double zv2 = domain.zd(n2) ;
      double zv3 = domain.zd(n3) ;
      double zv4 = domain.zd(n4) ;
      double zv5 = domain.zd(n5) ;
      double zv6 = domain.zd(n6) ;
      double zv7 = domain.zd(n7) ;

      double vol = domain.volo(i)*domain.vnew(i) ;
      double norm = double(1.0) / ( vol + ptiny ) ;

      double dxj = double(-0.25)*((x0+x1+x5+x4) - (x3+x2+x6+x7)) ;
      double dyj = double(-0.25)*((y0+y1+y5+y4) - (y3+y2+y6+y7)) ;
      double dzj = double(-0.25)*((z0+z1+z5+z4) - (z3+z2+z6+z7)) ;

      double dxi = double( 0.25)*((x1+x2+x6+x5) - (x0+x3+x7+x4)) ;
      double dyi = double( 0.25)*((y1+y2+y6+y5) - (y0+y3+y7+y4)) ;
      double dzi = double( 0.25)*((z1+z2+z6+z5) - (z0+z3+z7+z4)) ;

      double dxk = double( 0.25)*((x4+x5+x6+x7) - (x0+x1+x2+x3)) ;
      double dyk = double( 0.25)*((y4+y5+y6+y7) - (y0+y1+y2+y3)) ;
      double dzk = double( 0.25)*((z4+z5+z6+z7) - (z0+z1+z2+z3)) ;

      /* find delvk and delxk ( i cross j ) */

      ax = dyi*dzj - dzi*dyj ;
      ay = dzi*dxj - dxi*dzj ;
      az = dxi*dyj - dyi*dxj ;

      domain.delx_zeta(i) = vol / SQRT(ax*ax + ay*ay + az*az + ptiny) ;

      ax *= norm ;
      ay *= norm ;
      az *= norm ;

      dxv = double(0.25)*((xv4+xv5+xv6+xv7) - (xv0+xv1+xv2+xv3)) ;
      dyv = double(0.25)*((yv4+yv5+yv6+yv7) - (yv0+yv1+yv2+yv3)) ;
      dzv = double(0.25)*((zv4+zv5+zv6+zv7) - (zv0+zv1+zv2+zv3)) ;

      domain.delv_zeta(i) = ax*dxv + ay*dyv + az*dzv ;

      /* find delxi and delvi ( j cross k ) */

      ax = dyj*dzk - dzj*dyk ;
      ay = dzj*dxk - dxj*dzk ;
      az = dxj*dyk - dyj*dxk ;

      domain.delx_xi(i) = vol / SQRT(ax*ax + ay*ay + az*az + ptiny) ;

      ax *= norm ;
      ay *= norm ;
      az *= norm ;

      dxv = double(0.25)*((xv1+xv2+xv6+xv5) - (xv0+xv3+xv7+xv4)) ;
      dyv = double(0.25)*((yv1+yv2+yv6+yv5) - (yv0+yv3+yv7+yv4)) ;
      dzv = double(0.25)*((zv1+zv2+zv6+zv5) - (zv0+zv3+zv7+zv4)) ;

      domain.delv_xi(i) = ax*dxv + ay*dyv + az*dzv ;

      /* find delxj and delvj ( k cross i ) */

      ax = dyk*dzi - dzk*dyi ;
      ay = dzk*dxi - dxk*dzi ;
      az = dxk*dyi - dyk*dxi ;

      domain.delx_eta(i) = vol / SQRT(ax*ax + ay*ay + az*az + ptiny) ;

      ax *= norm ;
      ay *= norm ;
      az *= norm ;

      dxv = double(-0.25)*((xv0+xv1+xv5+xv4) - (xv3+xv2+xv6+xv7)) ;
      dyv = double(-0.25)*((yv0+yv1+yv5+yv4) - (yv3+yv2+yv6+yv7)) ;
      dzv = double(-0.25)*((zv0+zv1+zv5+zv4) - (zv3+zv2+zv6+zv7)) ;

      domain.delv_eta(i) = ax*dxv + ay*dyv + az*dzv ;
   }
}

/******************************************/

static inline
void CalcMonotonicQRegionForElems(Domain &domain, int r,
                                  double ptiny)
{
   double monoq_limiter_mult = domain.monoq_limiter_mult();
   double monoq_max_slope = domain.monoq_max_slope();
   double qlc_monoq = domain.qlc_monoq();
   double qqc_monoq = domain.qqc_monoq();

#pragma omp parallel for firstprivate(qlc_monoq, qqc_monoq, monoq_limiter_mult, monoq_max_slope, ptiny)
   for ( int i = 0 ; i < domain.regElemSize(r); ++i ) {
      int ielem = domain.regElemlist(r,i);
      double qlin;
      double qquad ;
      double phixi;
      double phieta;
      double phizeta ;
      int bcMask = domain.elemBC(ielem) ;
      double delvm = 0.0;
      double delvp =0.0;

      /*  phixi     */
      double norm = double(1.) / (domain.delv_xi(ielem)+ ptiny ) ;

      switch (bcMask & XI_M) {
         case XI_M_COMM: /* needs comm data */
         case 0:         delvm = domain.delv_xi(domain.lxim(ielem)); break ;
         case XI_M_SYMM: delvm = domain.delv_xi(ielem) ;       break ;
         case XI_M_FREE: delvm = double(0.0) ;      break ;
         default:          fprintf(stderr, "Error in switch at %s line %d\n",
                                   __FILE__, __LINE__);
            delvm = 0; /* ERROR - but quiets the compiler */
            break;
      }
      switch (bcMask & XI_P) {
         case XI_P_COMM: /* needs comm data */
         case 0:         delvp = domain.delv_xi(domain.lxip(ielem)) ; break ;
         case XI_P_SYMM: delvp = domain.delv_xi(ielem) ;       break ;
         case XI_P_FREE: delvp = double(0.0) ;      break ;
         default:          fprintf(stderr, "Error in switch at %s line %d\n",
                                   __FILE__, __LINE__);
            delvp = 0; /* ERROR - but quiets the compiler */
            break;
      }

      delvm = delvm * norm ;
      delvp = delvp * norm ;

      phixi = double(.5) * ( delvm + delvp ) ;

      delvm *= monoq_limiter_mult ;
      delvp *= monoq_limiter_mult ;

      if ( delvm < phixi ) phixi = delvm ;
      if ( delvp < phixi ) phixi = delvp ;
      if ( phixi < double(0.)) phixi = double(0.) ;
      if ( phixi > monoq_max_slope) phixi = monoq_max_slope;


      /*  phieta     */
      norm = double(1.) / ( domain.delv_eta(ielem) + ptiny ) ;

      switch (bcMask & ETA_M) {
         case ETA_M_COMM: /* needs comm data */
         case 0:          delvm = domain.delv_eta(domain.letam(ielem)) ; break ;
         case ETA_M_SYMM: delvm = domain.delv_eta(ielem) ;        break ;
         case ETA_M_FREE: delvm = double(0.0) ;        break ;
         default:          fprintf(stderr, "Error in switch at %s line %d\n",
                                   __FILE__, __LINE__);
            delvm = 0; /* ERROR - but quiets the compiler */
            break;
      }
      switch (bcMask & ETA_P) {
         case ETA_P_COMM: /* needs comm data */
         case 0:          delvp = domain.delv_eta(domain.letap(ielem)) ; break ;
         case ETA_P_SYMM: delvp = domain.delv_eta(ielem) ;        break ;
         case ETA_P_FREE: delvp = double(0.0) ;        break ;
         default:          fprintf(stderr, "Error in switch at %s line %d\n",
                                   __FILE__, __LINE__);
            delvp = 0; /* ERROR - but quiets the compiler */
            break;
      }

      delvm = delvm * norm ;
      delvp = delvp * norm ;

      phieta = double(.5) * ( delvm + delvp ) ;

      delvm *= monoq_limiter_mult ;
      delvp *= monoq_limiter_mult ;

      if ( delvm  < phieta ) phieta = delvm ;
      if ( delvp  < phieta ) phieta = delvp ;
      if ( phieta < double(0.)) phieta = double(0.) ;
      if ( phieta > monoq_max_slope)  phieta = monoq_max_slope;

      /*  phizeta     */
      norm = double(1.) / ( domain.delv_zeta(ielem) + ptiny ) ;

      switch (bcMask & ZETA_M) {
         case ZETA_M_COMM: /* needs comm data */
         case 0:           delvm = domain.delv_zeta(domain.lzetam(ielem)) ; break ;
         case ZETA_M_SYMM: delvm = domain.delv_zeta(ielem) ;         break ;
         case ZETA_M_FREE: delvm = double(0.0) ;          break ;
         default:          fprintf(stderr, "Error in switch at %s line %d\n",
                                   __FILE__, __LINE__);
            delvm = 0; /* ERROR - but quiets the compiler */
            break;
      }
      switch (bcMask & ZETA_P) {
         case ZETA_P_COMM: /* needs comm data */
         case 0:           delvp = domain.delv_zeta(domain.lzetap(ielem)) ; break ;
         case ZETA_P_SYMM: delvp = domain.delv_zeta(ielem) ;         break ;
         case ZETA_P_FREE: delvp = double(0.0) ;          break ;
         default:          fprintf(stderr, "Error in switch at %s line %d\n",
                                   __FILE__, __LINE__);
            delvp = 0; /* ERROR - but quiets the compiler */
            break;
      }

      delvm = delvm * norm ;
      delvp = delvp * norm ;

      phizeta = double(.5) * ( delvm + delvp ) ;

      delvm *= monoq_limiter_mult ;
      delvp *= monoq_limiter_mult ;

      if ( delvm   < phizeta ) phizeta = delvm ;
      if ( delvp   < phizeta ) phizeta = delvp ;
      if ( phizeta < double(0.)) phizeta = double(0.);
      if ( phizeta > monoq_max_slope  ) phizeta = monoq_max_slope;

      /* Remove length scale */

      if ( domain.vdov(ielem) > double(0.) )  {
         qlin  = double(0.) ;
         qquad = double(0.) ;
      }
      else {
         double delvxxi   = domain.delv_xi(ielem)   * domain.delx_xi(ielem)   ;
         double delvxeta  = domain.delv_eta(ielem)  * domain.delx_eta(ielem)  ;
         double delvxzeta = domain.delv_zeta(ielem) * domain.delx_zeta(ielem) ;

         if ( delvxxi   > double(0.) ) delvxxi   = double(0.) ;
         if ( delvxeta  > double(0.) ) delvxeta  = double(0.) ;
         if ( delvxzeta > double(0.) ) delvxzeta = double(0.) ;

         double rho = domain.elemMass(ielem) / (domain.volo(ielem) * domain.vnew(ielem)) ;

         qlin = -qlc_monoq * rho *
            (  delvxxi   * (double(1.) - phixi) +
               delvxeta  * (double(1.) - phieta) +
               delvxzeta * (double(1.) - phizeta)  ) ;

         qquad = qqc_monoq * rho *
            (  delvxxi*delvxxi     * (double(1.) - phixi*phixi) +
               delvxeta*delvxeta   * (double(1.) - phieta*phieta) +
               delvxzeta*delvxzeta * (double(1.) - phizeta*phizeta)  ) ;
      }

      domain.qq(ielem) = qquad ;
      domain.ql(ielem) = qlin  ;
   }
}

/******************************************/

static inline
void CalcMonotonicQForElems(Domain& domain)
{  
   //
   // initialize parameters
   // 
   double ptiny = double(1.e-36) ;

   //
   // calculate the monotonic q for all regions
   //
   for (int r=0 ; r<domain.numReg() ; ++r) {
      if (domain.regElemSize(r) > 0) {
         CalcMonotonicQRegionForElems(domain, r, ptiny) ;
      }
   }
}

/******************************************/

static inline
void CalcQForElems(Domain& domain)
{
   //
   // MONOTONIC Q option
   //

   int numElem = domain.numElem() ;

   if (numElem != 0) {
      int allElem = numElem +  /* local elem */
            2*domain.sizeX()*domain.sizeY() + /* plane ghosts */
            2*domain.sizeX()*domain.sizeZ() + /* row ghosts */
            2*domain.sizeY()*domain.sizeZ() ; /* col ghosts */

      domain.AllocateGradients(numElem, allElem);

#if USE_MPI      
      CommRecv(domain, MSG_MONOQ, 3,
               domain.sizeX(), domain.sizeY(), domain.sizeZ(),
               true, true) ;
#endif      

      /* Calculate velocity gradients */
      CalcMonotonicQGradientsForElems(domain);

#if USE_MPI      
      Domain_member fieldData[3] ;
      
      /* Transfer veloctiy gradients in the first order elements */
      /* problem->commElements->Transfer(CommElements::monoQ) ; */

      fieldData[0] = &Domain::delv_xi ;
      fieldData[1] = &Domain::delv_eta ;
      fieldData[2] = &Domain::delv_zeta ;

      CommSend(domain, MSG_MONOQ, 3, fieldData,
               domain.sizeX(), domain.sizeY(), domain.sizeZ(),
               true, true) ;

      CommMonoQ(domain) ;
#endif      

      CalcMonotonicQForElems(domain);

      // Free up memory
      domain.DeallocateGradients();

      /* Don't allow excessive artificial viscosity */
      int idx = -1; 
      for (int i=0; i<numElem; ++i) {
         if ( domain.q(i) > domain.qstop() ) {
            idx = i ;
            break ;
         }
      }

      if(idx >= 0) {
#if USE_MPI         
         MPI_Abort(MPI_COMM_WORLD, QStopError) ;
#else
         exit(QStopError);
#endif
      }
   }
}

/******************************************/

static inline
void CalcPressureForElems(double* p_new, double* bvc,
                          double* pbvc, double* e_old,
                          double* compression, double *vnewc,
                          double pmin,
                          double p_cut, double eosvmax,
                          int length, int *regElemList)
{
#pragma omp parallel for firstprivate(length)
   for (int i = 0; i < length ; ++i) {
      double c1s = double(2.0)/double(3.0) ;
      bvc[i] = c1s * (compression[i] + double(1.));
      pbvc[i] = c1s;
   }

#pragma omp parallel for firstprivate(length, pmin, p_cut, eosvmax)
   for (int i = 0 ; i < length ; ++i){
      int ielem = regElemList[i];
      
      p_new[i] = bvc[i] * e_old[i] ;

      if    (FABS(p_new[i]) <  p_cut   )
         p_new[i] = double(0.0) ;

      if    ( vnewc[ielem] >= eosvmax ) /* impossible condition here? */
         p_new[i] = double(0.0) ;

      if    (p_new[i]       <  pmin)
         p_new[i]   = pmin ;
   }
}

/******************************************/

static inline
void CalcEnergyForElems(double* p_new, double* e_new, double* q_new,
                        double* bvc, double* pbvc,
                        double* p_old, double* e_old, double* q_old,
                        double* compression, double* compHalfStep,
                        double* vnewc, double* work, double* delvc, double pmin,
                        double p_cut, double  e_cut, double q_cut, double emin,
                        double* qq_old, double* ql_old,
                        double rho0,
                        double eosvmax,
                        int length, int *regElemList)
{
   double *pHalfStep = Allocate<double>(length) ;

#pragma omp parallel for firstprivate(length, emin)
   for (int i = 0 ; i < length ; ++i) {
      e_new[i] = e_old[i] - double(0.5) * delvc[i] * (p_old[i] + q_old[i])
         + double(0.5) * work[i];

      if (e_new[i]  < emin ) {
         e_new[i] = emin ;
      }
   }

   CalcPressureForElems(pHalfStep, bvc, pbvc, e_new, compHalfStep, vnewc,
                        pmin, p_cut, eosvmax, length, regElemList);

#pragma omp parallel for firstprivate(length, rho0)
   for (int i = 0 ; i < length ; ++i) {
      double vhalf = double(1.) / (double(1.) + compHalfStep[i]) ;

      if ( delvc[i] > double(0.) ) {
         q_new[i] /* = qq_old[i] = ql_old[i] */ = double(0.) ;
      }
      else {
         double ssc = ( pbvc[i] * e_new[i]
                 + vhalf * vhalf * bvc[i] * pHalfStep[i] ) / rho0 ;

         if ( ssc <= double(.1111111e-36) ) {
            ssc = double(.3333333e-18) ;
         } else {
            ssc = SQRT(ssc) ;
         }

         q_new[i] = (ssc*ql_old[i] + qq_old[i]) ;
      }

      e_new[i] = e_new[i] + double(0.5) * delvc[i]
         * (  double(3.0)*(p_old[i]     + q_old[i])
              - double(4.0)*(pHalfStep[i] + q_new[i])) ;
   }

#pragma omp parallel for firstprivate(length, emin, e_cut)
   for (int i = 0 ; i < length ; ++i) {

      e_new[i] += double(0.5) * work[i];

      if (FABS(e_new[i]) < e_cut) {
         e_new[i] = double(0.)  ;
      }
      if (     e_new[i]  < emin ) {
         e_new[i] = emin ;
      }
   }

   CalcPressureForElems(p_new, bvc, pbvc, e_new, compression, vnewc,
                        pmin, p_cut, eosvmax, length, regElemList);

#pragma omp parallel for firstprivate(length, rho0, emin, e_cut)
   for (int i = 0 ; i < length ; ++i){
      double sixth = double(1.0) / double(6.0) ;
      int ielem = regElemList[i];
      double q_tilde ;

      if (delvc[i] > double(0.)) {
         q_tilde = double(0.) ;
      }
      else {
         double ssc = ( pbvc[i] * e_new[i]
                 + vnewc[ielem] * vnewc[ielem] * bvc[i] * p_new[i] ) / rho0 ;

         if ( ssc <= double(.1111111e-36) ) {
            ssc = double(.3333333e-18) ;
         } else {
            ssc = SQRT(ssc) ;
         }

         q_tilde = (ssc*ql_old[i] + qq_old[i]) ;
      }

      e_new[i] = e_new[i] - (  double(7.0)*(p_old[i]     + q_old[i])
                               - double(8.0)*(pHalfStep[i] + q_new[i])
                               + (p_new[i] + q_tilde)) * delvc[i]*sixth ;

      if (FABS(e_new[i]) < e_cut) {
         e_new[i] = double(0.)  ;
      }
      if (     e_new[i]  < emin ) {
         e_new[i] = emin ;
      }
   }

   CalcPressureForElems(p_new, bvc, pbvc, e_new, compression, vnewc,
                        pmin, p_cut, eosvmax, length, regElemList);

#pragma omp parallel for firstprivate(length, rho0, q_cut)
   for (int i = 0 ; i < length ; ++i){
      int ielem = regElemList[i];

      if ( delvc[i] <= double(0.) ) {
         double ssc = ( pbvc[i] * e_new[i]
                 + vnewc[ielem] * vnewc[ielem] * bvc[i] * p_new[i] ) / rho0 ;

         if ( ssc <= double(.1111111e-36) ) {
            ssc = double(.3333333e-18) ;
         } else {
            ssc = SQRT(ssc) ;
         }

         q_new[i] = (ssc*ql_old[i] + qq_old[i]) ;

         if (FABS(q_new[i]) < q_cut) q_new[i] = double(0.) ;
      }
   }

   Release(&pHalfStep) ;

   return ;
}

/******************************************/

static inline
void CalcSoundSpeedForElems(Domain &domain,
                            double *vnewc, double rho0, double *enewc,
                            double *pnewc, double *pbvc,
                            double *bvc, double ss4o3,
                            int len, int *regElemList)
{
#pragma omp parallel for firstprivate(rho0, ss4o3)
   for (int i = 0; i < len ; ++i) {
      int ielem = regElemList[i];
      double ssTmp = (pbvc[i] * enewc[i] + vnewc[ielem] * vnewc[ielem] *
                 bvc[i] * pnewc[i]) / rho0;
      if (ssTmp <= double(.1111111e-36)) {
         ssTmp = double(.3333333e-18);
      }
      else {
         ssTmp = SQRT(ssTmp);
      }
      domain.ss(ielem) = ssTmp ;
   }
}

/******************************************/

static inline
void EvalEOSForElems(Domain& domain, double *vnewc,
                     int numElemReg, int *regElemList, int rep)
{
   double  e_cut = domain.e_cut() ;
   double  p_cut = domain.p_cut() ;
   double  ss4o3 = domain.ss4o3() ;
   double  q_cut = domain.q_cut() ;

   double eosvmax = domain.eosvmax() ;
   double eosvmin = domain.eosvmin() ;
   double pmin    = domain.pmin() ;
   double emin    = domain.emin() ;
   double rho0    = domain.refdens() ;

   // These temporaries will be of different size for 
   // each call (due to different sized region element
   // lists)
   double *e_old = Allocate<double>(numElemReg) ;
   double *delvc = Allocate<double>(numElemReg) ;
   double *p_old = Allocate<double>(numElemReg) ;
   double *q_old = Allocate<double>(numElemReg) ;
   double *compression = Allocate<double>(numElemReg) ;
   double *compHalfStep = Allocate<double>(numElemReg) ;
   double *qq_old = Allocate<double>(numElemReg) ;
   double *ql_old = Allocate<double>(numElemReg) ;
   double *work = Allocate<double>(numElemReg) ;
   double *p_new = Allocate<double>(numElemReg) ;
   double *e_new = Allocate<double>(numElemReg) ;
   double *q_new = Allocate<double>(numElemReg) ;
   double *bvc = Allocate<double>(numElemReg) ;
   double *pbvc = Allocate<double>(numElemReg) ;
 
   //loop to add load imbalance based on region number 
   for(int j = 0; j < rep; j++) {
      /* compress data, minimal set */
#pragma omp parallel
      {
#pragma omp for nowait firstprivate(numElemReg)
         for (int i=0; i<numElemReg; ++i) {
            int ielem = regElemList[i];
            e_old[i] = domain.e(ielem) ;
            delvc[i] = domain.delv(ielem) ;
            p_old[i] = domain.p(ielem) ;
            q_old[i] = domain.q(ielem) ;
            qq_old[i] = domain.qq(ielem) ;
            ql_old[i] = domain.ql(ielem) ;
         }

#pragma omp for firstprivate(numElemReg)
         for (int i = 0; i < numElemReg ; ++i) {
            int ielem = regElemList[i];
            double vchalf ;
            compression[i] = double(1.) / vnewc[ielem] - double(1.);
            vchalf = vnewc[ielem] - delvc[i] * double(.5);
            compHalfStep[i] = double(1.) / vchalf - double(1.);
         }

      /* Check for v > eosvmax or v < eosvmin */
         if ( eosvmin != double(0.) ) {
#pragma omp for nowait firstprivate(numElemReg, eosvmin)
            for(int i=0 ; i<numElemReg ; ++i) {
               int ielem = regElemList[i];
               if (vnewc[ielem] <= eosvmin) { /* impossible due to calling func? */
                  compHalfStep[i] = compression[i] ;
               }
            }
         }
         if ( eosvmax != double(0.) ) {
#pragma omp for nowait firstprivate(numElemReg, eosvmax)
            for(int i=0 ; i<numElemReg ; ++i) {
               int ielem = regElemList[i];
               if (vnewc[ielem] >= eosvmax) { /* impossible due to calling func? */
                  p_old[i]        = double(0.) ;
                  compression[i]  = double(0.) ;
                  compHalfStep[i] = double(0.) ;
               }
            }
         }

#pragma omp for nowait firstprivate(numElemReg)
         for (int i = 0 ; i < numElemReg ; ++i) {
            work[i] = double(0.) ; 
         }
      }
      CalcEnergyForElems(p_new, e_new, q_new, bvc, pbvc,
                         p_old, e_old,  q_old, compression, compHalfStep,
                         vnewc, work,  delvc, pmin,
                         p_cut, e_cut, q_cut, emin,
                         qq_old, ql_old, rho0, eosvmax,
                         numElemReg, regElemList);
   }

#pragma omp parallel for firstprivate(numElemReg)
   for (int i=0; i<numElemReg; ++i) {
      int ielem = regElemList[i];
      domain.p(ielem) = p_new[i] ;
      domain.e(ielem) = e_new[i] ;
      domain.q(ielem) = q_new[i] ;
   }

   CalcSoundSpeedForElems(domain,
                          vnewc, rho0, e_new, p_new,
                          pbvc, bvc, ss4o3,
                          numElemReg, regElemList) ;

   Release(&pbvc) ;
   Release(&bvc) ;
   Release(&q_new) ;
   Release(&e_new) ;
   Release(&p_new) ;
   Release(&work) ;
   Release(&ql_old) ;
   Release(&qq_old) ;
   Release(&compHalfStep) ;
   Release(&compression) ;
   Release(&q_old) ;
   Release(&p_old) ;
   Release(&delvc) ;
   Release(&e_old) ;
}

/******************************************/

static inline
void ApplyMaterialPropertiesForElems(Domain& domain)
{
   int numElem = domain.numElem() ;

  if (numElem != 0) {
    /* Expose all of the variables needed for material evaluation */
    double eosvmin = domain.eosvmin() ;
    double eosvmax = domain.eosvmax() ;
    double *vnewc = Allocate<double>(numElem) ;

#pragma omp parallel
    {
#pragma omp for firstprivate(numElem)
       for(int i=0 ; i<numElem ; ++i) {
          vnewc[i] = domain.vnew(i) ;
       }

       // Bound the updated relative volumes with eosvmin/max
       if (eosvmin != double(0.)) {
#pragma omp for nowait firstprivate(numElem)
          for(int i=0 ; i<numElem ; ++i) {
             if (vnewc[i] < eosvmin)
                vnewc[i] = eosvmin ;
          }
       }

       if (eosvmax != double(0.)) {
#pragma omp for nowait firstprivate(numElem)
          for(int i=0 ; i<numElem ; ++i) {
             if (vnewc[i] > eosvmax)
                vnewc[i] = eosvmax ;
          }
       }

       // This check may not make perfect sense in LULESH, but
       // it's representative of something in the full code -
       // just leave it in, please
#pragma omp for nowait firstprivate(numElem)
       for (int i=0; i<numElem; ++i) {
          double vc = domain.v(i) ;
          if (eosvmin != double(0.)) {
             if (vc < eosvmin)
                vc = eosvmin ;
          }
          if (eosvmax != double(0.)) {
             if (vc > eosvmax)
                vc = eosvmax ;
          }
          if (vc <= 0.) {
#if USE_MPI
             MPI_Abort(MPI_COMM_WORLD, VolumeError) ;
#else
             exit(VolumeError);
#endif
          }
       }
    }

    for (int r=0 ; r<domain.numReg() ; r++) {
       int numElemReg = domain.regElemSize(r);
       int *regElemList = domain.regElemlist(r);
       int rep;
       //Determine load imbalance for this region
       //round down the number with lowest cost
       if(r < domain.numReg()/2)
	 rep = 1;
       //you don't get an expensive region unless you at least have 5 regions
       else if(r < (domain.numReg() - (domain.numReg()+15)/20))
         rep = 1 + domain.cost();
       //very expensive regions
       else
	 rep = 10 * (1+ domain.cost());
       EvalEOSForElems(domain, vnewc, numElemReg, regElemList, rep);
    }

    Release(&vnewc) ;
  }
}

/******************************************/

static inline
void UpdateVolumesForElems(Domain &domain,
                           double v_cut, int length)
{
   if (length != 0) {
#pragma omp parallel for firstprivate(length, v_cut)
      for(int i=0 ; i<length ; ++i) {
         double tmpV = domain.vnew(i) ;

         if ( FABS(tmpV - double(1.0)) < v_cut )
            tmpV = double(1.0) ;

         domain.v(i) = tmpV ;
      }
   }

   return ;
}

/******************************************/

static inline
void LagrangeElements(Domain& domain, int numElem)
{
  CalcLagrangeElements(domain) ;

  /* Calculate Q.  (Monotonic q option requires communication) */
  CalcQForElems(domain) ;

  ApplyMaterialPropertiesForElems(domain) ;

  UpdateVolumesForElems(domain, 
                        domain.v_cut(), numElem) ;
}

/******************************************/

static inline
void CalcCourantConstraintForElems(Domain &domain, int length,
                                   int *regElemlist,
                                   double qqc, double& dtcourant)
{
#if _OPENMP
   int threads = omp_get_max_threads();
   int courant_elem_per_thread[threads];
   double dtcourant_per_thread[threads];
#else
   int threads = 1;
   int courant_elem_per_thread[1];
   double  dtcourant_per_thread[1];
#endif

#pragma omp parallel firstprivate(length, qqc)
   {
      double   qqc2 = double(64.0) * qqc * qqc ;
      double   dtcourant_tmp = dtcourant;
      int  courant_elem  = -1 ;

#if _OPENMP
      int thread_num = omp_get_thread_num();
#else
      int thread_num = 0;
#endif      

#pragma omp for 
      for (int i = 0 ; i < length ; ++i) {
         int indx = regElemlist[i] ;
         double dtf = domain.ss(indx) * domain.ss(indx) ;

         if ( domain.vdov(indx) < double(0.) ) {
            dtf = dtf
                + qqc2 * domain.arealg(indx) * domain.arealg(indx)
                * domain.vdov(indx) * domain.vdov(indx) ;
         }

         dtf = SQRT(dtf) ;
         dtf = domain.arealg(indx) / dtf ;

         if (domain.vdov(indx) != double(0.)) {
            if ( dtf < dtcourant_tmp ) {
               dtcourant_tmp = dtf ;
               courant_elem  = indx ;
            }
         }
      }

      dtcourant_per_thread[thread_num]    = dtcourant_tmp ;
      courant_elem_per_thread[thread_num] = courant_elem ;
   }

   for (int i = 1; i < threads; ++i) {
      if (dtcourant_per_thread[i] < dtcourant_per_thread[0] ) {
         dtcourant_per_thread[0]    = dtcourant_per_thread[i];
         courant_elem_per_thread[0] = courant_elem_per_thread[i];
      }
   }

   if (courant_elem_per_thread[0] != -1) {
      dtcourant = dtcourant_per_thread[0] ;
   }

   return ;

}

/******************************************/

static inline
void CalcHydroConstraintForElems(Domain &domain, int length,
                                 int *regElemlist, double dvovmax, double& dthydro)
{
#if _OPENMP
   int threads = omp_get_max_threads();
   int hydro_elem_per_thread[threads];
   double dthydro_per_thread[threads];
#else
   int threads = 1;
   int hydro_elem_per_thread[1];
   double  dthydro_per_thread[1];
#endif

#pragma omp parallel firstprivate(length, dvovmax)
   {
      double dthydro_tmp = dthydro ;
      int hydro_elem = -1 ;

#if _OPENMP
      int thread_num = omp_get_thread_num();
#else      
      int thread_num = 0;
#endif      

#pragma omp for
      for (int i = 0 ; i < length ; ++i) {
         int indx = regElemlist[i] ;

         if (domain.vdov(indx) != double(0.)) {
            double dtdvov = dvovmax / (FABS(domain.vdov(indx))+double(1.e-20)) ;

            if ( dthydro_tmp > dtdvov ) {
                  dthydro_tmp = dtdvov ;
                  hydro_elem = indx ;
            }
         }
      }

      dthydro_per_thread[thread_num]    = dthydro_tmp ;
      hydro_elem_per_thread[thread_num] = hydro_elem ;
   }

   for (int i = 1; i < threads; ++i) {
      if(dthydro_per_thread[i] < dthydro_per_thread[0]) {
         dthydro_per_thread[0]    = dthydro_per_thread[i];
         hydro_elem_per_thread[0] =  hydro_elem_per_thread[i];
      }
   }

   if (hydro_elem_per_thread[0] != -1) {
      dthydro =  dthydro_per_thread[0] ;
   }

   return ;
}

/******************************************/

static inline
void CalcTimeConstraintsForElems(Domain& domain) {

   // Initialize conditions to a very large value
   domain.dtcourant() = 1.0e+20;
   domain.dthydro() = 1.0e+20;

   for (int r=0 ; r < domain.numReg() ; ++r) {
      /* evaluate time constraint */
      CalcCourantConstraintForElems(domain, domain.regElemSize(r),
                                    domain.regElemlist(r),
                                    domain.qqc(),
                                    domain.dtcourant()) ;

      /* check hydro constraint */
      CalcHydroConstraintForElems(domain, domain.regElemSize(r),
                                  domain.regElemlist(r),
                                  domain.dvovmax(),
                                  domain.dthydro()) ;
   }
}

/******************************************/

static inline
void LagrangeLeapFrog(Domain& domain)
{
#ifdef SEDOV_SYNC_POS_VEL_LATE
   Domain_member fieldData[6] ;
#endif

   /* calculate nodal forces, accelerations, velocities, positions, with
    * applied boundary conditions and slide surface considerations */
   LagrangeNodal(domain);


#ifdef SEDOV_SYNC_POS_VEL_LATE
#endif

   /* calculate element quantities (i.e. velocity gradient & q), and update
    * material states */
   LagrangeElements(domain, domain.numElem());

#if USE_MPI   
#ifdef SEDOV_SYNC_POS_VEL_LATE
   CommRecv(domain, MSG_SYNC_POS_VEL, 6,
            domain.sizeX() + 1, domain.sizeY() + 1, domain.sizeZ() + 1,
            false, false) ;

   fieldData[0] = &Domain::x ;
   fieldData[1] = &Domain::y ;
   fieldData[2] = &Domain::z ;
   fieldData[3] = &Domain::xd ;
   fieldData[4] = &Domain::yd ;
   fieldData[5] = &Domain::zd ;
   
   CommSend(domain, MSG_SYNC_POS_VEL, 6, fieldData,
            domain.sizeX() + 1, domain.sizeY() + 1, domain.sizeZ() + 1,
            false, false) ;
#endif
#endif   

   CalcTimeConstraintsForElems(domain);

#if USE_MPI   
#ifdef SEDOV_SYNC_POS_VEL_LATE
   CommSyncPosVel(domain) ;
#endif
#endif   
}


/******************************************/

int main(int argc, char *argv[])
{
   Domain *locDom ;
   int numRanks ;
   int myRank ;
   struct cmdLineOpts opts;

#if USE_MPI   
   Domain_member fieldData ;
   
#ifdef _OPENMP
   int thread_support;

   MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &thread_support);
   if (thread_support==MPI_THREAD_SINGLE)
    {
        fprintf(stderr,"The MPI implementation has no support for threading\n");
        MPI_Finalize();
        exit(1);
    }
#else
   MPI_Init(&argc, &argv);
#endif
    
   MPI_Comm_size(MPI_COMM_WORLD, &numRanks) ;
   MPI_Comm_rank(MPI_COMM_WORLD, &myRank) ;
#else
   numRanks = 1;
   myRank = 0;
#endif   

   /* Set defaults that can be overridden by command line opts */
   opts.its = 9999999;
   opts.nx  = 30;
   opts.numReg = 11;
   opts.numFiles = (int)(numRanks+10)/9;
   opts.showProg = 0;
   opts.quiet = 0;
   opts.viz = 0;
   opts.balance = 1;
   opts.cost = 1;

   ParseCommandLineOptions(argc, argv, myRank, &opts);

   if ((myRank == 0) && (opts.quiet == 0)) {
      std::cout << "Running problem size " << opts.nx << "^3 per domain until completion\n";
      std::cout << "Num processors: "      << numRanks << "\n";
#if _OPENMP
      std::cout << "Num threads: " << omp_get_max_threads() << "\n";
#endif
      std::cout << "Total number of elements: " << ((Int8_t)numRanks*opts.nx*opts.nx*opts.nx) << " \n\n";
      std::cout << "To run other sizes, use -s <integer>.\n";
      std::cout << "To run a fixed number of iterations, use -i <integer>.\n";
      std::cout << "To run a more or less balanced region set, use -b <integer>.\n";
      std::cout << "To change the relative costs of regions, use -c <integer>.\n";
      std::cout << "To print out progress, use -p\n";
      std::cout << "To write an output file for VisIt, use -v\n";
      std::cout << "See help (-h) for more options\n\n";
   }

   // Set up the mesh and decompose. Assumes regular cubes for now
   int col, row, plane, side;
   InitMeshDecomp(numRanks, myRank, &col, &row, &plane, &side);

   // Build the main data structure and initialize it
   locDom = new Domain(numRanks, col, row, plane, opts.nx,
                       side, opts.numReg, opts.balance, opts.cost) ;


#if USE_MPI   
   fieldData = &Domain::nodalMass ;

   // Initial domain boundary communication 
   CommRecv(*locDom, MSG_COMM_SBN, 1,
            locDom->sizeX() + 1, locDom->sizeY() + 1, locDom->sizeZ() + 1,
            true, false) ;
   CommSend(*locDom, MSG_COMM_SBN, 1, &fieldData,
            locDom->sizeX() + 1, locDom->sizeY() + 1, locDom->sizeZ() +  1,
            true, false) ;
   CommSBN(*locDom, 1, &fieldData) ;

   // End initialization
   MPI_Barrier(MPI_COMM_WORLD);
#endif   
   
   // BEGIN timestep to solution */
#if USE_MPI   
   double start = MPI_Wtime();
#else
   timeval start;
   gettimeofday(&start, NULL) ;
#endif
//debug to see region sizes
//   for(int i = 0; i < locDom->numReg(); i++)
//      std::cout << "region" << i + 1<< "size" << locDom->regElemSize(i) <<std::endl;
   while((locDom->time() < locDom->stoptime()) && (locDom->cycle() < opts.its)) {

      TimeIncrement(*locDom) ;
      LagrangeLeapFrog(*locDom) ;

      if ((opts.showProg != 0) && (opts.quiet == 0) && (myRank == 0)) {
         std::cout << "cycle = " << locDom->cycle()       << ", "
                   << std::scientific
                   << "time = " << double(locDom->time()) << ", "
                   << "dt="     << double(locDom->deltatime()) << "\n";
         std::cout.unsetf(std::ios_base::floatfield);
      }
   }

   // Use reduced max elapsed time
   double elapsed_time;
#if USE_MPI   
   elapsed_time = MPI_Wtime() - start;
#else
   timeval end;
   gettimeofday(&end, NULL) ;
   elapsed_time = (double)(end.tv_sec - start.tv_sec) + ((double)(end.tv_usec - start.tv_usec))/1000000 ;
#endif
   double elapsed_timeG;
#if USE_MPI   
   MPI_Reduce(&elapsed_time, &elapsed_timeG, 1, MPI_DOUBLE,
              MPI_MAX, 0, MPI_COMM_WORLD);
#else
   elapsed_timeG = elapsed_time;
#endif

   // Write out final viz file */
   if (opts.viz) {
      DumpToVisit(*locDom, opts.numFiles, myRank, numRanks) ;
   }
   
   if ((myRank == 0) && (opts.quiet == 0)) {
      VerifyAndWriteFinalOutput(elapsed_timeG, *locDom, opts.nx, numRanks);
   }

   delete locDom; 

#if USE_MPI
   MPI_Finalize() ;
#endif

   return 0 ;
}
