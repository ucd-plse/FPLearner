#include <iostream>
#include <fstream>
#include <cmath>
#include <stdlib.h>
#include <omp.h>

#define DOUBLE 0
#define FLOAT 1
#define INT 2


void writeData(double* ptr, size_t size, int type, char *name);
void writeData(float* ptr, size_t size, int type, char *name);
void writeData(int* ptr, size_t size, int type, char *name);

void writeData(double* ptr, size_t size, int type, const char *name);
void writeData(float* ptr, size_t size, int type, const char *name);
void writeData(int* ptr, size_t size, int type, const char *name);

float *  MP_Malloc(size_t size, float *);
double *  MP_Malloc(size_t size, double *);
float*  MP_Malloc(int size, float *);
double*  MP_Malloc(int size, double *);

void stopMeasure();
void startMeasure();




#define block_length 128
/*
 * Options
 *
 */
#define GAMMA 1.4
#define iterations 50

#define NDIM 3
#define NNB 4

#define RK 3	// 3rd order RK
#define ff_mach 1.2
#define deg_angle_of_attack 0.0f

/*
 * not options
 */
#define VAR_DENSITY 0
#define VAR_MOMENTUM  1
#define VAR_DENSITY_ENERGY (VAR_MOMENTUM+NDIM)
#define NVAR (VAR_DENSITY_ENERGY+1)


/*
 * Generic functions
 */

template <class T>
void dealloc(T* array)
{
    free(array);
}


void copy(double* dst, double* src, int N)
{
//#pragma omp parallel for default(shared) schedule(static)
    for(int i = 0; i < N; i++)
    {
        dst[i] = src[i];
    }
}



static inline void dump(double* variables, int nel, int nelr, const char *out_file_name)
{

    size_t outputSize = nel + nel*NDIM +nel;
    double* data = MP_Malloc(outputSize,data);
    size_t cnt = 0;
    std::ofstream file(out_file_name,std::ios::out | std::ios::binary);
    file.write(reinterpret_cast<char *>(&outputSize),sizeof(size_t));
    for(int i = 0; i < nel; i++) data[cnt++]= variables[i*NVAR+VAR_DENSITY];

    for(int i = 0; i < nel; i++)
    {
        for(int j = 0; j != NDIM; j++) data[cnt++] =  variables[i*NVAR+(VAR_MOMENTUM+j)];
    }
    for(int i = 0; i < nel; i++) data[cnt++] = variables[i*NVAR + VAR_DENSITY_ENERGY];
    writeData(data,outputSize, DOUBLE, out_file_name); 
    delete[] data;
}



/*
 * Element-based Cell-centered FVM solver functions
 */
double ff_variable[NVAR];
double ff_fc_momentum_xx;
double ff_fc_momentum_xy;
double ff_fc_momentum_xz;
double ff_fc_momentum_yx;
double ff_fc_momentum_yy;
double ff_fc_momentum_yz;
double ff_fc_momentum_zx;
double ff_fc_momentum_zy;
double ff_fc_momentum_zz;
double ff_fc_density_energyx;
double ff_fc_density_energyy;
double ff_fc_density_energyz;


static inline void initialize_variables(int nelr, double* variables)
{
//#pragma omp parallel for default(shared) schedule(static)
    for(int i = 0; i < nelr; i++)
    {
        for(int j = 0; j < NVAR; j++) variables[i*NVAR + j] = ff_variable[j];
    }
}


static inline void compute_flux_contribution(double density, 
double& momentumx,double& momentumy, double& momentumz, 
double& density_energy, double& pressure,
double& velocityx, double& velocityy, double& velocityz, 
double& fc_momentumxx, double& fc_momentumxy, double& fc_momentumxz, 
double& fc_momentumyx, double& fc_momentumyy, double& fc_momentumyz, 
double& fc_momentumzx, double& fc_momentumzy, double& fc_momentumzz, 
double& fc_density_energyx, double& fc_density_energyy, double &fc_density_energyz)
{
    fc_momentumxx = velocityx*momentumx + pressure;
    fc_momentumxy = velocityx*momentumy;
    fc_momentumxz = velocityx*momentumz;

    fc_momentumyx = fc_momentumxy;
    fc_momentumyy = velocityy*momentumy + pressure;
    fc_momentumyz = velocityy*momentumz;

    fc_momentumzx = fc_momentumxz;
    fc_momentumzy = fc_momentumyz;
    fc_momentumzz = velocityz*momentumz + pressure;

    double de_p = density_energy+pressure;
    fc_density_energyx = velocityx*de_p;
    fc_density_energyy = velocityy*de_p;
    fc_density_energyz = velocityz*de_p;
}


static inline void compute_velocity(double& density, 
        double& momentumx, double& momentumy, double& momentumz,
        double& velocityx, double& velocityy, double& velocityz)
{
    velocityx = momentumx / density;
    velocityy = momentumy / density;
    velocityz = momentumz / density;
}

static inline double compute_speed_sqd(double& velocityx, double& velocityy, double& velocityz)
{
    return velocityx*velocityx + velocityy*velocityy + velocityz*velocityz;
}

static inline double compute_pressure(double& density, double& density_energy, double& speed_sqd)
{
    return (double(GAMMA)-double(1.0))*(density_energy - double(0.5)*density*speed_sqd);
}

static inline double compute_speed_of_sound(double& density, double& pressure)
{
    return std::sqrt(double(GAMMA)*pressure/density);
}

static inline void compute_step_factor(int nelr, double* variables, double* areas, double* step_factors)
{
//#pragma omp parallel for default(shared) schedule(static)
    for(int i = 0; i < nelr; i++)
    {
        double density = variables[NVAR*i + VAR_DENSITY];

        double momentumx;
        double momentumy;
        double momentumz;
        momentumx = variables[NVAR*i + (VAR_MOMENTUM+0)];
        momentumy = variables[NVAR*i + (VAR_MOMENTUM+1)];
        momentumz = variables[NVAR*i + (VAR_MOMENTUM+2)];

        double density_energy = variables[NVAR*i + VAR_DENSITY_ENERGY];
        double velocityx;
        double velocityy;
        double velocityz;	   
        compute_velocity(density, momentumx, momentumy,momentumz, velocityx, velocityy, velocityz);
        double speed_sqd      = compute_speed_sqd(velocityx, velocityy, velocityz);
        double pressure       = compute_pressure(density, density_energy, speed_sqd);
        double speed_of_sound = compute_speed_of_sound(density, pressure);

        step_factors[i] = double(0.5f) / (std::sqrt(areas[i]) * (std::sqrt(speed_sqd) + speed_of_sound));
    }
}

static inline void compute_flux_contributions(int nelr, double* variables, double* fc_momentum_x, double* fc_momentum_y, double* fc_momentum_z, double* fc_density_energy)
{
//#pragma omp parallel for default(shared) schedule(static)
    for(int i = 0; i < nelr; i++)
    {
        double density_i = variables[NVAR*i + VAR_DENSITY];
        double momentum_ix;
        double momentum_iy;
        double momentum_iz;
        momentum_ix = variables[NVAR*i + (VAR_MOMENTUM+0)];
        momentum_iy = variables[NVAR*i + (VAR_MOMENTUM+1)];
        momentum_iz = variables[NVAR*i + (VAR_MOMENTUM+2)];
        double density_energy_i = variables[NVAR*i + VAR_DENSITY_ENERGY];

        double velocity_ix;
        double velocity_iy;
        double velocity_iz;
        compute_velocity(density_i, momentum_ix, momentum_iy,momentum_iz, velocity_ix, velocity_iy, velocity_iz);

        double speed_sqd_i  = compute_speed_sqd(velocity_ix, velocity_iy, velocity_iz);
        double speed_i  = sqrtf(speed_sqd_i);
        double pressure_i= compute_pressure(density_i, density_energy_i, speed_sqd_i);
        double speed_of_sound_i= compute_speed_of_sound(density_i, pressure_i);
        double fc_i_momentum_xx;
        double fc_i_momentum_xy;
        double fc_i_momentum_xz;
        double fc_i_momentum_yx;
        double fc_i_momentum_yy;
        double fc_i_momentum_yz;
        double fc_i_momentum_zx;
        double fc_i_momentum_zy;
        double fc_i_momentum_zz;
        double fc_i_density_energyx;
        double fc_i_density_energyy;
        double fc_i_density_energyz;	
        compute_flux_contribution(density_i, 
                momentum_ix,momentum_iy, momentum_iz,
                density_energy_i, pressure_i,
                velocity_ix,velocity_iy,velocity_iz,
                fc_i_momentum_xx, fc_i_momentum_xy, fc_i_momentum_xz,
                fc_i_momentum_yx, fc_i_momentum_yy, fc_i_momentum_yz,
                fc_i_momentum_zx, fc_i_momentum_zy, fc_i_momentum_zz,
                fc_i_density_energyx, fc_i_density_energyy, fc_i_density_energyz);

        fc_momentum_x[i*NDIM + 0] = fc_i_momentum_xx;
        fc_momentum_x[i*NDIM + 1] = fc_i_momentum_xy;
        fc_momentum_x[i*NDIM+  2] = fc_i_momentum_xz;

        fc_momentum_y[i*NDIM+ 0] = fc_i_momentum_yx;
        fc_momentum_y[i*NDIM+ 1] = fc_i_momentum_yy;
        fc_momentum_y[i*NDIM+ 2] = fc_i_momentum_yz;


        fc_momentum_z[i*NDIM+ 0] = fc_i_momentum_zx;
        fc_momentum_z[i*NDIM+ 1] = fc_i_momentum_zy;
        fc_momentum_z[i*NDIM+ 2] = fc_i_momentum_zz;

        fc_density_energy[i*NDIM+ 0] = fc_i_density_energyx;
        fc_density_energy[i*NDIM+ 1] = fc_i_density_energyy;
        fc_density_energy[i*NDIM+ 2] = fc_i_density_energyz;
    }

}

/*
 *
 *
 */

static inline void compute_flux(int nelr,
        int* elements_surrounding_elements,
        double* normals, double* variables, 
        double* fc_momentum_x, double* fc_momentum_y, 
        double* fc_momentum_z, double* fc_density_energy, 
        double* fluxes)
{
    double smoothing_coefficient = double(0.2f);

//#pragma omp parallel for default(shared) schedule(static)
    for(int i = 0; i < nelr; i++)
    {
        int j, nb;
        double normalx;
        double normaly;
        double normalz;
        double normal_len;
        double factor;

        double density_i = variables[NVAR*i + VAR_DENSITY];
        double momentum_ix;
        double momentum_iy;
        double momentum_iz;
        momentum_ix = variables[NVAR*i + (VAR_MOMENTUM+0)];
        momentum_iy = variables[NVAR*i + (VAR_MOMENTUM+1)];
        momentum_iz = variables[NVAR*i + (VAR_MOMENTUM+2)];
        double density_energy_i = variables[NVAR*i + VAR_DENSITY_ENERGY];
        double velocity_ix;
        double velocity_iy;
        double velocity_iz;
        compute_velocity(density_i, 
                momentum_ix, momentum_iy, momentum_iz,
                velocity_ix, velocity_iy, velocity_iz);
        double speed_sqd_i = compute_speed_sqd(velocity_ix, velocity_iy, velocity_iz);
        double speed_i= std::sqrt(speed_sqd_i);
        double pressure_i= compute_pressure(density_i, density_energy_i, speed_sqd_i);
        double speed_of_sound_i = compute_speed_of_sound(density_i, pressure_i);
        double fc_i_momentum_xx;
        double fc_i_momentum_xy;
        double fc_i_momentum_xz;
        double fc_i_momentum_yx;
        double fc_i_momentum_yy;
        double fc_i_momentum_yz;
        double fc_i_momentum_zx;
        double fc_i_momentum_zy;
        double fc_i_momentum_zz;
        double fc_i_density_energyx;
        double fc_i_density_energyy;
        double fc_i_density_energyz;

        fc_i_momentum_xx = fc_momentum_x[i*NDIM + 0];
        fc_i_momentum_xy = fc_momentum_x[i*NDIM + 1];
        fc_i_momentum_xz = fc_momentum_x[i*NDIM + 2];

        fc_i_momentum_yx = fc_momentum_y[i*NDIM + 0];
        fc_i_momentum_yy = fc_momentum_y[i*NDIM + 1];
        fc_i_momentum_yz = fc_momentum_y[i*NDIM + 2];

        fc_i_momentum_zx = fc_momentum_z[i*NDIM + 0];
        fc_i_momentum_zy = fc_momentum_z[i*NDIM + 1];
        fc_i_momentum_zz = fc_momentum_z[i*NDIM + 2];

        fc_i_density_energyx = fc_density_energy[i*NDIM + 0];
        fc_i_density_energyy = fc_density_energy[i*NDIM + 1];
        fc_i_density_energyz = fc_density_energy[i*NDIM + 2];

        double flux_i_density = double(0.0f);
        double flux_i_momentumx;
        double flux_i_momentumy;
        double flux_i_momentumz;
        flux_i_momentumx = double(0.0f);
        flux_i_momentumy = double(0.0f);
        flux_i_momentumz = double(0.0f);
        double flux_i_density_energy = double(0.0f);

        double velocity_nbx;
        double velocity_nby;
        double velocity_nbz;
        double density_nb;
        double density_energy_nb;
        double momentum_nbx;
        double momentum_nby;
        double momentum_nbz;
        double fc_nb_momentum_xx;
        double fc_nb_momentum_xy;
        double fc_nb_momentum_xz;
        double fc_nb_momentum_yx;
        double fc_nb_momentum_yy;
        double fc_nb_momentum_yz;
        double fc_nb_momentum_zx;
        double fc_nb_momentum_zy;
        double fc_nb_momentum_zz;

        double fc_nb_density_energyx;
        double fc_nb_density_energyy;
        double fc_nb_density_energyz;
        double speed_sqd_nb;
        double speed_of_sound_nb;
        double pressure_nb;

        for(j = 0; j < NNB; j++)
        {
            nb = elements_surrounding_elements[i*NNB + j];
            normalx = normals[(i*NNB + j)*NDIM + 0];
            normaly = normals[(i*NNB + j)*NDIM + 1];
            normalz = normals[(i*NNB + j)*NDIM + 2];
            normal_len = std::sqrt(normalx*normalx + normaly*normaly + normalz*normalz);

            if(nb >= 0) 	// a legitimate neighbor
            {
                density_nb =        variables[nb*NVAR + VAR_DENSITY];
                momentum_nbx =     variables[nb*NVAR + (VAR_MOMENTUM+0)];
                momentum_nby =     variables[nb*NVAR + (VAR_MOMENTUM+1)];
                momentum_nbz =     variables[nb*NVAR + (VAR_MOMENTUM+2)];
                density_energy_nb = variables[nb*NVAR + VAR_DENSITY_ENERGY];
                compute_velocity(density_nb, 
                        momentum_nbx, momentum_nby, momentum_nbz,
                        velocity_nbx, velocity_nby, velocity_nbz);
                speed_sqd_nb = compute_speed_sqd(velocity_nbx, velocity_nby, velocity_nbz);
                pressure_nb = compute_pressure(density_nb, density_energy_nb, speed_sqd_nb);
                speed_of_sound_nb   = compute_speed_of_sound(density_nb, pressure_nb);
                fc_nb_momentum_xx = fc_momentum_x[nb*NDIM + 0];
                fc_nb_momentum_xy = fc_momentum_x[nb*NDIM + 1];
                fc_nb_momentum_xz = fc_momentum_x[nb*NDIM + 2];

                fc_nb_momentum_yx = fc_momentum_y[nb*NDIM + 0];
                fc_nb_momentum_yy = fc_momentum_y[nb*NDIM + 1];
                fc_nb_momentum_yz = fc_momentum_y[nb*NDIM + 2];

                fc_nb_momentum_zx = fc_momentum_z[nb*NDIM + 0];
                fc_nb_momentum_zy = fc_momentum_z[nb*NDIM + 1];
                fc_nb_momentum_zz = fc_momentum_z[nb*NDIM + 2];

                fc_nb_density_energyx = fc_density_energy[nb*NDIM + 0];
                fc_nb_density_energyy = fc_density_energy[nb*NDIM + 1];
                fc_nb_density_energyz = fc_density_energy[nb*NDIM + 2];

                // artificial viscosity
                factor = -normal_len*smoothing_coefficient*double(0.5f)*(speed_i + std::sqrt(speed_sqd_nb) + speed_of_sound_i + speed_of_sound_nb);
                flux_i_density += factor*(density_i-density_nb);
                flux_i_density_energy += factor*(density_energy_i-density_energy_nb);
                flux_i_momentumx += factor*(momentum_ix-momentum_nbx);
                flux_i_momentumy += factor*(momentum_iy-momentum_nby);
                flux_i_momentumz += factor*(momentum_iz-momentum_nbz);

                // accumulate cell-centered fluxes
                factor = double(0.5f)*normalx;
                flux_i_density += factor*(momentum_nbx+momentum_ix);
                flux_i_density_energy += factor*(fc_nb_density_energyx+fc_i_density_energyx);
                flux_i_momentumx += factor*(fc_nb_momentum_xx+fc_i_momentum_xx);
                flux_i_momentumy += factor*(fc_nb_momentum_yx+fc_i_momentum_yx);
                flux_i_momentumz += factor*(fc_nb_momentum_zx+fc_i_momentum_zx);

                factor = double(0.5f)*normaly;
                flux_i_density += factor*(momentum_nby+momentum_iy);
                flux_i_density_energy += factor*(fc_nb_density_energyy+fc_i_density_energyy);
                flux_i_momentumx += factor*(fc_nb_momentum_xy+fc_i_momentum_xy);
                flux_i_momentumy += factor*(fc_nb_momentum_yy+fc_i_momentum_yy);
                flux_i_momentumz += factor*(fc_nb_momentum_zy+fc_i_momentum_zy);

                factor = double(0.5f)*normalz;
                flux_i_density += factor*(momentum_nbz+momentum_iz);
                flux_i_density_energy += factor*(fc_nb_density_energyz+fc_i_density_energyz);
                flux_i_momentumx += factor*(fc_nb_momentum_xz+fc_i_momentum_xz);
                flux_i_momentumy += factor*(fc_nb_momentum_yz+fc_i_momentum_yz);
                flux_i_momentumz += factor*(fc_nb_momentum_zz+fc_i_momentum_zz);
            }
            else if(nb == -1)	// a wing boundary
            {
                flux_i_momentumx += normalx*pressure_i;
                flux_i_momentumy += normaly*pressure_i;
                flux_i_momentumz += normalz*pressure_i;
            }
            else if(nb == -2) // a far field boundary
            {
                factor = double(0.5f)*normalx;
                flux_i_density += factor*(ff_variable[VAR_MOMENTUM+0]+momentum_ix);
                flux_i_density_energy += factor*(ff_fc_density_energyx+fc_i_density_energyx);
                flux_i_momentumx += factor*(ff_fc_momentum_xx + fc_i_momentum_xx);
                flux_i_momentumy += factor*(ff_fc_momentum_yx + fc_i_momentum_yx);
                flux_i_momentumz += factor*(ff_fc_momentum_zx + fc_i_momentum_zx);

                factor = double(0.5f)*normaly;
                flux_i_density += factor*(ff_variable[VAR_MOMENTUM+1]+momentum_iy);
                flux_i_density_energy += factor*(ff_fc_density_energyy+fc_i_density_energyy);
                flux_i_momentumx += factor*(ff_fc_momentum_xy + fc_i_momentum_xy);
                flux_i_momentumy += factor*(ff_fc_momentum_yy + fc_i_momentum_yy);
                flux_i_momentumz += factor*(ff_fc_momentum_zy + fc_i_momentum_zy);

                factor = double(0.5f)*normalz;

                flux_i_density += factor*(ff_variable[VAR_MOMENTUM+2]+momentum_iz);
                flux_i_density_energy += factor*(ff_fc_density_energyz+fc_i_density_energyz);
                flux_i_momentumx += factor*(ff_fc_momentum_xz + fc_i_momentum_xz);
                flux_i_momentumy += factor*(ff_fc_momentum_yz + fc_i_momentum_yz);
                flux_i_momentumz += factor*(ff_fc_momentum_zz + fc_i_momentum_zz);

            }
        }

        fluxes[i*NVAR + VAR_DENSITY] = flux_i_density;
        fluxes[i*NVAR + (VAR_MOMENTUM+0)] = flux_i_momentumx;
        fluxes[i*NVAR + (VAR_MOMENTUM+1)] = flux_i_momentumy;
        fluxes[i*NVAR + (VAR_MOMENTUM+2)] = flux_i_momentumz;
        fluxes[i*NVAR + VAR_DENSITY_ENERGY] = flux_i_density_energy;
    }
}

static inline void time_step(int j, int nelr, double* old_variables, double* variables, double* step_factors, double* fluxes)
{
//#pragma omp parallel for  default(shared) schedule(static)
    for(int i = 0; i < nelr; i++)
    {
        double factor = step_factors[i]/double(RK+1-j);

        variables[NVAR*i + VAR_DENSITY] = old_variables[NVAR*i + VAR_DENSITY] + factor*fluxes[NVAR*i + VAR_DENSITY];
        variables[NVAR*i + VAR_DENSITY_ENERGY] = old_variables[NVAR*i + VAR_DENSITY_ENERGY] + factor*fluxes[NVAR*i + VAR_DENSITY_ENERGY];
        variables[NVAR*i + (VAR_MOMENTUM+0)] = old_variables[NVAR*i + (VAR_MOMENTUM+0)] + factor*fluxes[NVAR*i + (VAR_MOMENTUM+0)];
        variables[NVAR*i + (VAR_MOMENTUM+1)] = old_variables[NVAR*i + (VAR_MOMENTUM+1)] + factor*fluxes[NVAR*i + (VAR_MOMENTUM+1)];
        variables[NVAR*i + (VAR_MOMENTUM+2)] = old_variables[NVAR*i + (VAR_MOMENTUM+2)] + factor*fluxes[NVAR*i + (VAR_MOMENTUM+2)];
    }
}
/*
 * Main function
 */
int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::cout << "specify input file and output file" << std::endl;
        return 0;
    }
    const char* data_file_name = argv[1];
    const char* out_file_name = argv[2];

    // set far field conditions
    {
        double angle_of_attack = double(3.1415926535897931 / 180.0f) * double(deg_angle_of_attack);

        ff_variable[VAR_DENSITY] = double(1.4);

        double ff_pressure = double(1.0f);
        double ff_speed_of_sound = sqrt(GAMMA*ff_pressure / ff_variable[VAR_DENSITY]);
        double ff_speed = double(ff_mach)*ff_speed_of_sound;

        double ff_velocityx;
        double ff_velocityy;
        double ff_velocityz;
        ff_velocityx = ff_speed*double(cos((double)angle_of_attack));
        ff_velocityy = ff_speed*double(sin((double)angle_of_attack));
        ff_velocityz = 0.0f;

        ff_variable[VAR_MOMENTUM+0] = ff_variable[VAR_DENSITY] * ff_velocityx;
        ff_variable[VAR_MOMENTUM+1] = ff_variable[VAR_DENSITY] * ff_velocityy;
        ff_variable[VAR_MOMENTUM+2] = ff_variable[VAR_DENSITY] * ff_velocityz;

        ff_variable[VAR_DENSITY_ENERGY] = ff_variable[VAR_DENSITY]*(double(0.5f)*(ff_speed*ff_speed)) + (ff_pressure / double(GAMMA-1.0f));

        double ff_momentumx;
        double ff_momentumy;
        double ff_momentumz;
        ff_momentumx = *(ff_variable+VAR_MOMENTUM+0);
        ff_momentumy = *(ff_variable+VAR_MOMENTUM+1);
        ff_momentumz = *(ff_variable+VAR_MOMENTUM+2);
        
        compute_flux_contribution(ff_variable[VAR_DENSITY], 
                ff_momentumx, ff_momentumy, ff_momentumz,
                ff_variable[VAR_DENSITY_ENERGY], ff_pressure,
                ff_velocityx, ff_velocityy, ff_velocityz,
                ff_fc_momentum_xx,ff_fc_momentum_xy,ff_fc_momentum_xz,
                ff_fc_momentum_yx,ff_fc_momentum_yy,ff_fc_momentum_yz,
                ff_fc_momentum_zx,ff_fc_momentum_zy,ff_fc_momentum_zz,
                ff_fc_density_energyx, ff_fc_density_energyy, ff_fc_density_energyz);
    }
    int nel;
    int nelr;

    // read in domain geometry
    double* areas;
    int* elements_surrounding_elements;
    double* normals;
    {
        std::ifstream file(data_file_name);

        file >> nel;
        nelr = block_length*((nel / block_length )+ std::min(1, nel % block_length));
        areas = MP_Malloc(nelr, areas);
        elements_surrounding_elements = new int[nelr*NNB];
        normals = MP_Malloc(NDIM*NNB*nelr, normals);

        // read in data
        for(int i = 0; i < nel; i++)
        {
            file >> areas[i];
            for(int j = 0; j < NNB; j++)
            {
                file >> elements_surrounding_elements[i*NNB + j];
                if(elements_surrounding_elements[i*NNB+j] < 0) elements_surrounding_elements[i*NNB+j] = -1;
                elements_surrounding_elements[i*NNB + j]--; //it's coming in with Fortran numbering

                for(int k = 0; k < NDIM; k++)
                {
                    file >>  normals[(i*NNB + j)*NDIM + k];
                    normals[(i*NNB + j)*NDIM + k] = -normals[(i*NNB + j)*NDIM + k];
                }
            }
        }

        // fill in remaining data
        int last = nel-1;
        for(int i = nel; i < nelr; i++)
        {
            areas[i] = areas[last];
            for(int j = 0; j < NNB; j++)
            {
                // duplicate the last element
                elements_surrounding_elements[i*NNB + j] = elements_surrounding_elements[last*NNB + j];
                for(int k = 0; k < NDIM; k++) normals[(i*NNB + j)*NDIM + k] = normals[(last*NNB + j)*NDIM + k];
            }
        }
    }

    // Create arrays and set initial conditions
    double* variables =  MP_Malloc(nelr*NVAR,variables);
    initialize_variables(nelr, variables);

    double* old_variables =  MP_Malloc(nelr*NVAR,old_variables);
    double* fluxes =  MP_Malloc(nelr*NVAR,fluxes);
    double* step_factors =  MP_Malloc(nelr,step_factors);
    double* fc_momentum_x =  MP_Malloc(nelr*NDIM,fc_momentum_x); 
    double* fc_momentum_y =  MP_Malloc(nelr*NDIM,fc_momentum_y);
    double* fc_momentum_z =  MP_Malloc(nelr*NDIM,fc_momentum_z);
    double* fc_density_energy =  MP_Malloc(nelr*NDIM,fc_density_energy);

    // these need to be computed the first time in order to compute time step
    startMeasure();
    // Begin iterations
    for(int i = 0; i < iterations; i++)
    {
        copy(old_variables, variables, nelr*NVAR);

        // for the first iteration we compute the time step
        compute_step_factor(nelr, variables, areas, step_factors);

        for(int j = 0; j < RK; j++)
        {
            compute_flux_contributions(nelr, variables, fc_momentum_x, fc_momentum_y, fc_momentum_z, fc_density_energy);
            compute_flux(nelr, elements_surrounding_elements, normals, variables, fc_momentum_x, fc_momentum_y, fc_momentum_z, fc_density_energy, fluxes);
            time_step(j, nelr, old_variables, variables, step_factors, fluxes);
        }
    }
    stopMeasure();

    std::cout << "Saving solution..." << std::endl;
    dump(variables, nel, nelr,out_file_name);
    std::cout << "Saved solution..." << std::endl;


    std::cout << "Cleaning up..." << std::endl;
    dealloc(areas);
    dealloc(elements_surrounding_elements);
    dealloc(normals);

    dealloc(variables);
    dealloc(old_variables);
    dealloc(fluxes);
    dealloc(step_factors);
    dealloc(fc_momentum_x); 
    dealloc(fc_momentum_y);
    dealloc(fc_momentum_z);
    dealloc(fc_density_energy);
    std::cout << "Done..." << std::endl;

    return 0;
}
