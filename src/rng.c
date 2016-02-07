/*! \file rng.c
 *  \brief Function definitions for easy random number generation */
#include "rng.h"
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

/*! \var int flag_rng_gaussian
 *  \brief Initialization flag for Gaussian RNG */
int flag_rng_gaussian;

/*! \var int flag_rng_uniform
 *  \brief Initialization flag for uniform RNG */
int flag_rng_uniform;

/*! \var int flag_rng_exponential
 *  \brief Initialization flag for exponential RNG */
int flag_rng_exponential;

/*! \var int flag_rng_poisson
 *  \brief Initialization flag for poisson RNG */
int flag_rng_poisson;

/*! \var int flag_rng_direction
 *  \brief Initialization flag for direction RNG */
int flag_rng_direction;

/*! \var int flag_rng_integer
 *  \brief Initialization flag for direction RNG */
int flag_rng_integer;

/*! \var int flag_rng_permutation
 *  \brief Initialization flag for permutation RNG*/
int flag_rng_permutation;

/*! \var int flag_rng_tdist
 *  \brief Initialization flag for tdist RNG*/
int flag_rng_tdist;

/*! \var int flag_rng_levy
 *  \brief Initialization flag for levy RNG*/
int flag_rng_levy;

/*! \var const gsl_rng_type *T_rng_gaussian
 * \brief RNG type for Gaussian PDF */
const gsl_rng_type *T_rng_gaussian;

/*! \var const gsl_rng_type *T_rng_uniform;
 * \brief RNG type for uniform PDF */
const gsl_rng_type *T_rng_uniform;

/*! \var const gsl_rng_type *T_rng_exponential;
 * \brief RNG type for exponential PDF */
const gsl_rng_type *T_rng_exponential;


/*! \var const gsl_rng_type *T_rng_poisson;
 * \brief RNG type for poisson PDF */
const gsl_rng_type *T_rng_poisson;

/*! \var const gsl_rng_type *T_rng_direction;
 * \brief RNG type for direction PDF */
const gsl_rng_type *T_rng_direction;

/*! \var const gsl_rng_type *T_rng_integer;
 * \brief RNG type for integer PDF */
const gsl_rng_type *T_rng_integer;


/*! \var const gsl_rng_type *T_rng_permutation;
 * \brief RNG type for permutation PDF */
const gsl_rng_type *T_rng_permutation;

/*! \var const gsl_rng_type *T_rng_tdist;
 * \brief RNG type for t-distribution PDF */
const gsl_rng_type *T_rng_tdist;

/*! \var const gsl_rng_type *T_rng_levy;
 * \brief RNG type for a Levy distribution PDF */
const gsl_rng_type *T_rng_levy;

/*! \var gsl_rng *r_rng_gaussian
 *  \brief Gaussian RNG */
gsl_rng *r_rng_gaussian;

/*! \var gsl_rng *r_rng_uniform
 *  \brief Uniform RNG */
gsl_rng *r_rng_uniform;

/*! \var gsl_rng *r_rng_exponential
 *  \brief Exponential RNG */
gsl_rng *r_rng_exponential;

/*! \var gsl_rng *r_rng_poisson
 *  \brief Poisson RNG */
gsl_rng *r_rng_poisson;

/*! \var gsl_rng *r_rng_integer
 *  \brief Integer RNG */
gsl_rng *r_rng_integer;

/*! \var gsl_rng *r_rng_direction
 *  \brief Direction RNG */
gsl_rng *r_rng_direction;

/*! \var gsl_rng *r_rng_permutation
 *  \brief Permutation RNG */
gsl_rng *r_rng_permutation;

/*! \var gsl_permuation *p_rng_permutation
 *  \brief Permutation */
gsl_permutation *p_rng_permutation;

/*! \var gsl_rng *r_rng_tdist
 *  \brief t-distribution RNG */
gsl_rng *r_rng_tdist;

/*! \var gsl_rng *r_rng_levy
 *  \brief Levy-distribution RNG */
gsl_rng *r_rng_levy;

/*! \fn double rng_gaussian(double mu, double sigma)
 *  \brief Returns a Gaussianly-distributed random number with mean mu and dispersion sigma */
double rng_gaussian(double mu, double sigma)
{
	if(flag_rng_gaussian!=1337)
		initialize_rng_gaussian();
	return gsl_ran_gaussian(r_rng_gaussian, sigma) + mu;
}

/*! \fn void initialize_rng_gaussian(void)
 *  \brief Initializes Gaussian random number generator */
void initialize_rng_gaussian(void)
{
	flag_rng_gaussian = 1337;

	gsl_rng_env_setup();

	T_rng_gaussian = gsl_rng_default;
	r_rng_gaussian = gsl_rng_alloc(T_rng_gaussian);
	
}

/*! \fn double rng_uniform(double a, double b)
 *  \brief Returns a uniformly-distributed random number between a and b*/
double rng_uniform(double a, double b)
{
	if(flag_rng_uniform!=1337)
		initialize_rng_uniform();
	return gsl_ran_flat(r_rng_uniform, a, b);
}

/*! \fn double rng_exponential(double mu)
 *  \brief Returns a exponentially-distributed random number with scale mu*/
double rng_exponential(double mu)
{
	if(flag_rng_exponential!=1337)
		initialize_rng_exponential();
	return gsl_ran_exponential(r_rng_exponential, mu);
}

/*! \fn int rng_poisson(double mu)
 *  \brief Returns a poisson-distributed random number with ave mu*/
int rng_poisson(double mu)
{
	if(flag_rng_poisson!=1337)
		initialize_rng_poisson();
	return gsl_ran_poisson(r_rng_poisson, mu);
}

/*! \fn double rng_tdist(double nu)
 *  \brief Returns a t-distributed random number with parameter nu*/
double rng_tdist(double nu)
{
	if(flag_rng_tdist!=1337)
		initialize_rng_tdist();
	return gsl_ran_tdist(r_rng_tdist, nu);
}

/*! \fn double rng_levy(double nu)
 *  \brief Returns a Levy-distributed random number with parameter nu*/
double rng_levy(double nu)
{
	if(flag_rng_levy!=1337)
		initialize_rng_levy();
	return gsl_ran_levy(r_rng_levy, 1., nu);
}


/*! \fn size_t *rng_permutation(int n)
 *  \brief Returns [0,n-1], randomly permutated*/
size_t *rng_permutation(int n)
{
	size_t *permutation;
	if(flag_rng_permutation!=1337)
		initialize_rng_permutation(n);

	if(flag_rng_permutation==1337)
		if(n!=p_rng_permutation->size)
			initialize_rng_permutation(n);
	

	//permutation = calloc_size_t_array(n);

	//for(int i=0;i<n;i++)
	//	permutation[i] = i;

	//gsl_permute(p_rng_permutation, permutation, 1, n);

	//return permutation;

	gsl_ran_shuffle(r_rng_permutation,p_rng_permutation->data, n, sizeof(size_t));
	return gsl_permutation_data(p_rng_permutation);
}

/*! \fn void initialize_rng_permutation(void)
 *  \brief Initializes integer random permutation generator*/
void initialize_rng_permutation(int n)
{
	if(flag_rng_permutation!=1337)
	{
		flag_rng_permutation = 1337;
		gsl_rng_env_setup();
		T_rng_permutation = gsl_rng_taus;
		r_rng_permutation = gsl_rng_alloc(T_rng_permutation);

		p_rng_permutation = gsl_permutation_alloc(n);
	}else{
		if(n!=p_rng_permutation->size)
		{
			gsl_permutation_free(p_rng_permutation);
			p_rng_permutation = gsl_permutation_alloc(n);
		}
	}

	gsl_permutation_init(p_rng_permutation);
	gsl_ran_shuffle(r_rng_permutation,p_rng_permutation->data, n, sizeof(size_t));

}


/*! \fn int rng_integer(int n)
 *  \brief Returns a uniformly-distributed random number between a and b*/
int rng_integer(int n)
{
	if(flag_rng_integer!=1337)
		initialize_rng_integer();
	return gsl_rng_uniform_int(r_rng_integer, n);
}


/*! \fn void initialize_rng_integer(void)
 *  \brief Initializes integer random number generator */
void initialize_rng_integer(void)
{
	flag_rng_integer = 1337;

	gsl_rng_env_setup();

	T_rng_integer = gsl_rng_taus;
	r_rng_integer = gsl_rng_alloc(T_rng_integer);
}


/*! \fn void initialize_rng_uniform(void)
 *  \brief Initializes uniform random number generator */
void initialize_rng_uniform(void)
{
	flag_rng_uniform = 1337;

	gsl_rng_env_setup();

	//T_rng_uniform = gsl_rng_default;
	T_rng_uniform = gsl_rng_taus;
	r_rng_uniform = gsl_rng_alloc(T_rng_uniform);
}

/*! \fn void initialize_rng_exponential(void)
 *  \brief Initializes exponential random number generator */
void initialize_rng_exponential(void)
{
	flag_rng_exponential = 1337;

	gsl_rng_env_setup();

	T_rng_exponential = gsl_rng_taus;
	r_rng_exponential = gsl_rng_alloc(T_rng_exponential);
}

/*! \fn void initialize_rng_poisson(void)
 *  \brief Initializes poisson random number generator */
void initialize_rng_poisson(void)
{
	flag_rng_poisson = 1337;

	gsl_rng_env_setup();

	T_rng_poisson = gsl_rng_taus;
	r_rng_poisson = gsl_rng_alloc(T_rng_poisson);
}

/*! \fn void initialize_rng_tdist(void)
 *  \brief Initializes tdist random number generator */
void initialize_rng_tdist(void)
{
	flag_rng_tdist = 1337;

	gsl_rng_env_setup();

	T_rng_tdist = gsl_rng_taus;
	r_rng_tdist = gsl_rng_alloc(T_rng_tdist);
}

/*! \fn void initialize_rng_levy(void)
 *  \brief Initializes levy random number generator */
void initialize_rng_levy(void)
{
	flag_rng_levy = 1337;

	gsl_rng_env_setup();

	T_rng_levy = gsl_rng_taus;
	r_rng_levy = gsl_rng_alloc(T_rng_levy);
}



/*! \fn void initialize_rng_direction(void)
 *  \brief Initializes direction random number generator */
void initialize_rng_direction(void)
{
	flag_rng_direction = 1337;

	gsl_rng_env_setup();

	//T_rng_direction = gsl_rng_default;
	T_rng_direction = gsl_rng_taus;
	r_rng_direction = gsl_rng_alloc(T_rng_direction);
}


/*! \fn void set_rng_uniform_seed(int seed)
 *  \brief Set uniform rng seed */
void set_rng_uniform_seed(int seed)
{
	if(flag_rng_uniform!=1337)
		initialize_rng_uniform();
	gsl_rng_set(r_rng_uniform, seed);
}

/*! \fn void set_rng_exponential_seed(int seed)
 *  \brief Set exponential rng seed */
void set_rng_exponential_seed(int seed)
{
	if(flag_rng_exponential!=1337)
		initialize_rng_exponential();
	gsl_rng_set(r_rng_exponential, seed);
}

/*! \fn void set_rng_poisson_seed(int seed)
 *  \brief Set poisson rng seed */
void set_rng_poisson_seed(int seed)
{
	if(flag_rng_poisson!=1337)
		initialize_rng_poisson();
	gsl_rng_set(r_rng_poisson, seed);
}

/*! \fn void set_rng_tdist_seed(int seed)
 *  \brief Set tdist rng seed */
void set_rng_tdist_seed(int seed)
{
	if(flag_rng_tdist!=1337)
		initialize_rng_tdist();
	gsl_rng_set(r_rng_tdist, seed);
}

/*! \fn void set_rng_levy_seed(int seed)
 *  \brief Set levy rng seed */
void set_rng_levy_seed(int seed)
{
	if(flag_rng_levy!=1337)
		initialize_rng_levy();
	gsl_rng_set(r_rng_levy, seed);
}

/*! \fn void set_rng_gaussian_seed(int seed)
 *  \brief Set gaussian rng seed */
void set_rng_gaussian_seed(int seed)
{
	if(flag_rng_gaussian!=1337)
		initialize_rng_gaussian();
	gsl_rng_set(r_rng_gaussian, seed);
}

/*! \fn void set_rng_direction_seed(int seed)
 *  \brief Set direction rng seed */
void set_rng_direction_seed(int seed)
{
	if(flag_rng_direction!=1337)
		initialize_rng_direction();
	gsl_rng_set(r_rng_direction, seed);
}

/*! \fn double *rng_direction(int ndim)
 *  \brief Get a random unit vector */
double *rng_direction(int ndim)
{
	double *x;
	if(flag_rng_direction!=1337)
		initialize_rng_direction();
  
	if(ndim<2)
	{
		printf("ndim must be > 1\n");
		return NULL;
	}
	//x = calloc_double_array(ndim);
	if(!(x = (double *) calloc(ndim,sizeof(double))))
	{
		printf("Error allocating x of size %d\n",ndim);
		return NULL;
	}
	switch(ndim)
	{
		case 2: gsl_ran_dir_2d(r_rng_direction,&x[0],&x[1]);
			break;
		case 3: gsl_ran_dir_3d(r_rng_direction,&x[0],&x[1],&x[2]);
			break;
		default: gsl_ran_dir_nd(r_rng_direction,ndim,x);
	}
	return x;
}
