#ifndef GAUSSIANMIXTURE_H_
#define GAUSSIANMIXTURE_H_

#define _USE_MATH_DEFINES
#include <array>
#include <cmath>

struct BivGaussParams
{
  float weight;
  size_t n_k;

  float m_x;
  float m_y;
  float c_1;
  float c_23;
  float c_4;
};


template < size_t COMPONENTS >
class GaussianMixture
{
private:

	/**
	 * Gaussian parameters for all COMPONENTS
	 */
	std::array< BivGaussParams, COMPONENTS > gaussians;

	/**
	 * Number of added Datapoints
	 */
	size_t n;

  public:
	GaussianMixture( float s_x = 100.0f, float s_y = 100.0f ) :
		n( 0 )
	{
		// initial gauss gaussians
		for( unsigned k = 0; k < COMPONENTS; k++ ) {
		  BivGaussParams& g = gaussians[ k ];
		  g.weight = 0.0f;
		  g.n_k = 0;
		  g.m_x = 0.0f;
		  g.m_y = 0.0f;
		  g.c_1 = s_x;
		  g.c_4 = s_y;
		  g.c_23 = 0.0f;
		}
	}

	GaussianMixture( const GaussianMixture& other ) :
		gaussians( other.gaussians ),
		n( other.n )
	{}

	GaussianMixture& operator=( const GaussianMixture& other )
	{
		if( this != &other )
		{
		  gaussians = other.gaussians;
		  n = other.n;
		}
		return *this;
	}

    GaussianMixture& operator+=( const GaussianMixture& other )
	{
		if( this != &other )
		{
		  if( n == 0) {
			  *this = other;
		  } else if( other.n == 0 ) {
			  return *this;
		  }

		  std::array< BivGaussParams, COMPONENTS > gaussiansA = gaussians;
		  std::array< BivGaussParams, COMPONENTS > gaussiansB = other.gaussians;
		  std::sort( gaussiansA.begin(), gaussiansA.end(), gaussianCompare );
		  std::sort( gaussiansB.begin(), gaussiansB.end(), gaussianCompare );

		  auto itA = gaussiansA.begin();
		  auto itB = gaussiansB.begin();
		  n = 0;
		  float wSum = 0.0f;
		  for( unsigned k = 0; k < COMPONENTS; k++ ) {
			  if( itA->weight >= itB->weight ) {
				  gaussians[ k ] = *itA++;
			  } else {
				  gaussians[ k ] = *itB++;
			  }
			  n += gaussians[ k ].n_k;
			  wSum += gaussians[ k ].weight;
		  }

		  const float wSumInv = 1.0f / wSum;
		  for( unsigned k = 0; k < COMPONENTS; k++ ) {
				  gaussians[ k ].weight *= wSumInv;
		  }
		}
		return *this;
	}

	/**
	 * Bivariate normal distribution
	 */
	static float biv_normal_d(float x, float y, const BivGaussParams& bgp )
	{
		float covDet = bgp.c_1 * bgp.c_4 - bgp.c_23 * bgp.c_23;

		float a = 1.0f / (2.0f * M_PI * std::sqrt( covDet ) );
		float covDet_inv = 1.0f / covDet;
		float c_inv_1 = bgp.c_4 * covDet_inv;
		float c_inv_23 = - bgp.c_23 * covDet_inv;
		float c_inv_4 = bgp.c_1 * covDet_inv;

		float d_x = x - bgp.m_x;
		float d_y = y - bgp.m_y;

		float e = d_x * c_inv_1 + d_y * c_inv_23;
		float f = d_x * c_inv_23 + d_y * c_inv_4;

		float g = e * d_x + f * d_y;

		return a * std::exp( -0.5f * g );
	}

	/**
	 * Aproximate eliptic angle of gaussian
	 */
	static float elipse( const BivGaussParams& bgp )
	{
		const float b = - bgp.c_1 - bgp.c_4;
		const float c = bgp.c_1*bgp.c_4 - bgp.c_23*bgp.c_23;
		const float d = std::sqrt( b*b - 4.0f*c );
		const float m1 = ( - b + d ) / 2.0f;
		const float m2 = ( - b - d ) / 2.0f;

		return std::atan2( m1, m2 );
	}

	/**
	 * Add datapoint using online EM-Algorithm
	 */
	void addDatapoint( float x, float y )
	{
		if( n++ < COMPONENTS ) {

			addGaussian( x, y);

		} else {

			unsigned maxK = estep( x, y );
			mstep( x, y , maxK );

		}
	}

	/**
	 * Get number of Components 
	 */
	inline size_t size() const { return COMPONENTS; }

	/**
	 * Get number of added Datapoints
	 */
	inline size_t getN() const { return n; }

	/**
	 * Get gaussian parameters for k-th component
	 */
	const BivGaussParams& getGaussian( size_t k )
	{
		return gaussians[ k ];
	}

	/**
	 * Get weight parameter for k-th component
	 */
	float getWeight( size_t k )
	{
		return gaussians[ k ].weight;
	}

	/**
	 * Get number of added datapoints for k-th component
	 */
    size_t getNK( size_t k )
	{
		return gaussians[ k ].n_k;
	}


	/**
	 * Get mixture's mean postion
	 */
	std::pair< float, float > getMean() const
	{
		float m_x = 0.0f, m_y = 0.0f;
		for( unsigned k = 0; k < COMPONENTS; k++ ) {
		    const BivGaussParams& g = gaussians[ k ];
		    m_x += g.weight * g.m_x;
		    m_y += g.weight * g.m_y;
		}

		return std::pair< float, float >( m_x, m_y );
	}

	/**
	 * Get mixture's variances in x and y direction
	 */
	std::pair< float, float > getVariances() const
	{
	    const std::pair< float, float > means = getMean();

		float s_x = 0.0f, s_y = 0.0f;
		for( unsigned k = 0; k < COMPONENTS; k++ ) {
		    const BivGaussParams& g = gaussians[ k ];
			const float dx = ( g.m_x - means.first );
			const float dy = ( g.m_y - means.second );
			s_x += g.weight * ( dx*dx + g.c_1 );
			s_y += g.weight * ( dy*dy + g.c_4 );
		}

		return std::pair< float, float >( s_x, s_y );
	}

	/**
	 * Get mixture's mean variance (avg from x and y)
	 */
	float getCovMean() const
	{
	    // no weigths: return default cov
	    if( n == 0 ) return ( gaussians[ 0 ].c_1 + gaussians[ 0 ].c_4 ) * 0.5f;

		float s_x = 0.0f, s_y = 0.0f;
		for( unsigned k = 0; k < COMPONENTS; k++ ) {
		    const BivGaussParams& g = gaussians[ k ];
			s_x += g.weight * g.c_1;
			s_y += g.weight * g.c_4;
		}

		return ( s_x + s_y ) * 0.5f;
	}



	/**
	 * Get mixture's Shannon Entropy
	 */
	float getShannonEntropy() const
	{
		float sum = 0.0f;
		for( unsigned k = 0; k < COMPONENTS; k++ ) {
		    const float& weight = gaussians[ k ].weight;
			if( weight > 0.0001f ) {
				sum -= weight * std::log2( weight );
			}
		}
		return sum;
	}

	/**
	* Print paramters
	*/
	friend std::ostream& operator<<( std::ostream& os, const GaussianMixture<T>& gm )
	{
		for( unsigned k = 0; k < gn.size(); k++ ) {
			const BivGaussParams& g = gm.getGaussian( k );
			os << "[Component " << k << " " << g.weight << "]" <<
			" mx: " << g.m_x << " my: " << g.m_y <<
			" c_1: " << g.c_1 << " c_4: " << g.c_4 << " c_23: " << g.c_23;
		}
	}

  private:

	/**
	 * E step of EM-Algorithm
	 */
	unsigned estep(float x, float y) const
	{
		std::array< float, COMPONENTS> distributions;
	    float distributionSum = 0.0f;

	    for( unsigned k = 0; k < COMPONENTS; k++ ) {
	        const float p = biv_normal_d( x, y, gaussians[ k ] ) * gaussians[ k ].weight;
	        distributions[ k ] = p;
	        distributionSum += p;
	    }

	    float maxWeight = 0.0f;
	    unsigned maxK = 0;

	    for( unsigned k = 0; k < COMPONENTS; k++ ) {
	        const float weight = distributions[ k ] / distributionSum;
	        if( weight > maxWeight ) {
	            maxWeight = weight;
	            maxK = k;
	        }
	    }

	    return maxK;

	}

	/**
	 * M step of EM-Algorithm
	 */
	void mstep( float x, float y, unsigned maxK )
	{
	    BivGaussParams& gK = gaussians[ maxK ];
	    gK.n_k++;

		const float n_inv = 1.0f / n;

		// update weights for all k
		for( unsigned k = 0; k < COMPONENTS; k++ ) {
	        const float zk = (k == maxK);
	        gaussians[ k ].weight += ( zk - gaussians[ k ].weight ) * n_inv;
		}

		// update gauss gaussians
	    const float dx = x - gK.m_x;
	    const float dy = y - gK.m_y;
	    const float nk_inv = 1.0f / gK.n_k;

		// update means
	    gK.m_x += dx * nk_inv;
	    gK.m_y += dy * nk_inv;

	    // update covariance
	    const float a = 1.0f - nk_inv;
	    const float c_1  = a * dx * dx - gK.c_1;
	    const float c_4  = a * dy * dy - gK.c_4;
	    const float c_23 = a * dx * dy - gK.c_23;

	    gK.c_1 += nk_inv * c_1;
	    gK.c_4 += nk_inv * c_4;
	    gK.c_23 += nk_inv * c_23;

	}

	/**
	 * For the first data points,
	 * center gaussians on them
	 */
	void addGaussian( float m_x, float m_y )
 	{

		const size_t i = n - 1;

		if( i >= COMPONENTS ) return;

		BivGaussParams& g = gaussians[ i ];
		g.m_x = m_x;
		g.m_y = m_y;
		g.n_k++;

		const float n_inv = 1.0f / n;
		for( unsigned k = 0; k < n; k++ ) {
			gaussians[ k ].weight = n_inv;
		}
	}

	/**
	 * Sort helper
	 */
	static bool gaussianCompare(BivGaussParams const& a, BivGaussParams const& b)
 	{
	    return a.weight > b.weight;
	}

};



#endif
