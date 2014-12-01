#ifndef GAUSSIANMIXTURE_H_
#define GAUSSIANMIXTURE_H_

#define _USE_MATH_DEFINES
#include <array>
#include <cmath>

/*
 * For drawing only
 */
#include <cvt/gfx/Image.h>
#include <cvt/gfx/GFXEngineImage.h>
#include <cvt/gfx/IExpr.h>


template < size_t COMPONENTS >
class GaussianMixture
{
private:

	struct BivGaussParams
	{
	  float m_x;
	  float m_y;
	  float c_1;
	  float c_23;
	  float c_4;
	};

	std::array< float, COMPONENTS> weights;
	std::array< size_t, COMPONENTS> n_k;
	std::array< BivGaussParams, COMPONENTS > params;
	size_t n;

  public:
	GaussianMixture( float s_x = 100.0f, float s_y = 100.0f ):
		n( 0 )
    {
		weights.fill( 0.0f );
		n_k.fill( 0 );

		// initial gauss params
		for( unsigned k = 0; k < COMPONENTS; k++ ) {
		  BivGaussParams& gaussParam = params[ k ];
		  gaussParam.m_x = 0.0f;
		  gaussParam.m_y = 0.0f;
		  gaussParam.c_1 = s_x;
		  gaussParam.c_4 = s_y;
		  gaussParam.c_23 = 0.0f;
		}
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

	void drawGaussians(cvt::Image& img) const {
		cvt::IMapScoped< float > imgMap( img );

		for( unsigned k = 0; k < COMPONENTS; k++ ) {
			for( unsigned y = 0; y < img.height(); y++ ) {
				for( unsigned x = 0; x < img.width(); x++ ) {
					float val = biv_normal_d( x, y, params[ k ] ) * weights [ k ];
					imgMap( x, y ) += 255.0f * val;
				}
			}
		}
	}

	void outputParams() {
		for( unsigned k = 0; k < COMPONENTS; k++ ) {
			std::cout << "[Component " << k << " " << weights[ k ] << "]" <<
			" mx: " << params[ k ].m_x << " my: " << params[ k ].m_y <<
			" c_1: " << params[ k ].c_1 << " c_4: " << params[ k ].c_4 << " c_23: " << params[ k ].c_23 <<
			std::endl;
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
	        const float p = biv_normal_d( x, y, params[ k ] ) * weights [ k ];
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
		n_k[ maxK ]++;

		const float n_inv = 1.0f / n;

		// update weights for all k
		for( unsigned k = 0; k < COMPONENTS; k++ ) {
	        const float zk = (k == maxK);
	        weights[ k ] += ( zk - weights[ k ] ) * n_inv;
		}

		// update gauss params
		BivGaussParams& paramsK = params[ maxK ];
	    const float dx = x - paramsK.m_x;
	    const float dy = y - paramsK.m_y;
	    const float nk_inv = 1.0f / n_k[ maxK ];

		// update means
	    paramsK.m_x += dx * nk_inv;
	    paramsK.m_y += dy * nk_inv;

	    // update covariance
	    const float a = 1.0f - nk_inv;
	    const float c_1  = a * dx * dx - paramsK.c_1;
	    const float c_4  = a * dy * dy - paramsK.c_4;
	    const float c_23 = a * dx * dy - paramsK.c_23;

	    paramsK.c_1 += nk_inv * c_1;
	    paramsK.c_4 += nk_inv * c_4;
	    paramsK.c_23 += nk_inv * c_23;

	}

	/**
	 * For the first data points,
	 * center gaussians on them
	 */
	void addGaussian( float m_x, float m_y ) {

		const size_t gaussian = n - 1;

		if( gaussian >= COMPONENTS ) return;

		params[ gaussian ].m_x = m_x;
		params[ gaussian ].m_y = m_y;
		n_k[ gaussian ]++;

		const float n_inv = 1.0f / n;
		for( unsigned k = 0; k < n; k++ ) {
			weights[ k ] = n_inv;
		}
	}

};



#endif
