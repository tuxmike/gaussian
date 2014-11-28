#include <cvt/gfx/Image.h>
#include <cvt/gfx/GFXEngineImage.h>
#include <cvt/gfx/IExpr.h>

#define _USE_MATH_DEFINES
#include <cmath>
#include <random>
#include <iostream>
#include <vector>

#define COMPONENTS 5

struct BivGaussParams
{
  float m_x;
  float m_y;
  float c_1;
  float c_23;
  float c_4;
};

typedef std::array<float, COMPONENTS> ArrFloatComp;


/**
 * Bivariate normal distribution
 */
float biv_normal_d(float x, float y, const BivGaussParams& bgp )
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


unsigned estep( const ArrFloatComp& weights,
            const std::array< BivGaussParams, COMPONENTS >& params,
            const std::vector< cvt::Vector2i >& dataset )
{
    ArrFloatComp distributions;
    float distributionSum = 0.0f;

    for( unsigned k = 0; k < COMPONENTS; k++ ) {
        const cvt::Vector2i& data = dataset.back();
        const float p = biv_normal_d( data.x, data.y, params[ k ] ) * weights [ k ];
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

void mstep( std::array< size_t, COMPONENTS>& n_k,
            ArrFloatComp& weights,
			std::array< BivGaussParams, COMPONENTS >& params,
			const std::vector< cvt::Vector2i >& dataset,
			unsigned maxK )
{
	n_k[ maxK ]++;

	const float n_inv = 1.0f / std::max( dataset.size(), 5UL);

	// update weights for all k
	for( unsigned k = 0; k < COMPONENTS; k++ ) {
        const float zk = (k == maxK);
        weights[ k ] += ( zk - weights[ k ] ) * n_inv;
	}

	// update gauss params
	BivGaussParams& paramsK = params[ maxK ];
    const float dx = dataset.back().x - paramsK.m_x;
    const float dy = dataset.back().y - paramsK.m_y;
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

void drawData(cvt::Image& img, std::vector< cvt::Vector2i >& points) {
	cvt::GFXEngineImage gi( img );
	cvt::GFX gfx( &gi );
	gfx.setColor( cvt::Color::WHITE );

	for( auto it = points.begin(); it < points.end(); ++it )
	{
		gfx.drawLine( it->x - 1, it->y - 1, it->x + 1, it->y + 1 );
		gfx.drawLine( it->x + 1, it->y - 1, it->x - 1, it->y + 1 );
	}
}

void drawGaussians(cvt::Image& img, std::array< BivGaussParams, COMPONENTS >& gaussians, ArrFloatComp& weights) {
	cvt::IMapScoped< float > imgMap( img );

	for( unsigned k = 0; k < gaussians.size(); k++ ) {
		for( unsigned y = 0; y < img.height(); y++ ) {
			for( unsigned x = 0; x < img.width(); x++ ) {
				float val = biv_normal_d( x, y, gaussians[ k ] ) * weights [ k ];
				imgMap( x, y ) += 255.0f * val;
			}
		}
	}
}

int main(int argc, char *argv[])
{

  auto const seed = std::random_device()();
  std::mt19937 rand_generator = std::mt19937 ( seed );
  std::uniform_int_distribution< int > d_50_150( 50, 150 );
  std::uniform_int_distribution< int > d_25_75( 25, 75 );
  std::bernoulli_distribution distribution2( 0.5f );

  std::vector< cvt::Vector2i > dataset;

  std::array< float, COMPONENTS> weights;
  std::array< size_t, COMPONENTS> n_k;
  std::array< BivGaussParams, COMPONENTS > params;

  // some start configuration...
  weights.fill( 1.0f / COMPONENTS );
  n_k.fill( 1 );

  for( unsigned k = 0; k < COMPONENTS; k++ ) {
	  BivGaussParams& gaussParam = params[ k ];
	  gaussParam.m_x = d_50_150( rand_generator );
	  gaussParam.m_y = d_50_150( rand_generator );
	  gaussParam.c_1 = 200.0f;
	  gaussParam.c_4 = 200.0f;
	  gaussParam.c_23 = 0.0f;
  }

  unsigned iteration = 0;

  while ( iteration++ < 100 )
  {
      // add data point
      int upDown = ( distribution2( rand_generator ) ) ? 0 : 100;
      cvt::Vector2i vec( d_50_150( rand_generator ), d_25_75( rand_generator ) + upDown );
      dataset.push_back( vec );

      // run E M
	  unsigned maxK = estep( weights, params, dataset );
	  mstep( n_k, weights, params, dataset, maxK );

	  // show results
	  if(iteration % 5 == 0)
	  {
          cvt::Image img( 200, 200, cvt::IFormat::GRAY_FLOAT );
          img.fill( cvt::Color::BLACK );
          drawData( img, dataset );
          drawGaussians( img, params, weights );

          cvt::String file;
          file.sprintf( "out%i.png", iteration );
          img.save( file );

          std::cout << "Weights:" << weights[0] << "," << weights[1] << "," << weights[2] << "," << weights[3] << "," << weights[4] << std::endl;
          std::cout << "Params:" << params[0].m_x << "," << params[0].m_y << "," << params[0].c_1 << "," << params[0].c_4 << "," << params[0].c_23 << std::endl;
	  }
  }

  return 0;
}
