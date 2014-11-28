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

typedef std::array<float, COMPONENTS> CompArr;


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


void estep( const CompArr& weights,
			std::vector< CompArr >& memberWeights,
			const std::array< BivGaussParams, COMPONENTS >& params,
			const std::vector< cvt::Vector2i >& dataset )
{
	for( unsigned i = 0; i < dataset.size(); i++ ) {

		CompArr distributions;
		float distributionSum = 0.0f;

		for( unsigned k = 0; k < COMPONENTS; k++ ) {
			const cvt::Vector2i& data = dataset[ i ];
			const float p = biv_normal_d( data.x, data.y, params[ k ] ) * weights [ k ];
			distributions[ k ] = p;
			distributionSum += p;
		}

		for( unsigned k = 0; k < COMPONENTS; k++ ) {
			memberWeights[ i ][ k ] = distributions[ k ] / distributionSum;
		}
	}
}

void mstep( CompArr& weights,
			std::vector< CompArr >& memberWeights,
			std::array< BivGaussParams, COMPONENTS >& params,
			const std::vector< cvt::Vector2i >& dataset )
{
	CompArr n_k;
	n_k.fill( 0.0f );

	// update weights
	const float n = dataset.size();
	for( unsigned k = 0; k < COMPONENTS; k++ ) {
		for( unsigned i = 0; i < dataset.size(); i++ ) {
			n_k[ k ] += memberWeights[ i ][ k ];
		}
		weights[ k ] = n_k[ k ] / n;
	}

	// update gauss params
	for( unsigned k = 0; k < COMPONENTS; k++ ) {

		BivGaussParams& paramsK = params[ k ];
		const float n_k_inv = 1.0f / n_k[ k ];

		// update means
		float m_x = 0.0f;
		float m_y = 0.0f;

		for( unsigned i = 0; i < dataset.size(); i++ ) {
			m_x += memberWeights[ i ][ k ] * dataset[ i ].x;
			m_y += memberWeights[ i ][ k ] * dataset[ i ].y;
		}


		paramsK.m_x = n_k_inv * m_x;
		paramsK.m_y = n_k_inv * m_y;

		// update covariance
		float sx_2 = 0.0f;
		float sy_2 = 0.0f;
		float p_sx_sy = 0.0f;

		for( unsigned i = 0; i < dataset.size(); i++ ) {
			const float dx = dataset[ i ].x - paramsK.m_x;
			const float dy = dataset[ i ].y - paramsK.m_y;
			const float w_i_k = memberWeights[ i ][ k ];

			sx_2 += dx * dx * w_i_k;
			sy_2 += dy * dy * w_i_k;
			p_sx_sy += dx * dy * w_i_k;
		}

		paramsK.c_1 = n_k_inv * sx_2;
		paramsK.c_4 = n_k_inv * sy_2;
		paramsK.c_23 = n_k_inv * p_sx_sy;

	}

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

void drawGaussians(cvt::Image& img, std::array< BivGaussParams, COMPONENTS >& gaussians, CompArr& weights) {
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

  for( unsigned i = 0; i < 100; i++ ) {
      int upDown = ( distribution2( rand_generator ) ) ? 0 : 100;
      cvt::Vector2i vec( d_50_150( rand_generator ), d_25_75( rand_generator ) + upDown );
	  dataset.push_back( vec );
  }

  const float initWeight = 1.0f / COMPONENTS;

  CompArr weights;
  weights.fill( initWeight );

  std::vector< CompArr > memberWeights( dataset.size(), weights);

  std::array< BivGaussParams, COMPONENTS > params;

  // some start configuration...
  for( unsigned k = 0; k < COMPONENTS; k++ ) {
	  BivGaussParams& gaussParam = params[ k ];
	  gaussParam.m_x = dataset[ k ].x;
	  gaussParam.m_y = dataset[ k ].y;
	  gaussParam.c_1 = 100.0f;
	  gaussParam.c_4 = 100.0f;
	  gaussParam.c_23 = 0.0f;
  }

  unsigned iteration = 0;

  while ( iteration++ < 20 )
  {
      cvt::Image img( 200, 200, cvt::IFormat::GRAY_FLOAT );
      img.fill( cvt::Color::BLACK );
      drawData( img, dataset );
      drawGaussians( img, params, weights );

      cvt::String file;
      file.sprintf( "out%i.png", iteration );
      img.save( file );

      std::cout << "Weights:" << weights[0] << "," << weights[1] << "," << weights[2] << "," << weights[3] << "," << weights[4] << std::endl;
      std::cout << "Params:" << params[0].m_x << "," << params[0].m_y << "," << params[0].c_1 << "," << params[0].c_1 << "," << params[0].c_23 << std::endl;
      std::cout << "MWeights:" << memberWeights[0][0] << "," << memberWeights[0][1] << "," << memberWeights[0][2] << "," << memberWeights[0][3] << "," << memberWeights[0][4] << std::endl;

	  estep( weights, memberWeights, params, dataset );
	  mstep( weights, memberWeights, params, dataset );

  }

  return 0;
}
