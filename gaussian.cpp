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
  float s_x;
  float s_y;
  float corr;
};

typedef std::array<float, COMPONENTS> CompArr;


/**
 * Bivariate normal distribution
 */
float biv_normal_d(float x, float y, const BivGaussParams& bgp )
{

    float a = 1.0f - ( bgp.corr * bgp.corr );
    float b = 1.0f / (2.0f * M_PI * std::sqrt( a ) );
    float c = -1.0f / ( 2 * a);
    float d = ( x - bgp.m_x );
    float e = ( y - bgp.m_y );
    float f = ( d*d ) / ( bgp.s_x * bgp.s_x );
    float g = ( e*e ) / ( bgp.s_y * bgp.s_y );
    float h = ( 2.0f * bgp.corr * d * e ) / ( bgp.s_x * bgp.s_y );
    float i = c * ( f + g - h );
    float j = b * std::exp( i );
    return b * std::exp( i );
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
	for( unsigned k = 0; k < COMPONENTS; k++ ) {
		for( unsigned i = 0; i < dataset.size(); i++ ) {
			n_k[ k ] += memberWeights[ i ][ k ];
		}
		weights[ k ] = n_k[ k ] / dataset.size();
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

		paramsK.s_x = std::sqrt( n_k_inv * sx_2);
		paramsK.s_y = std::sqrt( n_k_inv * sy_2);
		paramsK.corr = p_sx_sy / ( paramsK.s_x * paramsK.s_y );

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
				imgMap( x, y ) += val;
			}
		}
	}
}

int main(int argc, char *argv[])
{

  /*
  std::vector< cvt::Vector2i > dataset = { cvt::Vector2i( 10, 8), cvt::Vector2i(12, 8), cvt::Vector2i(12, 12),
		  cvt::Vector2i(4, 16), cvt::Vector2i(2, 16), cvt::Vector2i(6, 14), cvt::Vector2i(8, 9), cvt::Vector2i(10, 10),
		  cvt::Vector2i(24, 8), cvt::Vector2i(16, 8), cvt::Vector2i(46, 10), cvt::Vector2i(10, 16), cvt::Vector2i(8, 4),
		  cvt::Vector2i(6, 15), cvt::Vector2i(6, 17), cvt::Vector2i(2, 5), cvt::Vector2i(13, 17), cvt::Vector2i(7, 9),
		  cvt::Vector2i(18, 6), cvt::Vector2i(4, 2), cvt::Vector2i(14, 7), cvt::Vector2i(10, 12), cvt::Vector2i(16, 12) };
  */

  auto const seed = std::random_device()();
  std::mt19937 rand_generator = std::mt19937 ( seed );
  std::uniform_int_distribution< int > distribution( 50, 150 );

  std::vector< cvt::Vector2i > dataset;

  for( unsigned i = 0; i < 100; i++ ) {
	  cvt::Vector2i vec( distribution( rand_generator ), distribution( rand_generator ) );
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
	  gaussParam.s_x = 10.0f;
	  gaussParam.s_y = 10.0f;
	  gaussParam.corr = 0.0f;
  }

  unsigned iteration = 1;

  std::cout << "Weights:" << weights[0] << "," << weights[1] << "," << weights[2] << "," << weights[3] << "," << weights[4] << std::endl;
  std::cout << "Params:" << params[0].m_x << "," << params[0].m_y << "," << params[0].s_x << "," << params[0].s_y << "," << params[0].corr << std::endl;

  while ( iteration++ < 2 )
  {
	  estep( weights, memberWeights, params, dataset );
	  mstep( weights, memberWeights, params, dataset );

	  std::cout << "Weights:" << weights[0] << "," << weights[1] << "," << weights[2] << "," << weights[3] << "," << weights[4] << std::endl;
	  std::cout << "Params:" << params[0].m_x << "," << params[0].m_y << "," << params[0].s_x << "," << params[0].s_y << "," << params[0].corr << std::endl;
  }

  cvt::Image img( 200, 200, cvt::IFormat::GRAY_FLOAT );
  img.fill( cvt::Color::BLACK );
  drawData( img, dataset );
  drawGaussians( img, params, weights );

  img.save( "out.png" );


  return 0;
}
