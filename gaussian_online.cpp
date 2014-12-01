#include <cvt/gfx/Image.h>
#include <cvt/gfx/GFXEngineImage.h>
#include <cvt/gfx/IExpr.h>

#include <random>
#include <iostream>
#include <vector>

#include "GaussianMixture.h"

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


int main(int argc, char *argv[])
{

  auto const seed = std::random_device()();
  std::mt19937 rand_generator = std::mt19937 ( seed );
  std::normal_distribution< float > n_0_10( 0.0f , 10.0f );
  std::bernoulli_distribution bern_05( 0.5f );

  std::vector< cvt::Vector2i > dataset;

  GaussianMixture< 5 > gm;

  unsigned iteration = 0;

  while ( iteration++ < 200 )
  {
      // add data point
      int upDown = ( bern_05( rand_generator ) ) ? 70 : 130;
      cvt::Vector2i vec( n_0_10( rand_generator ) + 100 , n_0_10( rand_generator ) + upDown );
      dataset.push_back( vec );

      gm.addDatapoint( vec.x, vec.y );

	  // show results
	  if(iteration % 5 == 0)
	  {
          cvt::Image img( 200, 200, cvt::IFormat::GRAY_FLOAT );
          img.fill( cvt::Color::BLACK );
          drawData( img, dataset );
          gm.drawGaussians( img );

          gm.outputParams();

          cvt::String file;
          file.sprintf( "out%i.png", iteration );
          img.save( file );
	  }
  }

  return 0;
}
