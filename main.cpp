#include "GaussianMixture.h"

#include <iostream>

int main(int argc, char *argv[])
{

	auto const seed = std::random_device()();
	std::mt19937 rand_generator = std::mt19937 ( seed );
	std::uniform_int_distribution< int > d_50_150( 50, 150 );
	std::uniform_int_distribution< int > d_25_75( 25, 75 );
	std::bernoulli_distribution distribution2( 0.5f );

	// Gaussian Mixture with 2 gaussians
	GaussianMixture<2> gm;

	// fill gm with 500 datapoints
	for( unsigned i = 0; i < 500; i++ ) {
		int upDown = ( distribution2( rand_generator ) ) ? 0 : 100;
		// x: [50,150], y: [25;75]u[125;175]
		int x = d_50_150( rand_generator );
		int y = d_25_75( rand_generator ) + upDown;
		
		gm.addDatapoint( x, y );
	}
	
	// print components parameters, for our simple example they should form two spots: 100,50 and 100,150 
	std::cout << gm;
	
	std::pair< float, float > mean = gm.getMean();
	std::pair< float, float > variances = gm.getVariances();

	// print mixture properties
	std::cout << "Entropy: " << gm.getShannonEntropy() << " Mean: " << mean.first << "," << mean.second << " Variance:" << variances.first << "," << variances.second << std::endl;

	return 0;
}
