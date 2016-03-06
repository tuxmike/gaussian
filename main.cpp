#include "GaussianMixture.h"

int main(int argc, char *argv[])
{

	auto const seed = std::random_device()();
	std::mt19937 rand_generator = std::mt19937 ( seed );
	std::uniform_int_distribution< int > d_50_150( 50, 150 );
	std::uniform_int_distribution< int > d_25_75( 25, 75 );
	std::bernoulli_distribution distribution2( 0.5f );

	GaussianMixture<2> gm;

	// fill gm with 500 datapoints
	for( unsigned i = 0; i < 500; i++ ) {
		int upDown = ( distribution2( rand_generator ) ) ? 0 : 100;
		// x: [50,150], y: [25;75]u[125;175]
		int x = d_50_150( rand_generator );
		int y = d_25_75( rand_generator ) + upDown;
		
		gm.addDatapoint( x, y );
	}
	
	std::cout << gm << std::endl;
	std::cout << "Entropy: " << gm.getShannonEntropy() << " Mean: " << gm.getMean() << " Variance:" << gm.getVariances() << std::endl;

	return 0;
}
