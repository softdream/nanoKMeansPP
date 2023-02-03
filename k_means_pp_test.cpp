#include "k_means_pp.h"
#include <opencv2/opencv.hpp>

template<typename T, int Dimension = 2>
class KMeansPPTest
{
public:
	using KMeansValueType = typename means::KMeansPP<T, Dimension>::ValueType;
	using KMeansDataType = typename means::KMeansPP<T, Dimension>::DataType;

	void generateTestDataSet( std::vector<KMeansDataType>& data_set, const int cluster_num )
	{
		std::random_device rd;
		std::default_random_engine random_engine( rd() );
		
		// 1. randomly generate cendroids
		std::vector<KMeansDataType> centroids;
		std::uniform_real_distribution<KMeansValueType> uniform_dist_gen( 0, 500 );
		for( int i = 0; i < cluster_num; i ++ ) {
			KMeansDataType centroid = KMeansDataType::Zero();
			for( int j = 0; j < Dimension; j ++ ) {
				centroid[j] = uniform_dist_gen( random_engine );	
			}
			centroids.push_back( centroid );
		}

		// 2. randomly generate data around the cendroids
		for( auto& cendroid : centroids ) {
			for( int i = 0; i < 50; i ++ ) {
				KMeansDataType data = KMeansDataType::Zero();
				for( int j = 0; j < Dimension; j ++ ) {
                                	std::normal_distribution<KMeansValueType> normal_dist_gen( cendroid[j], covriance_ );
					data[j] = normal_dist_gen( random_engine );
                        	}
				data_set.push_back( data );
			}
		}

	}	
	
	void drawDataSet( const  std::vector<KMeansDataType>& data_set )
	{
		const int bias_x = 150;
		const int bias_y = 150;

		for( auto& data : data_set ) {
			cv::circle( image_, cv::Point( data[0] + bias_x, data[1] + bias_y ), 3, cv::Scalar(0, 255, 0), -1 );
		}
		
		cv::imshow( "dataset", image_ );
		cv::waitKey(0);
	}

	void drawCentroids( const std::vector<KMeansDataType>& centroids )
	{
		const int bias_x = 150;
                const int bias_y = 150;

		for( auto& data : centroids ) {
                        cv::circle( image_, cv::Point( data[0] + bias_x, data[1] + bias_y ), 5, cv::Scalar(0, 0, 255), -1 );
                }

                cv::imshow( "dataset", image_ );
                cv::waitKey(0);

	}

	void test( const int K )
	{
		// 1. generate the dataset
		std::vector<KMeansDataType> data_set;
		generateTestDataSet( data_set, K );
		drawDataSet( data_set );

		// 2. clustering
		k_means_pp_ptr_ = std::make_unique<means::KMeansPP<T, Dimension>>( data_set );
		std::vector<KMeansDataType> centroids;
		k_means_pp_ptr_->runKmeansPP( K, centroids );

		drawCentroids( centroids );
	}

private:
	cv::Mat image_ = cv::Mat( 800, 800, CV_8UC3, cv::Scalar(255, 255, 255 ) );

	const KMeansValueType covriance_ = 10;

	std::unique_ptr<means::KMeansPP<T, Dimension>> k_means_pp_ptr_;
};

int main()
{
	std::cout<<"------------------ K Means PP TEST -----------------"<<std::endl;
	
	KMeansPPTest<double> k_means_test;

	k_means_test.test( 10 );

	return 0;
}
