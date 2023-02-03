#ifndef __K_MEANS_PP_H
#define __K_MEANS_PP_H

#include <vector>
#include <iostream>
#include <random>
#include <type_traits>

#include <limits>

#include <Eigen/Dense>

namespace means
{

template<typename T>
struct is_double_or_float
{
	static constexpr bool value = false;
};

template<>
struct is_double_or_float<float>
{
        static constexpr bool value = true;
};

template<>
struct is_double_or_float<double>
{
        static constexpr bool value = true;
};



template<typename T, int Dimension>
using VecType = typename Eigen::Matrix<T, Dimension, 1>;


template<typename T, int Dimension, typename = typename std::enable_if<is_double_or_float<T>::value>::type>
class KMeansPP
{
public:
	using ValueType = T;
	using DataType = VecType<T, Dimension>;

	KMeansPP() = delete;

	KMeansPP( const std::vector<DataType>& input_data ) : input_data_( input_data ),
							      random_index_gen_( 0, input_data.size() - 1 ),
							      random_real_gen_( 0, 1 ),
							      random_engine_( rd_() )
	{
	
	}

	~KMeansPP()
	{
	
	}

	const std::vector<std::vector<int>> runKmeansPP( const int K, std::vector<DataType>& centroids )
	{
		assert( K <= input_data_.size() );

		std::vector<std::vector<int>> clusters_vec( K );

		// 1. initalize the initial centroids according to K-Means Plus Plus Algorithm
		init(); // allocate the memory

		int first_centroid = random_index_gen_( random_engine_ );
		initial_centroids_.push_back( input_data_[ first_centroid ] ); // randomly select a sample data as an initial centroid

#ifdef TEST_LOG_ON
		std::cout<<"first centroid index = "<<first_centroid<<std::endl;
#endif

		for( int i = 1; i < K; i ++ ) {
			updateNearestCluster( initial_centroids_ );
			initial_centroids_.push_back( input_data_[ getNextInitialCentroidIndex() ] );
		}

#ifdef TEST_LOG_ON
		std::cout<<"initial centroids : "<<std::endl;
	        for( auto it : initial_centroids_ ) {
	
        	        for( int i = 0; i < 10; i ++ ) {
                	        std::cout<<it[i]<<" ";
	                }
        	        std::cout<<std::endl;
        	}
#endif
		// 2. 
		cur_centroids_ = initial_centroids_;
		do {
			prev_centroids_ = cur_centroids_;
			updateNearestCluster( cur_centroids_ );
	
			updateCentroids( cur_centroids_ );			
		}while( !equalCentroids( cur_centroids_, prev_centroids_ ) );

		// 3. get the results
		for( int i = 0; i < input_data_.size(); i ++ ) {
			clusters_vec[ nearest_cluster_idx_[ i ] ].push_back( i );
		}

		centroids = cur_centroids_;

#ifdef TEST_LOG_ON
		std::cout<<"result centroids : "<<std::endl;
		for( auto it : centroids ) {

	                for( int i = 0; i < 10; i ++ ) {
        	                std::cout<<it[i]<<" ";
                	}
	                std::cout<<std::endl;
        	}
#endif

		return clusters_vec;
	}

private:
	void init()
	{
		nearest_cluster_idx_.assign( input_data_.size(), -1 );
		nearest_cluster_dist_.assign( input_data_.size(), 0 );
	}

	void updateNearestCluster( const std::vector<DataType>& centroids )
	{
		for( int i = 0; i < input_data_.size(); i ++ ) { 
			int idx = getClosestCentroidIndex( i, centroids );
			//std::cout<<"closet idx = "<<idx<<std::endl;

			nearest_cluster_idx_[i] = idx;
			nearest_cluster_dist_[i] = distance( centroids[ idx ], input_data_[ i ] );
		}
		
	}

	const int getClosestCentroidIndex( const int data_idx, const std::vector<DataType>& centroids )
	{
		int closest_cluster = -1;
		ValueType min_dist = std::numeric_limits<ValueType>::max();

		for( int i = 0; i < centroids.size(); i ++ ) {
			ValueType dist = distance( centroids[i], input_data_[ data_idx ] );

			if( min_dist > dist ) {
				min_dist = dist;
				closest_cluster = i;
			}
		}

		assert( closest_cluster != -1 );

		return closest_cluster;
	}

	const ValueType distance( const DataType& p1, const DataType& p2 ) const
	{
		//return ( p1 - p2 ).norm();
		return ( p1 - p2 ).squaredNorm();
	}

	const int getNextInitialCentroidIndex()
	{
		ValueType d_sum = 0;
		for( auto& dist : nearest_cluster_dist_ ) {
			d_sum += dist;
		}

#ifdef TEST_LOG_ON
		std::cout<<"total distance = "<<d_sum<<std::endl;
#endif
		// The probability D(x)/SIGMA(D(x))
		std::vector<ValueType> prob_vec( input_data_.size(), 0 );
		for( int i = 0; i < input_data_.size(); i ++ ) {
			prob_vec[i] = ( nearest_cluster_dist_[i] / d_sum );
		}
	
		for( int i = 1; i < input_data_.size(); i ++ ) {
                        prob_vec[i] += prob_vec[ i - 1 ];
                }

#ifdef TEST_LOG_ON
		std::cout<<"probabilities : "<<std::endl;
		for( auto it : prob_vec ) {
			std::cout<<it<< " "; 
		}
		std::cout<<std::endl;
#endif
		// Choosing the next point with a probabilty D(x)/SIGMA(D(x))
		ValueType random_num = random_real_gen_( random_engine_ );

#ifdef TEST_LOG_ON
		std::cout<<"random num = "<<random_num<<std::endl;
#endif
		
		for( int i = 0; i < prob_vec.size(); i ++ ) {
			if( random_num < prob_vec[i] ) {
				return i;
			}
		}

		return prob_vec.size() - 1;
	}

	void updateCentroids( const std::vector<DataType>& centroids )
	{
		std::vector<int> freq( centroids.size(), 0 );

		std::vector<DataType> new_centroids( centroids.size(), DataType::Zero() );

		for( int i = 0; i < input_data_.size(); i ++ ) {
			++ freq[ nearest_cluster_idx_[ i ] ];
		
			new_centroids[ nearest_cluster_idx_[ i ] ] += input_data_[ i ];
		}

		for( int i = 0; i < centroids.size(); i ++ ) {
			if( freq[i] ) {
				new_centroids[i] *= ( 1.0 / ( static_cast<ValueType>( freq[i] ) ) );
			}
		}

		cur_centroids_ = new_centroids;
	}

	bool equalCentroids( const std::vector<DataType>& centroids1, const std::vector<DataType>& centroids2 ) const
	{
		ValueType dist_sum = 0;
		for( int i = 0; i < centroids1.size(); i ++ ) {
			dist_sum += distance( centroids1[i], centroids2[i] );
		}

		return ( dist_sum < 1e-9 );
	}

private:
	// random
	std::random_device rd_;
	std::default_random_engine random_engine_;
	std::uniform_int_distribution<int> random_index_gen_;
	std::uniform_real_distribution<ValueType> random_real_gen_;  

private:
	// variables
	std::vector<DataType> input_data_;

	std::vector<int> nearest_cluster_idx_;
	std::vector<ValueType> nearest_cluster_dist_;

	std::vector<DataType> initial_centroids_;
	std::vector<DataType> cur_centroids_;
	std::vector<DataType> prev_centroids_;
};


}

#endif
