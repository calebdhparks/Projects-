#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cstdio>
#include "rapidcsv.h"
#include <math.h>
#include <float.h>
#include <set>
#include <limits>
#include <ctime>
#include <chrono>
#include "cuda_tree.h"
#include <stdlib.h>
#include <stdio.h>

const int DEBUG = 0;

void convert_to_num(std::vector<std::vector<std::string>> &input, std::vector<std::vector<double>> &output){
	for(int i = 0; i < input.size(); i++){
		std::vector<double> temp;
		temp.clear();
		for(int j = 0; j < input[i].size(); j++){
			try{
				temp.push_back(std::stod(input[i][j]));
			} catch(std::exception &e){
				temp.push_back(-1);
			}
		}
		output.push_back(temp);
	}
}

// Given some samples and a column index, return a set containing all the possible labels in that column
// The index specified by 'col' should be categorical, not numerical
std::set<std::string> get_labels(std::vector<std::vector<std::string>> &samples, int col){
	std::set<std::string> retval;
	
	for(int i = 0; i < samples.size(); i++){
		retval.insert(samples[i][col]);
	}
	return retval;
}

double percent_change(double v1, double v2){
	return ((v1 - v2)/v1) * 100;
}

std::vector<double> get_split_candidates(std::vector<std::vector<double>> &samples,
										 std::vector<int> &members, int col, int splits){
	std::vector<double> retval;
	double feat_min = DBL_MAX;
	double feat_max = 0.0;
	double val;
	for(int i = 0; i < members.size(); i++){
		val = samples[members[i]][col];
		if(val < feat_min)
			feat_min = val;
		if(val > feat_max)
			feat_max = val;
	}

	if(feat_min < 0)
		feat_min = 0;

	for(int i = 0; i < splits; i++){
		retval.push_back(feat_min + 
			( (i + 1) * ( (feat_max - feat_min) / (splits + 1) ))); 		
	}

	return retval;
}

class Node {
	public:
		std::vector<std::vector<double>> &samples;
		double split_val;
		double best_err = DBL_MAX;
		int best_feature;
		std::vector<int> members;
		std::vector<Node> children;
		double best_improvement = std::numeric_limits<double>::lowest();
		int depth = 1;
		int col_to_exclude = 10;
		Node(std::vector<std::vector<double>> &data): samples(data){}

		Node(std::vector<std::vector<double>> &data, std::vector<int> membs): samples(data){
			this->members = membs;
		}

		void set_depth(int depth){
			this->depth = depth;
		}
		
		int get_depth(){
			return this->depth;
		}

		void split_numeric(int featureIdx,
						   double value,
						   std::vector<int> &left,
						   std::vector<int> &right)
		{
			for(int i = 0; i < this->members.size(); i++){
				if(samples[members[i]][featureIdx] < value){
					left.push_back(members[i]);
				} else{
					right.push_back(members[i]);
				}
			}
		}

		// Calculates the average value of a column.
		double compute_average(int col){
			double current = 0.0;
			if(this->members.size() == 0)
				return 0.0;

			for(int i = 0; i < this->members.size(); i++){
				current += this->samples[this->members[i]][col];
			}
			return current / this->members.size();
		}
		
		//Calculates the squared error of a split
		double compute_sq_error(int col){
			double avg = compute_average(col);
			double num;
			double current = 0.0;

			double cuda_current = 0.0;

			double *tempVals, *deviceVals;
			if(members.size() > 500){
				tempVals = (double*)malloc(members.size() * sizeof(double));
				for(int i = 0; i < members.size(); i++){
					tempVals[i] = samples[members[i]][col];
				}
				cuda_current = cuda::find_err(tempVals, members.size(), avg);
				free(tempVals);
				return cuda_current;
			} else{
				for(int i = 0; i < members.size(); i++){
					num = samples[members[i]][col];
					if(num > 0)
						current += pow(num - avg, 2);
				}
			}
			return current;
		}

		void find_best_split(){
			if(DEBUG)
				printf("This node has %d members\n", members.size());

			for(int featureIdx = 1; featureIdx < samples[0].size() && featureIdx != col_to_exclude; featureIdx++){
				
				Node *left, *right;

				left = new Node(samples);
				right = new Node(samples);
				std::vector<int> left_members, right_members;
				double improvement;


				//Getting split candidates for a continuous variable
				std::vector<double> candidates = get_split_candidates(samples, members, featureIdx, 30);

				for(int i = 0; i < candidates.size(); i++){
						split_numeric(featureIdx, candidates[i], left_members, right_members);	
						left->members = left_members;
						right->members = right_members;
						double new_err = left->compute_sq_error(col_to_exclude) + right->compute_sq_error(col_to_exclude);
						if(new_err < best_err){
							best_err = new_err;
							best_feature = featureIdx;
							split_val = candidates[i];
						}
						left_members.clear();
						right_members.clear();
				}
			}

			if(DEBUG)
				printf("Best Split at Feature (%d) with a value: of %f\n", best_feature, split_val);
		}

		void split(){
			std::vector<int> left_members, right_members;
			for(int i = 0; i < members.size(); i++){
				if(samples[i][best_feature] < split_val){
					left_members.push_back(members[i]);
				} else {
					right_members.push_back(members[i]);
				}
			}
			Node child1 = Node(samples, left_members);
			Node child2 = Node(samples, right_members);
			child1.set_depth(depth + 1);
			child2.set_depth(depth + 1);
			children.push_back(child1);
			children.push_back(child2);
		}

};

class DTree {
	public:
		Node *root;
		int col_to_predict = 10;

		DTree(std::vector<std::vector<double>> &samples){
			root = new Node(samples);
			for(int i = 0; i < samples.size(); i++){
				root->members.push_back(i);
			}
		}

		void build(int max_depth){
			Node *current;
			root->find_best_split();
			// printf("Improvement: %f\n", root->best_improvement);
			root->split();
			current = root;
			
			for(int i = 0; i < current->children.size(); i++){
				Node &child = current->children[i];
				//child = &current->children[i];
				r_helper(max_depth, child);
			}

		}

		void r_helper(int max_depth, Node &n){
			if(n.depth > max_depth || n.members.size() < 2)
				return;
			n.find_best_split();
			n.split();
			// printf("Improvement: %f\n", n.best_improvement);
			for(int i = 0; i < n.children.size(); i++){
				r_helper(max_depth, n.children[i]);
			}
		}

		double predict(std::vector<double> &sample){
			double prediction = 0.0;
			Node *current = root;
			while(current->children.size() > 0){
				if(sample[current->best_feature] < current->split_val){
					current = &current->children[0];
				} else {
					current = &current->children[1];
				}
			}
			
			if(current->members.size() > 0){
				double mySum = 0.0;
				for(int i = 0; i < current->members.size(); i++){
					mySum += current->samples[current->members[i]][col_to_predict];
				}
				prediction = mySum / current->members.size();
			}
			return prediction;
		}

		double evaluate(std::vector<std::vector<double>> &samples){
			double prediction;
			double avg_err = 0.0;
			for(int i = 0; i < samples.size(); i++){
				prediction = predict(samples[i]);
				avg_err += abs(samples[i][col_to_predict] - prediction);
			}
			avg_err /= samples.size();
			return avg_err;
		}
};



void createMap(std::map<std::string,int> &Dict, std::vector<std::vector<std::string>> &sample, int column){
//	Dict["E"]=0;
//	std::cout<<Dict[sample[0][column]]<<std::endl;
	int value=0;
	for (int i =0;i<sample.size();i++){
		if(Dict.count(sample[i][column])==0){
			Dict[sample[i][column]]=value;
			value+=1;
		}
	}
	Dict[" "] = -1;
	Dict[""] = -1;
	// for(auto elem : Dict)
	// {
	// 	std::cout << elem.first <<" " <<elem.second<<"\n";
	// }
	// std::cout<<sample[1][column]<<std::endl;	
}

void delete_column(std::vector<std::vector<std::string>> &vec, int col){
	for(auto &r: vec) r.erase(r.begin() + col);
}

int main(int argc, char *argv[]) {

	if(argc != 2){
		printf("Usage: %s [input file]\n", argv[0]);
		exit(1);
	}

	std::srand( unsigned ( std::time(NULL) ));
	rapidcsv::Document doc(argv[1], rapidcsv::LabelParams(0, -1));

	std::vector<std::string> column_names = doc.GetColumn<std::string>("Name");

	std::vector<std::string> row = doc.GetRow<std::string>(0);
	
	std::map<std::string,int> RatingMap;
	std::map<std::string,int> PlatformMap;
	std::map<std::string,int> GenreMap;
	std::map<std::string,int> PublisherMap;
	std::map<std::string,int> DeveloperMap;

	std::vector<std::vector<std::string>> samples;
	int num_samples = doc.GetRowCount();
	printf("There are %d samples.\n", num_samples);
	
	for(int i = 0; i < num_samples; i++){
		samples.push_back(doc.GetRow<std::string>(i));
	}
	// auto rng = std::default_random_engine {};
	// std::shuffle(std::begin(samples), std::end(samples), rng);


	createMap(RatingMap,samples,15);
	createMap(DeveloperMap,samples,14);
	createMap(PlatformMap,samples,1);
	createMap(GenreMap,samples,3);
	createMap(PublisherMap,samples,4);

	// Here we make everything numeric
	for(int i = 0; i < samples.size(); i++){
		samples[i][15] = std::to_string(RatingMap[samples[i][15]]);
		samples[i][14] = std::to_string(DeveloperMap[samples[i][14]]);
		samples[i][1] = std::to_string(PlatformMap[samples[i][1]]);
		samples[i][3] = std::to_string(GenreMap[samples[i][3]]);
		samples[i][4] = std::to_string(PublisherMap[samples[i][4]]);
	}
	

	std::vector<std::vector<double>> converted_samples;
	convert_to_num(samples, converted_samples);

	std::random_shuffle(converted_samples.begin(), converted_samples.end());
	int eighty_percent = floor((float)num_samples * 0.8);
	printf("%d samples belong in the training set\n", eighty_percent);
	printf("%d samples belong in the test set\n", converted_samples.size() - eighty_percent);

	std::vector<std::vector<double>> training_data, test_data;
	
	int i;
	
	for(i = 0; i < eighty_percent; i++){
		training_data.push_back(converted_samples[i]);
	}

	for(; i < samples.size(); i++){
		test_data.push_back(converted_samples[i]);
	}
	
	DTree decision_tree = DTree(training_data);
	
	auto start = std::chrono::high_resolution_clock::now();
	decision_tree.build(15);
	auto stop = std::chrono::high_resolution_clock::now();


	printf("Tree has finished building.\n");
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop-start);

	double avg_err = decision_tree.evaluate(test_data);
	printf("The Average Error: %2f\n", avg_err);
	
	printf("\n\n --- Time Metrics --- \n\n");
	printf("Tree Building: ");
	std::cout << duration.count() <<  " milliseconds." << std::endl;
	return 0;
}
