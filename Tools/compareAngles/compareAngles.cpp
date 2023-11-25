#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <math.h>
#include <algorithm>
#include <iomanip>

using std::cout, std::endl, std::setw;

#define SPACE 12

int main(int argc, char* argv[]) {
	const char* inputFile = argv[1];
	std::ifstream input(inputFile);
	float min = 180, max = 0;
	int max_angle = 0, min_angle = 0;
	int count = 0, skip = 0;

	std::string left;
	std::string right;

	std::vector<std::pair<float, float>> angles;
	while (input >> left >> right){
		std::string left_angle = left.substr(0, left.find("_"));
		angles.push_back(std::make_pair(std::stof(left_angle), std::stof(right)));
	}

	std::sort(angles.begin(), angles.end(), [] (auto& left, auto& right) {return left.first < right.first;});

	cout << setw(SPACE) << "before" << setw(SPACE) << "after" << setw(SPACE) << "diff" << setw(SPACE) << "false" << endl;

	float total = 0;
	for (auto& angle : angles) {
		float before = angle.first;
		float after = angle.second;

        float diff = abs(before - after);

		cout << setw(SPACE) << before << setw(SPACE) << after << setw(SPACE) << diff;

        if (diff > 90) {
			cout << setw(SPACE) << "*" << endl;
            skip++;
            continue;
        }
        total += diff;
        count++;
        if (diff > max) {
            max = diff;
            max_angle = before;
        }
        if (diff < min) {
            min = diff;
            min_angle = before;
        }
		cout << endl;
	}

	cout << "------------------------------------------------" << std::endl;
	cout << "Average = " << total / count << endl;
	cout << "Max = " << max << " at " << max_angle << " degree" << endl;
	cout << "Min = " << min << " at " << min_angle << " degree" << endl;
	cout << "False angle = " << skip << endl;

	return 0;
}