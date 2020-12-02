///
///  Assignment 1
///  Skin Color Classification
///
#include <opencv2/opencv.hpp>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include <fstream>

#include "skinmodel.h"
#include "ROC.h"
using namespace std;

vector<string> getAllImages(string path) {

	vector<string> images;
	namespace fs = boost::filesystem;
	for (fs::directory_iterator it(fs::path(path + "/.")); it != fs::directory_iterator(); it++)
		if (is_regular_file(*it) and it->path().filename().string().substr(0, 5) != "mask-")
			images.push_back(it->path().filename().string());

	return images;
}

int main(int argc, char* argv[]) {

	//// parse command line options
	boost::program_options::variables_map pom;
	{
		namespace po = boost::program_options;
		po::options_description pod(string("Allowed options for ") + argv[0]);
		pod.add_options()
			("help,h", "produce this help message")
			("gui,g", "Enable the GUI");

		po::store(po::command_line_parser(argc, argv).options(pod).run(), pom);
		po::notify(pom);

		if (pom.count("help")) {
			cout << "Usage:" << endl << pod << "\n";
			return 0;
		}
	}

	//// TRAINING
	/// create skin color model instance
	SkinModel model;

	/// train model with all images in the train folder
	model.startTraining();

	for (auto&& f : getAllImages("data/train")) {
		cout << "Extracting Features on Image " << "data/train/" + f << endl;
		cv::Mat3b img = cv::imread("data/train/" + f);
		cv::Mat1b mask = cv::imread("data/train/mask-" + f, 0);

		cv::threshold(mask, mask, 127, 255, cv::THRESH_BINARY);
		model.train(img, mask);
	}
	/// notify the model that we finished training
	model.finishTraining();


	//// VALIDATION
	ROC<int> roc;
	for (auto&& f : getAllImages("data/validation")) {
		cout << "Validation on Image " << "data/validation/" + f << endl;
		cv::Mat3b img = cv::imread("data/validation/" + f);
		cv::Mat1b hyp = model.classify(img);

		cv::Mat1b mask = cv::imread("data/validation/mask-" + f, 0);

		if (pom.count("gui")) {
			cv::imshow("Validation Image", img);
			cv::imshow("Validation Hypothesis", hyp);
			cv::waitKey(10);
		}

		for (int i = 0; i < hyp.rows; i++)
			for (int j = 0; j < hyp.cols; j++)
				roc.add(mask(i, j) > 127, hyp(i, j));
	}
	/// update statistics and show results
	roc.update();
	cout << "Overall F1 score: " << roc.F1 << endl;
	if (pom.count("gui")) {
		cv::imshow("ROC", roc.draw());
		cv::waitKey(0);
	}

	//// TEST
	{
		boost::filesystem::create_directory(boost::filesystem::path("out"));
		for (auto&& f : getAllImages("data/test")) {
			cout << "Test Image " << "data/test/" + f << endl;
			cv::Mat3b img = cv::imread("data/test/" + f);
			cv::Mat1b hyp = model.classify(img);

			cv::imwrite("out/out-" + f, hyp);

			if (pom.count("gui")) {
				cv::imshow("Test Image", img);
				cv::imshow("Test Hypothesis", hyp);
				cv::waitKey(10);
			}
		}
	}
}
