#include "skinmodel.h"
#include <cmath>
#include <iostream>

using namespace std;

/// Constructor
SkinModel::SkinModel() {
	pimpl = new SkinModelPimpl();
}

/// Destructor
SkinModel::~SkinModel() {
	delete pimpl;
}

/// Start the training.  This resets/initializes the model.
///
/// Implementation hint:
/// Use this function to initialize/clear data structures used for training the skin model.
void SkinModel::startTraining() {
	pimpl->features;
	cout << "Initializing..." << endl;
	pimpl->svm = cv::ml::SVM::create();
	pimpl->svm->setType(cv::ml::SVM::C_SVC);
	pimpl->svm->setKernel(cv::ml::SVM::RBF);
	pimpl->svm->setTermCriteria(
		cv::TermCriteria(
			cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS,
			1000, 1e-6
		)
	);
	//pimpl->svm->setC(300);//for C_SVC, PS_SVR and NU_SVR
	//pimpl->svm->setCoef0(1);//for poly/sigmoid
	//pimpl->svm->setGamma(0.5);//for poly/rbf/sigmoid
	//pimpl->svm->setDegree(2);//for poly
}

/// Add a new training image/mask pair.  The mask should
/// denote the pixels in the training image that are of skin color.
///
/// @param img:  input image
/// @param mask: mask which specifies, which pixels are skin/non-skin
void SkinModel::train(const cv::Mat3b& img, const cv::Mat1b& mask) {
	cv::Mat hsv;
	cv::Mat feature(1, FEATURE_SIZE, CV_32FC1);
	cv::Mat3b src;

	pimpl->preprocess(img, src);
	cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);

	size_t counter = 0;
	cv::RNG rng;
	size_t randRow = rng.operator()(img.rows),
		randCol = rng.operator()(img.cols);
	while (counter < SAMPLE_SIZE * .49) {//background samples should be slightly less than foreground?
										//mysterious, F1 score peak at 47% of bg samples
		if (mask.at<uchar>(randRow, randCol) == 255) {
			feature.at<float>(0, 0)
				= static_cast<float>(hsv.at<cv::Vec3b>(randRow, randCol)[0]) / 180.;
			feature.at<float>(0, 1)
				= static_cast<float>(hsv.at<cv::Vec3b>(randRow, randCol)[1]) / 255.;
			feature.at<float>(0, 2)
				= static_cast<float>(hsv.at<cv::Vec3b>(randRow, randCol)[2]) / 255.;

			feature.at<float>(0, 3)
				= static_cast<float>(src.at<cv::Vec3b>(randRow, randCol)[0]) / 255;
			feature.at<float>(0, 4)
				= static_cast<float>(src.at<cv::Vec3b>(randRow, randCol)[1]) / 255.;
			feature.at<float>(0, 5)
				= static_cast<float>(src.at<cv::Vec3b>(randRow, randCol)[2]) / 255.;
			pimpl->features.push_back(feature.clone());
			pimpl->labels.push_back(1);
			counter++;
		}
		randRow = rng.operator()(img.rows);
		randCol = rng.operator()(img.cols);
	}
	while (counter < SAMPLE_SIZE) {
		if (mask.at<uchar>(randRow, randCol) == 0) {
			feature.at<float>(0, 0)
				= static_cast<float>(hsv.at<cv::Vec3b>(randRow, randCol)[0]) / 180.;
			feature.at<float>(0, 1)
				= static_cast<float>(hsv.at<cv::Vec3b>(randRow, randCol)[1]) / 255.;
			feature.at<float>(0, 2)
				= static_cast<float>(hsv.at<cv::Vec3b>(randRow, randCol)[2]) / 255.;

			feature.at<float>(0, 3)
				= static_cast<float>(src.at<cv::Vec3b>(randRow, randCol)[0]) / 255;
			feature.at<float>(0, 4)
				= static_cast<float>(src.at<cv::Vec3b>(randRow, randCol)[1]) / 255.;
			feature.at<float>(0, 5)
				= static_cast<float>(src.at<cv::Vec3b>(randRow, randCol)[2]) / 255.;
			pimpl->features.push_back(feature.clone());
			pimpl->labels.push_back(-1);
			counter++;
		}
		randRow = rng.operator()(img.rows);
		randCol = rng.operator()(img.cols);
	}
}

/// Finish the training.  This finalizes the model.  Do not call
/// train() afterwards anymore.
///
/// Implementation hint:
/// e.g normalize w.r.t. the number of training images etc.
void SkinModel::finishTraining() {

	auto start = std::chrono::system_clock::now();
	cout << "Training ..." << endl;

	cv::Ptr<cv::ml::ParamGrid> nogrid = cv::ml::ParamGrid::create(0, 0, 0);

	//pimpl->svm->train(pimpl->features, cv::ml::ROW_SAMPLE, pimpl->labels);
	pimpl->svm->trainAuto(
		pimpl->features,
		cv::ml::ROW_SAMPLE,
		pimpl->labels,
		6,
		cv::ml::SVM::getDefaultGridPtr(cv::ml::SVM::C),
		cv::ml::SVM::getDefaultGridPtr(cv::ml::SVM::GAMMA),
		nogrid,
		nogrid,
		nogrid,
		cv::ml::SVM::getDefaultGridPtr(cv::ml::SVM::DEGREE),
		true
		);

	cout //<< "Support Vectors: " << pimpl->svm->getSupportVectors() << endl
		<< "C: " << pimpl->svm->getC() << endl //312.5
		<< "Gamma: " << pimpl->svm->getGamma() << endl//0.03375
		<< "P: " << pimpl->svm->getP() << endl
		<< "Nu: " << pimpl->svm->getNu() << endl
		<< "Coef0: " << pimpl->svm->getCoef0() << endl
		<< "Degree: " << pimpl->svm->getDegree() << endl;

	auto end = std::chrono::system_clock::now();

	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";
	//pimpl->svm->save("svm.trainmodel.xml");
}


/// Classify an unknown test image.  The result is a probability
/// mask denoting for each pixel how likely it is of skin color.
///
/// @param img: unknown test image
/// @return:    probability mask of skin color likelihood
cv::Mat1b SkinModel::classify(const cv::Mat3b& img)
{
	//pimpl->svm->load("svm.trainmodel.xml");

	cv::Mat1b skin = cv::Mat1b::zeros(img.rows, img.cols);
	cv::Mat1b dst = cv::Mat1b::zeros(img.rows, img.cols);
	cv::Mat hsv;
	cv::Mat feature(1, FEATURE_SIZE, CV_32FC1);
	cv::Mat3b src;

	pimpl->preprocess(img, src);
	cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);

	for (size_t row = 0; row < img.rows; row++) {
		for (size_t col = 0; col < img.cols; col++) {
			feature.at<float>(0, 0)
				= static_cast<float>(hsv.at<cv::Vec3b>(row, col)[0]) / 180.;
			feature.at<float>(0, 1)
				= static_cast<float>(hsv.at<cv::Vec3b>(row, col)[1]) / 255.;
			feature.at<float>(0, 2)
				= static_cast<float>(hsv.at<cv::Vec3b>(row, col)[2]) / 255.;

			feature.at<float>(0, 3)
				= static_cast<float>(src.at<cv::Vec3b>(row, col)[0]) / 255;
			feature.at<float>(0, 4)
				= static_cast<float>(src.at<cv::Vec3b>(row, col)[1]) / 255.;
			feature.at<float>(0, 5)
				= static_cast<float>(src.at<cv::Vec3b>(row, col)[2]) / 255.;

			float prediction = pimpl->svm->predict(feature);
			if (prediction > 0) {
				skin(row, col) = 255;
			}
			else {
				skin(row, col) = 0;
			}
		}
	}
	pimpl->postprocess(skin, dst);
	return dst;
}

void SkinModel::SkinModelPimpl::preprocess(const cv::Mat3b& src, cv::Mat3b& dst) {

	//blur
	bilateralFilter(src, dst, 3, 3 * 2, 3 / 2);

	//illumination correction
	//cv::Mat lab;
	//cv::cvtColor(dst, lab, cv::COLOR_BGR2Lab);
	//vector<cv::Mat> labPlanes(3);
	//cv::split(lab, labPlanes);
	//cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
	//clahe->setClipLimit(4);
	//cv::Mat mid;
	//clahe->apply(labPlanes[0], mid);// apply the CLAHE algorithm to the L channel
	//mid.copyTo(labPlanes[0]);
	//cv::merge(labPlanes, lab);
	//cv::cvtColor(lab, dst, cv::COLOR_Lab2BGR);

	//equalize intensity
	cv::Mat yCrCb;
	cvtColor(dst, yCrCb, cv::COLOR_BGR2YCrCb);
	vector<cv::Mat> channels(3);
	split(yCrCb, channels);
	equalizeHist(channels[0], channels[0]);
	merge(channels, yCrCb);
	cvtColor(yCrCb, dst, cv::COLOR_YCrCb2BGR);

	//cv::imshow("preprocess", dst);
	//cv::waitKey(2000);
}

void SkinModel::SkinModelPimpl::postprocess(const cv::Mat1b& src, cv::Mat1b& dst) {
	cv::Mat1b mid = dst.clone();
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(src, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, cv::Point(0, 0));
	for (size_t i = 0; i < contours.size(); i++) {
		cv::Point2f pc;
		float radius;
		cv::Rect rect = boundingRect(contours[i]);
		cv::minEnclosingCircle(contours[i], pc, radius);
		float circleArea = CV_PI * radius * radius;
		if (circleArea < 200000 && circleArea > 2000) {
			cv::drawContours(mid, contours, i, cv::Scalar(255), cv::FILLED);
			//cout << circleArea << endl;
		}
	}
	auto kernel = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5), cv::Point(-1, -1));
	morphologyEx(mid, dst, cv::MORPH_OPEN, kernel, cv::Point(-1, -1), 2, cv::BORDER_REPLICATE);
}
