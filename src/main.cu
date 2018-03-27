#include <exception>
#include <ctime>
#include <stdlib.h>
#include <sstream>
#include <string>
#include <random>
#include <chrono>
#include <regex>
#include <experimental/filesystem>
#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <queue>
#include <boost/lockfree/queue.hpp>

#include "GpuMat.cuh"
#include "GpuVector.cuh"
#include "helpers.h"
#include "colours.h"
namespace fs = std::experimental::filesystem;

//std::atomic_bool done = false;
//boost::lockfree::queue<cv::Mat> matQueue(2000);

// void readThread(std::vector<std::string>& all_files)
// {
// 	for(int i = 0; i < all_files.size(); i++)
// 	{
// 		std::cout << "reading " << all_files[i] << std::endl;
// 		cv::Mat img = cv::imread(all_files[i], CV_LOAD_IMAGE_COLOR);
// 		matQueue.push(img);
// 	}
// 	done = true;
// }

void processFiles(std::vector<std::string> files, std::string outfile)
{
	std::ofstream fout(outfile);
	std::unique_ptr<GpuMat<unsigned char>> scratchGpuMat;
	std::unique_ptr<GpuMat<unsigned char>> outputImg;

	float readTime = 0.f;

	auto startTime = std::chrono::high_resolution_clock::now(); //to beat 58 s
	for(int i = 0; i < files.size(); i++)
	{
		auto readStart = std::chrono::high_resolution_clock::now();
		fout << "reading " << files[i] << std::endl;
		cv::Mat img = cv::imread(files[i], CV_LOAD_IMAGE_COLOR);
		auto readEnd = std::chrono::high_resolution_clock::now();

		if(i == 0)
		{
			scratchGpuMat = std::unique_ptr<GpuMat<unsigned char>>(new GpuMat<unsigned char>(img.rows, img.cols, img.channels(), false));//do this to allocate memory
			outputImg = std::unique_ptr<GpuMat<unsigned char>>(new GpuMat<unsigned char>(img.rows, img.cols, img.channels(), false));
		}
		scratchGpuMat->load(img);
		scratchGpuMat->mapColours(*outputImg); //(GpuMat<dtype>& to, GpuVector<dtype>& map)
		
		cv::Mat outMat = outputImg->getMat();

		std::string outFileName = std::regex_replace(files[i], std::regex("n_"), "r_");
		fs::create_directories(fs::path(outFileName).parent_path());
		fout << "writing " << outFileName << std::endl;
		cv::imwrite(outFileName, outMat);

		readTime += (float)std::chrono::duration_cast<std::chrono::milliseconds>(readEnd - readStart).count()/1000.f;
	}
	auto endTime = std::chrono::high_resolution_clock::now();

	float totalTime = (float)std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count()/1000.0f;

	fout << files.size() << std::endl;
	fout << "total time: " << totalTime << " seconds" << std::endl;
	fout << "read time: " << readTime << " seconds" << std::endl;
}

template<typename T>
std::vector<std::vector<T>> SplitVector(const std::vector<T>& vec, size_t n)
{
    std::vector<std::vector<T>> outVec;

    size_t length = vec.size() / n;
    size_t remain = vec.size() % n;

    size_t begin = 0;
    size_t end = 0;

    for (size_t i = 0; i < std::min(n, vec.size()); ++i)
    {
        end += (remain > 0) ? (length + !!(remain--)) : length;

        outVec.push_back(std::vector<T>(vec.begin() + begin, vec.begin() + end));

        begin = end;
    }

    return outVec;
}

/**
 * contains cuda specific initializations
 */
int main(int argc, char** argv )
{	
	// grab the arguments
	std::string image_path;
	std::string coloursFile;
	std::string output_path;
	int numProc = 8;
	std::vector<int> availGpu = {0,1,2};
	for (int i = 0; i < argc; i++)
	{
		if (strcmp(argv[i], "-i") == 0)
			image_path = argv[i+1];
		if (strcmp(argv[i], "-o") == 0)
			output_path = argv[i+1];
		if (strcmp(argv[i], "-c") == 0)
			coloursFile = argv[i+1];
		if (strcmp(argv[i], "-n") == 0)
			numProc = atoi(argv[i+1]);
	}

	std::vector<std::string> all_files;

	for(auto& p: fs::recursive_directory_iterator(image_path))
	{
		std::string val = p.path().string();
		std::string fname = p.path().filename().string();
		if(Helpers::hasEnding(val,"png") && fname[0] == 'n')
			all_files.push_back(val);
	}

	std::cout << "found " << all_files.size() << " images" << std::endl;

	std::vector<std::vector<std::string>> chunks = SplitVector(all_files, numProc);

	std::vector<std::thread> threads;

	for(int i = 0; i < numProc; i++)
	{
		std::stringstream ss;
		ss << i << ".out";
		threads.push_back(std::thread(processFiles, chunks[i], ss.str()));
	}

	for(int i = 0; i < numProc; i++)
	{
		threads[i].join();
	}

	return 0;
}
