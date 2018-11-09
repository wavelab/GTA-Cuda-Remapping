/* Copyright (c) 2018, Waterloo Autonomous Vehicles Laboratory (WAVELab),
 * Waterloo Intelligent Systems Engineering (WISE) Lab,
 * University of Waterloo. All Rights Reserved.
 *
 * ############################################################################
 ******************************************************************************
 |                                                                            |
 |                         /\/\__/\_/\      /\_/\__/\/\                       |
 |                         \          \____/          /                       |
 |                          '----________________----'                        |
 |                              /                \                            |
 |                            O/_____/_______/____\O                          |
 |                            /____________________\                          |
 |                           /    (#UNIVERSITY#)    \                         |
 |                           |[**](#OFWATERLOO#)[**]|                         |
 |                           \______________________/                         |
 |                            |_""__|_,----,_|__""_|                          |
 |                            ! !                ! !                          |
 |                            '-'                '-'                          |
 |       __    _   _  _____  ___  __  _  ___  _    _  ___  ___   ____  ____   |
 |      /  \  | | | ||_   _|/ _ \|  \| |/ _ \| \  / |/ _ \/ _ \ /     |       |
 |     / /\ \ | |_| |  | |  ||_||| |\  |||_|||  \/  |||_||||_|| \===\ |====   |
 |    /_/  \_\|_____|  |_|  \___/|_| \_|\___/|_|\/|_|\___/\___/ ____/ |____   |
 |                                                                            |
 ******************************************************************************
 * ############################################################################
 *
 * File: main.cu
 * Desc: Main logic for running remapping.
 * Auth: Matt Angus
 *
 * ############################################################################
*/
#include <exception>
#include <ctime>
#include <stdlib.h>
#include <sstream>
#include <string>
#include <random>
#include <chrono>
#include <regex>
#include <experimental/filesystem>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <queue>
#include <boost/lockfree/queue.hpp>
#include <unordered_map>
#include <vector>

#include "ecuda/ecuda.hpp"

#include "zupply.hpp"

#include "GpuMat.cuh"
#include "helpers.h"
#include "colours.h"
namespace fs = std::experimental::filesystem;

// TODO: add to args
const int device = 0;

bool hasEnding (std::string const &fullString, std::string const &ending) {
    if (fullString.length() >= ending.length()) {
        return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
    } else {
        return false;
    }
}

void processFiles(std::vector<std::string> files, std::string image_path, std::string output_path, ecuda::vector<std::pair<int,int>>* d_map, std::string outfile, zz::log::ProgBar* pBar)
{
	cudaSetDevice(device);
	auto& fout = std::cout;
	//std::ofstream fout(outfile, std::ofstream::out | std::ofstream::app);
	std::unique_ptr<GpuMat<unsigned char>> scratchGpuMat;
	std::unique_ptr<GpuMat<unsigned char>> outputImg;

	float readTime = 0.f;

	auto startTime = std::chrono::high_resolution_clock::now();
	for(int i = 0; i < files.size(); i++)
	{
		auto readStart = std::chrono::high_resolution_clock::now();
		//fout << "reading " << files[i] << std::endl;
		cv::Mat img;
		// wait for all data to be saved to disk from game engine
		while(!img.data)
		{
			try
			{
				img = cv::imread(files[i], CV_LOAD_IMAGE_COLOR);
			}
			catch(std::exception& ex)
			{
				fout << "caught a thing" << std::endl;
			}
			if(!img.data) //only be able to parse if IEND chunk is found (i.e. transer complete)
				std::this_thread::sleep_for(std::chrono::milliseconds(100));
		}
		auto readEnd = std::chrono::high_resolution_clock::now();
		if(i == 0)
		{
			scratchGpuMat = std::unique_ptr<GpuMat<unsigned char>>(new GpuMat<unsigned char>(img.rows, img.cols, img.channels(), false));//do this to allocate memory
			outputImg = std::unique_ptr<GpuMat<unsigned char>>(new GpuMat<unsigned char>(img.rows, img.cols, img.channels(), false));
		}
		if(img.rows != scratchGpuMat->height || img.cols != scratchGpuMat->width || img.channels() != scratchGpuMat->depth)
		{
			fout << "skipping " << files[i] << std::endl;
			continue;
		}
		scratchGpuMat->load(img);
		scratchGpuMat->mapColours(*outputImg, *d_map);
		
		cv::Mat outMat = outputImg->getMat();

		std::string outFileName = std::regex_replace(files[i], std::regex(image_path), output_path);
		//fout << "writing " << outFileName << std::endl;
		fs::create_directories(fs::path(outFileName).parent_path());
		cv::imwrite(outFileName, outMat);

		readTime += (float)std::chrono::duration_cast<std::chrono::milliseconds>(readEnd - readStart).count()/1000.f;
		pBar->step();
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

std::vector<std::string> GetImagesToProcess(std::string& inputPath, std::string& outputPath)
{
	std::vector<std::string> ret;
	for(auto& p: fs::recursive_directory_iterator(inputPath))
	{
		std::string curPath = p.path().string();
		bool regFile = fs::is_regular_file(p);
		if(regFile && hasEnding(curPath, "png"))
		{
			std::string imgPath = std::regex_replace(curPath, std::regex(inputPath), outputPath);

			if(!fs::is_regular_file(imgPath))
			{
				ret.push_back(curPath);
			}
		}
	}
	return ret;
}

/**
 * contains cuda specific initializations
 */

int main(int argc, char** argv )
{	
	// grab the arguments
	std::string image_path;
	std::string output_path;
	std::string mapFile;
	int numProc = 8;
	std::vector<int> availGpu = {0,1,2};
	// TODO: add argparse lib
	for (int i = 0; i < argc; i++)
	{
		if (strcmp(argv[i], "-i") == 0)
			image_path = argv[i+1];
		if (strcmp(argv[i], "-o") == 0)
			output_path = argv[i+1];
		if (strcmp(argv[i], "-n") == 0)
			numProc = atoi(argv[i+1]);
		if (strcmp(argv[i], "-m") == 0)
			mapFile = argv[i+1];
	}

	cudaSetDevice(device);

	std::vector<std::pair<int,int>> map = makeMap(mapFile);
	ecuda::vector<std::pair<int,int>> d_map(map.size());
	ecuda::copy(map.begin(), map.end(), d_map.begin());

	for(;;)
	{
		std::vector<std::string> toProcess = GetImagesToProcess(image_path, output_path);
		if(toProcess.size() > 0)
		{
			std::cout << "found " << toProcess.size() << " images" << std::endl;
			zz::log::ProgBar pBar(toProcess.size());
			//TODO: use queue instead of even split
			std::vector<std::vector<std::string>> chunks = SplitVector(toProcess, numProc);

			std::vector<std::thread> threads;

			for(int i = 0; i < numProc && i < chunks.size(); i++)
			{
				std::stringstream ss;
				ss << i << ".out";
				std::string temp = ss.str();
				threads.push_back(std::thread(processFiles, chunks[i], image_path, output_path, &d_map, temp, &pBar));
			}

			for(std::thread& t : threads)
			{
				t.join();
			}
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(5000));
	}

	//return 0; //unreachable
}
