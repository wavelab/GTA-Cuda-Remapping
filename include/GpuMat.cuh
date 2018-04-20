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
 * File: GpuMat.cuh
 * Desc: Wrapper class to interface with cv::Mat and cuda
 * Auth: Matt Angus
 *
 * ############################################################################
*/
#pragma once

#include <cuda.h>
#include <opencv2/opencv.hpp>

#include "ecuda/ecuda.hpp"

#include "helpers.cuh"
#include "ops.cuh"
#include "GpuVector.cuh"

template<typename dtype>
class GpuMat
{
private:
    const static int kThreadsPerBlock = 1024;
public:
    int height, width, depth, numElem, size;
    dtype* gpu_data;

    /**
     * Initialize a gpu mat with value "value"
     */
    GpuMat(int height, int width, int depth, dtype value) : GpuMat(height, width, depth, false)
    {
        fill(value);
    }

    /**
     * Initialize a gpu mat with the data from an opencv matrix
     */
    GpuMat(cv::Mat& mat) : GpuMat(mat.rows, mat.cols, mat.channels(), false)
    {
        gpuErrchk( cudaMemcpy(gpu_data, mat.ptr(), size, cudaMemcpyHostToDevice) );
    }

    GpuMat(GpuMat<dtype>& other) : GpuMat(other.height, other.width, other.depth, false)
    {
        gpuErrchk( cudaMemcpy(gpu_data, other.gpu_data, size, cudaMemcpyDeviceToDevice) );
    }

    /**
     * Initialize a gpu mat with zeros
     */
    GpuMat(int height, int width, int depth, bool zero = true)
    {
        this->height = height;
        this->width = width;
        this->depth = depth;
        numElem = height*width*depth;
        size = numElem*sizeof(dtype);

        gpuErrchk( cudaMalloc((void**) &gpu_data, size) );
        if(zero)
            gpuErrchk( cudaMemset( gpu_data, 0, size) );	
    }

    ~GpuMat()
    {
        gpuErrchk( cudaFree(gpu_data) ); 
    }

    cv::Mat getMat()
    {
        dtype* output_im = new dtype[numElem];
        gpuErrchk( cudaMemcpy(output_im, gpu_data, size, cudaMemcpyDeviceToHost) );
        std::vector<int> sizes = {height, width, depth};
        //clone so opencv owns data
        cv::Mat ret = cv::Mat(height, width, CV_MAKETYPE(cv::DataType<dtype>::type, depth), output_im).clone();
        delete output_im;
        return ret;
    }

    template <typename target>
    void convertTo(GpuMat<target>& targetMat)
    {
        if(targetMat.height != height || targetMat.width != width || targetMat.depth != depth)
            throw std::runtime_error("targetMat must have same height, width and depth as source");
        LAUNCH(SINGLE_ARG(ops::convertTo<dtype, target>))(gpu_data, targetMat.gpu_data, numElem);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
    }
        
    void fill(dtype value)
    {
        LAUNCH(ops::fill<dtype>)(value, gpu_data, numElem);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
    }

    void load(cv::Mat& other)
    {
        if(other.rows != height || other.cols != width || other.channels() != depth)
            throw std::runtime_error("Loading Mats that don't have the same height width and depth is not supported");
        gpuErrchk( cudaMemcpy(gpu_data, other.ptr(), size, cudaMemcpyHostToDevice) );
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
    }

    void mapColours(GpuMat<dtype>& to, ecuda::vector<std::pair<int,int>>& d_map)
    {
        //void mapColours(dtype* from, dtype* to, dtype* map, int N)
        LAUNCH(ops::mapColours<dtype>)(gpu_data, to.gpu_data, d_map, numElem);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
    }
};