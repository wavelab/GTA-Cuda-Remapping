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
 * File: ops.cuh
 * Desc: Header containing all the cuda kernels
 * Auth: Matt Angus
 *
 * ############################################################################
*/
#pragma once

#include "helpers.cuh"
#include "colours.h"

#define ADD_FUNC(type)          \
    template __global__ void add<type>(type scalar, type* arr, int N)

#define DIVIDE_FUNC(type)       \
    template __global__ void divide<type>(type scalar, type* arr, int N)

#define MULTIPLY_FUNC(type)     \
    template __global__ void multiply<type>(type scalar, type* arr, int N)

#define FILL_FUNC(type)     \
    template __global__ void fill<type>(type value, type* arr, int N)

#define MAXTI_FUNC(type)    \
    template __global__ void maxTrackInds<type>(type* maxVals, type* toTest, int* inds, int testInd, int N)

#define ITOC_FUNC(type)     \
    template __global__ void indsToColour<type>(int* inds, type* maxVals, type* colours, int* r, int* g, int* b, int N)

#define CONV_FUNC(t1, t2)     \
    template __global__ void convertTo<t1, t2>(t1* src, t2* dest, int N)

#define DECLARE_FUNC(func)      \
    func(int);                  \
    func(float);                \
    func(double);               \
    func(char);


namespace ops
{
    template<typename dtype> __global__
    void divide(dtype scalar, dtype* arr, int N)
    {
        CUDA_1D_KERNEL_LOOP(i, N)
        {
            arr[i] /= scalar;
        }
    }

    template<typename dtype> __global__
    void multiply(dtype scalar, dtype* arr, int N)
    {
        CUDA_1D_KERNEL_LOOP(i, N)
        {
            arr[i] *= scalar;
        }
    }

    template<typename dtype> __global__
    void add(dtype scalar, dtype* arr, int N)
    {
        CUDA_1D_KERNEL_LOOP(i, N)
        {
            arr[i] += scalar;
        }
    }

    template<typename dtype> __global__
    void fill(dtype value, dtype* arr, int N)
    {
        CUDA_1D_KERNEL_LOOP(i, N)
        {
            arr[i] = value;
        }
    }

    template<typename dtype> __global__
    void maxTrackInds(dtype* maxVals, dtype* toTest, int* inds, int testInd, int N)
    {
        CUDA_1D_KERNEL_LOOP(i, N)
        {
            if(maxVals[i] < toTest[i])
            {
                maxVals[i] = toTest[i];
                inds[i] = testInd;
            }
        }
    }

    template<typename dtype> __global__
    void indsToColour(int* inds, dtype* maxVals, dtype* colours, int* r, int* g, int* b, int N)
    {
        CUDA_1D_KERNEL_LOOP(i, N)
        {
            if(maxVals[i/3] == 0)
                colours[i] = 0;
            else
            {
                if(i % 3 == 0)
                    colours[i] = (dtype)b[inds[i/3]];
                else if(i % 3 == 1)
                    colours[i] = (dtype)g[inds[i/3]];
                else if(i % 3 == 2)
                    colours[i] = (dtype)r[inds[i/3]];
            }
        }
    }

    template<typename dtype, typename target> __global__
    void convertTo(dtype* src, target* dest, int N)
    {
        CUDA_1D_KERNEL_LOOP(i, N)
        {
            dest[i] = (target)src[i];
        }
    }

    template<typename dtype> __global__
    void mapColours(dtype* from, dtype* to, typename ecuda::vector<std::pair<int, int>>::kernel_argument d_map, int N)
    {
        CUDA_1D_KERNEL_LOOP(i, N)
        {
            if(i % 3 == 0)
            {
                dtype b = from[i];
                dtype g = from[i+1];
                dtype r = from[i+2];
                int ind = RGB(r,g,b);//((r & 0xFF) << 16) | ((g & 0xFF) << 8) | (b & 0xFF);
                // TODO: make default value a parameter.
                int new_ind = 0;
                for(int j = 0; j < d_map.size(); j++)
                {
                    if(d_map[j].first == ind)
                    {
                        new_ind = d_map[j].second;
                        break;
                    }
                }

                to[i] = GET_B(new_ind);
                to[i+1] = GET_G(new_ind);
                to[i+2] = GET_R(new_ind);
            }
        }
    }

    DECLARE_FUNC(MULTIPLY_FUNC);
    DECLARE_FUNC(DIVIDE_FUNC);
    DECLARE_FUNC(ADD_FUNC);
    DECLARE_FUNC(FILL_FUNC);
    DECLARE_FUNC(MAXTI_FUNC);
    DECLARE_FUNC(ITOC_FUNC);
    CONV_FUNC(int,float);
    CONV_FUNC(int,double);
    CONV_FUNC(int,char);
    CONV_FUNC(float,int);
    CONV_FUNC(float,double);
    CONV_FUNC(float,char);
    CONV_FUNC(double,int);
    CONV_FUNC(double,float);
    CONV_FUNC(double,char);
    CONV_FUNC(char,int);
    CONV_FUNC(char,float);
    CONV_FUNC(char,double);

}

#undef DECLARE_FUNC
#undef FILL_FUNC
#undef ADD_FUNC
#undef MULTIPLY_FUNC
#undef DIVIDE_FUNC
