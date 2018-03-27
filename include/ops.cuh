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
    void mapColours(dtype* from, dtype* to, int N)
    {
        CUDA_1D_KERNEL_LOOP(i, N)
        {
            if(i % 3 == 0)
            {
                int b = from[i];
                int g = from[i+1];
                int r = from[i+2];
                int ind = RGB(r,g,b);//((r & 0xFF) << 16) | ((g & 0xFF) << 8) | (b & 0xFF);
                int new_ind;
                switch(ind)
                {
                    case Dont_care_old: new_ind = Dont_care_new; break;
                    case bridge_old: new_ind = bridge_new; break;
                    case building_old: new_ind = building_new; break;
                    case construction_barrel_old: new_ind = construction_barrel_new; break;
                    case construction_barricade_old: new_ind = construction_barricade_new; break;
                    case crosswalk_old: new_ind = crosswalk_new; break;
                    case curb_old: new_ind = curb_new; break;
                    case white_old: new_ind = white_new; break;
                    case debris_old: new_ind = debris_new; break;
                    case fence_old: new_ind = fence_new; break;
                    case guard_rail_old: new_ind = guard_rail_new; break;
                    case lane_separator_old: new_ind = lane_separator_new; break;
                    case pavement_marking_old: new_ind = pavement_marking_new; break;
                    case rail_track_old: new_ind = rail_track_new; break;
                    case road_old: new_ind = road_new; break;
                    case roadside_structure_old: new_ind = roadside_structure_new; break;
                    case rumble_strip_old: new_ind = rumble_strip_new; break;
                    case sidewalk_old: new_ind = sidewalk_new; break;
                    case terrain_old: new_ind = terrain_new; break;
                    case traffic_cone_old: new_ind = traffic_cone_new; break;
                    case traffic_light_old: new_ind = traffic_light_new; break;
                    case traffic_marker_old: new_ind = traffic_marker_new; break;
                    case traffic_sign_old: new_ind = traffic_sign_new; break;
                    case tunnel_old: new_ind = tunnel_new; break;
                    case utility_pole_old: new_ind = utility_pole_new; break;
                    case vegetation_old: new_ind = vegetation_new; break;
                    case wall_old: new_ind = wall_new; break;
                    case Car_old: new_ind = Car_new; break;
                    case Trailer_old: new_ind = Trailer_new; break;
                    case Bus_old: new_ind = Bus_new; break;
                    case Truck_old: new_ind = Truck_new; break;
                    case Airplane_old: new_ind = Airplane_new; break;
                    case Moterbike_old: new_ind = Moterbike_new; break;
                    case Bycicle_old: new_ind = Bycicle_new; break;
                    case Boat_old: new_ind = Boat_new; break;
                    //case Railed_old: new_ind = Railed_new; break;
                    case Pedestrian_old: new_ind = Pedestrian_new; break;
                    case Animal_old: new_ind = Animal_new; break;
                    default: new_ind = 0;
                }

                to[i] = (new_ind);
                to[i+1] = (new_ind);
                to[i+2] = (new_ind);
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
