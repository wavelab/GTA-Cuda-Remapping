#pragma once

template<typename dtype>
class GpuVector
{
private:
    const static int kThreadsPerBlock = 1024;
public:
    int numElem, size;
    dtype* gpu_data;

    /**
     * Initialize a gpu mat with value "value"
     */
    GpuVector(int numElem, dtype value) : GpuVector(numElem, false)
    {
        fill(value);
    }

    /**
     * Initialize a gpu mat with the data from an opencv matrix
     */
    GpuVector(std::vector<dtype>& vec) : GpuVector(vec.size(), false)
    {
        gpuErrchk( cudaMemcpy(gpu_data, &vec[0], size, cudaMemcpyHostToDevice) );
    }

    GpuVector(GpuVector<dtype>& other) : GpuVector(other.numElem, false)
    {
        gpuErrchk( cudaMemcpy(gpu_data, other.gpu_data, size, cudaMemcpyDeviceToDevice) );
    }

    /**
     * Initialize a gpu mat with zeros
     */
    GpuVector(int numElem, bool zero = true)
    {
        this->numElem = numElem;
        size = numElem*sizeof(dtype);

        gpuErrchk( cudaMalloc((void**) &gpu_data, size) );
        if(zero)
            gpuErrchk( cudaMemset( gpu_data, 0, size) );	
    }

    ~GpuVector()
    {
        gpuErrchk( cudaFree(gpu_data) ); 
    }

    /*cv::Mat getVec()
    {
        dtype* output_im = new dtype[numElem];
        gpuErrchk( cudaMemcpy(output_im, gpu_data, size, cudaMemcpyDeviceToHost) );
        std::vector<int> sizes = {height, width, depth};
        //clone so opencv owns data
        cv::Mat ret = cv::Mat(height, width, CV_MAKETYPE(cv::DataType<dtype>::type, depth), output_im).clone();
        delete output_im;
        return ret;
    }*/

    void divide(dtype scalar)
    {
        LAUNCH(ops::divide)(scalar, gpu_data, numElem);
    }
        
    void multiply(dtype scalar)
    {
        LAUNCH(ops::multiply)(scalar, gpu_data, numElem);
    }
        
    void add(dtype scalar)
    {
        LAUNCH(ops::add)(scalar, gpu_data, numElem);
    }
        
    void fill(dtype value)
    {
        LAUNCH(ops::fill)(value, gpu_data, numElem);
    }

    void load(std::vector<dtype>& other)
    {
        if(other.size != size)
            throw "Loading vectors that don't have the same size is not supported";
        gpuErrchk( cudaMemcpy(gpu_data, &other[0], size, cudaMemcpyHostToDevice) );
    }

    void maxTrackInds(GpuVector<dtype>& other, GpuVector<int>* inds, int otherInd)
    {
        if(numElem != other.numElem)
            throw "Number of elements must match";
        LAUNCH(ops::maxTrackInds)(gpu_data, other.gpu_data, inds->gpu_data, otherInd, numElem);
    }
};