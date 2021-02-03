#include "tvm/runtime/packed_func.h"
#include "tvm/runtime/registry.h"
#include <nlohmann/json.hpp>
#include <iostream>
#include <cuda.h>
#include <cupti.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>
#include <fstream>
#include <chrono>

using json = nlohmann::json;
using namespace tvm;
using namespace runtime;

class TVMCuptiInterface
{
    public:

        ~TVMCuptiInterface();

        typedef struct DeviceData
        {
            int device_id;
            CUdevice device;
            CUcontext device_context;
            std::string device_name;
        }DeviceData_t;

        typedef struct EventData
        {
            int num_metrics, num_events; 
            std::string *metric_names; 
            CUpti_EventID *eventId;    // event ids 
            CUpti_MetricID *metricId;  // metric ids 
            uint32_t *num_metric_passes;
            uint32_t *num_events_passes;
            CUpti_EventGroupSets *metric;
            CUpti_EventGroupSets *event;                   
            cudaEvent_t Start, Stop;
        }EventData_t;

        typedef struct Kernel
        {
            uint64_t start, end, completed;
            uint32_t contextID, streamID;
            int32_t gridX, gridY, gridZ, blockX, blockY, blockZ, staticSharedMemory, dynamicSharedMemory, localMemoryPerThread, localMemoryTotal;
            int64_t gridID;
            CUpti_ActivityPartitionedGlobalCacheConfig partitionedGlobalCacheRequested, partitionedGlobalCacheExecuted;
            const char *name;
        }Kernel_t;

        // typedef struct ActivityData
        // {
        //     int num_activities;
        //     std::vector<uint8_t*> buffers;
        //     std::vector<size_t*> bufferSize;
        //     std::vector<size_t*> maxBufferSize;
        //     std::vector<Kernel_t*> kernels;
        // }ActivityData_t;
    
        static EventData *CurrentConfiguration;
        static DeviceData *CurrentDevice;
        static std::vector<Kernel_t*> Currentkernels;

        static void parse(std::string eventNames[], std::string metricNames[], int num_metrics, int num_events, int deviceID);

        static void setup_CUPTI_Gathering();

        static void start_CUPTI_Gathering();

        static void CUPTIAPI allocate_Buffer(uint8_t **buffer, size_t *size, size_t *maxNumRecords);

        static void CUPTIAPI get_CUPTI_Activity(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize);

        static json stop_CUPTI_Gathering();
        
        static void Insert_CUPTI_Config(json config);

};