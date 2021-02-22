#include <nlohmann/json.hpp>
#include <iostream>
#include <cuda.h>
#include <cupti.h>
#include <nvml.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>
#include <fstream>
#include <chrono>

class TVMCuptiInterface
{
    public:

        ~TVMCuptiInterface();

        typedef struct DeviceData
        {
            int device_id;
            CUdevice device;
            nvmlDevice_t NVML_Device;
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
            cudaEvent_t Start, Stop;
        }Kernel_t;

        typedef struct EventValues
        {
            std::vector<uint64_t*> eventValues;
            std::vector<CUpti_EventID*> eventIDs;
            std::vector<size_t> eventSizes;
            std::vector<size_t> eventNumbers;
            cudaEvent_t Start;
        }EventValues_t;

        typedef struct NVMLData
        {
            uint *temperature, *powerUsage, *powerLimit;
        }NVMLData_t;

        static EventValues *CurrentEventValues;
        static EventData *CurrentConfiguration;
        static DeviceData *CurrentDevice;
        static std::vector<Kernel_t*> Currentkernels;
        static void parse(std::string eventNames[], std::string metricNames[], int num_metrics, int num_events);
        static void setup_CUPTI_Metrics(int index);
        static void setup_CUPTI_Events();
        static void setup_Kernel_Gathering();
        static std::string stop_Kernel_Gathering();
        static void start_CUPTI_Gathering(int metGroupSet, int metGroup, bool eventsSampled);
        static void createDevice(int id);
        static std::string stop_CUPTI_Gathering(int metGroupSet, int metGroup, bool eventsSampled, int metIndex);
        static void Insert_CUPTI_Config(std::string input);
        static void newConfig();
};