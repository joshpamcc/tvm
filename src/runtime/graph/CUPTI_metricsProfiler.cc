#include "CUPTI_metricsProfiler.h"
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
#define BUF_SIZE (16 * 1024)
#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)                                            \
  (((uintptr_t) (buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t) (buffer) & ((align)-1))) : (buffer))

TVMCuptiInterface::EventData_t *TVMCuptiInterface::CurrentConfiguration = NULL;
TVMCuptiInterface::DeviceData_t *TVMCuptiInterface::CurrentDevice = NULL;
std::vector<TVMCuptiInterface::Kernel_t*> TVMCuptiInterface::Currentkernels;

void catchCUDAError(CUresult error)
{
    if (error != CUDA_SUCCESS)
    {
        std::cout<<"Error For CUDA API: "<<error<<std::endl;
        exit(-1);
    }
}

void catchCUPTIError(CUptiResult error)
{
    if (error != CUPTI_SUCCESS)
    {
        const char *errMsg = (char*) malloc(sizeof(char)*100);
        cuptiGetResultString(error,&errMsg);
        std::cout<<"Error For CUPTI API: "<<error<<" msg: "<<errMsg<<std::endl;
        exit(-1);
    }
}

void CUPTIAPI TVMCuptiInterface::allocate_Buffer(uint8_t **buffer, size_t *size, size_t *maxNumRecords)
{
    uint8_t *buff = (uint8_t*) malloc(BUF_SIZE + ALIGN_SIZE); 
    if (buff == NULL)
    {
        std::cerr<<"Not enough memory for buffer";
        exit(-1);
    }
    *size = BUF_SIZE;
    *buffer = ALIGN_BUFFER(buff, ALIGN_SIZE);
    *maxNumRecords = 0;
}

void CUPTIAPI TVMCuptiInterface::get_CUPTI_Activity(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize)
{
    CUptiResult status;
    CUpti_Activity *record = NULL;
    CUpti_ActivityKernel4 * kernel;
    if (validSize > 0)
    {
        while(kernel->kind == CUPTI_ACTIVITY_KIND_KERNEL && status == CUPTI_SUCCESS)
        {
            status = cuptiActivityGetNextRecord(buffer, size, &record);
            kernel = (CUpti_ActivityKernel4 *) record;
            if (kernel->kind != CUPTI_ACTIVITY_KIND_KERNEL)
            {
                std::cerr<<"record is not a kernel";
                exit(-1);
            }
            TVMCuptiInterface::Kernel_t *Kernel = new TVMCuptiInterface::Kernel_t();
            Kernel->name = kernel->name;
            Kernel->start = kernel->start;
            Kernel->end = kernel->end;
            Kernel->completed = kernel->completed;
            Kernel->blockX = kernel->blockX;
            Kernel->blockY = kernel->blockY;
            Kernel->blockZ = kernel->blockZ;
            Kernel->gridID = kernel->gridId;
            Kernel->gridX = kernel->gridX;
            Kernel->gridY = kernel->gridY;
            Kernel->gridZ = kernel->gridZ;
            Kernel->contextID = kernel->contextId;
            Kernel->streamID = kernel->streamId;
            Kernel->dynamicSharedMemory = kernel->dynamicSharedMemory;
            Kernel->staticSharedMemory = kernel->staticSharedMemory;
            Kernel->localMemoryPerThread = kernel->localMemoryPerThread;
            Kernel->localMemoryTotal = kernel->localMemoryTotal;  
            Kernel->partitionedGlobalCacheRequested = kernel->partitionedGlobalCacheRequested;
            Kernel->partitionedGlobalCacheExecuted = kernel->partitionedGlobalCacheExecuted;
            TVMCuptiInterface::Currentkernels.push_back(Kernel);
        }
    }
    free(buffer);
}

void createEventData(CUpti_EventID *event_ID[], CUpti_MetricID *metric_ID[], int num_metrics, int num_events, std::string metricNames[])
{
    TVMCuptiInterface::EventData_t *eventData = new TVMCuptiInterface::EventData_t ();
    eventData->eventId = (CUpti_EventID*) calloc(sizeof(CUpti_EventID), num_events);
    eventData->metricId = (CUpti_MetricID*) calloc(sizeof(CUpti_MetricID), num_metrics);
    eventData->metric_names = (std::string*) calloc(sizeof(std::string), num_metrics);
    eventData->event = NULL;
    eventData->metric = NULL;
    for (int i = 0; i < num_metrics; i++)
    {
        eventData->metric_names[i] = metricNames[i]; 
    }
    for (int i = 0; i < num_events; i++)
    {
        eventData->eventId[i] = *event_ID[i];
    }
    for (int i = 0; i < num_metrics; i++)
    {
        eventData->metricId[i] = *metric_ID[i];
    }
    eventData->num_events = num_events;
    eventData->num_metrics = num_metrics;
    cudaEventCreate(&(eventData->Stop));
    cudaEventCreate(&(eventData->Start));
    TVMCuptiInterface::CurrentConfiguration = eventData;
}

/** 
 * Retrieves the metric ids from CUPTI of the metrics supplied
*/
void getMetrics(std::string metrics[], int num_metrics, CUpti_MetricID *metricList[])
{
    for (int i = 0; i < num_metrics; i++)
    {
        CUpti_MetricID *retrieved_metricId = new CUpti_MetricID; 
        char * metric = new char [metrics[i].length()+1];
        std::strcpy(metric, metrics[i].c_str());
        catchCUPTIError(cuptiMetricGetIdFromName(TVMCuptiInterface::CurrentDevice->device, metric, retrieved_metricId));
        metricList[i] = retrieved_metricId;
        retrieved_metricId = NULL;
    }
}

void getEvents(std::string events[], int num_events, CUpti_EventID *eventList[])
{
    for (int i = 0; i < num_events; i++)
    {
        CUpti_EventID *retrieved_eventID = new CUpti_EventID; 
        char * event = new char [events[i].length()+1];
        std::strcpy(event, events[i].c_str());
        catchCUPTIError(cuptiEventGetIdFromName(TVMCuptiInterface::CurrentDevice->device, event, retrieved_eventID));
        eventList[i] = retrieved_eventID;
        retrieved_eventID = NULL;
    }
}

void createDevice(int deviceID) //creates and gets device context for the provided device id
{
    TVMCuptiInterface::DeviceData_t *Devicedata = new TVMCuptiInterface::DeviceData_t();
    Devicedata->device_id = deviceID;
    catchCUDAError(cuDeviceGet(&(Devicedata->device), deviceID));
    catchCUDAError(cuDevicePrimaryCtxRetain(&(Devicedata->device_context), Devicedata->device));
    char name[64];
    cuDeviceGetName(name, sizeof(name) - 1, Devicedata->device);
    name[sizeof(name) - 1] = '\0';
    Devicedata->device_name = std::string(name);
    TVMCuptiInterface::CurrentDevice = Devicedata;
}

void TVMCuptiInterface::parse(std::string events[], std::string metrics[], int num_metrics, int num_events, int deviceID)
{
    std::cout<<"parsing"<<std::endl;
    catchCUDAError(cuInit(0));
    createDevice(deviceID);
    CUpti_MetricID *metricID[num_metrics];
    CUpti_EventID *eventID[num_events];
    getMetrics(metrics, num_metrics, metricID);
    getEvents(events, num_events, eventID);
    createEventData(eventID, metricID, num_metrics, num_events, metrics);
    std::cout<<"created event data"<<std::endl;
}

void TVMCuptiInterface::setup_CUPTI_Gathering()
{
    cudaDeviceSynchronize();
    // cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL);
    // cuptiActivityRegisterCallbacks(TVMCuptiInterface::allocate_Buffer, TVMCuptiInterface::get_CUPTI_Activity);
    //

    catchCUPTIError(cuptiSetEventCollectionMode(TVMCuptiInterface::CurrentDevice->device_context, CUPTI_EVENT_COLLECTION_MODE_KERNEL));
    if (TVMCuptiInterface::CurrentConfiguration->num_metrics > 0)
    {
        catchCUPTIError(cuptiMetricCreateEventGroupSets(TVMCuptiInterface::CurrentDevice->device_context, 
                                                        sizeof(CUpti_MetricID)*TVMCuptiInterface::CurrentConfiguration->num_metrics, 
                                                        TVMCuptiInterface::CurrentConfiguration->metricId, 
                                                        &(TVMCuptiInterface::CurrentConfiguration->metric)));
        TVMCuptiInterface::CurrentConfiguration->num_metric_passes = &(TVMCuptiInterface::CurrentConfiguration->metric->numSets);
    }
    std::cout<<"created metrics"<<std::endl;
    if (TVMCuptiInterface::CurrentConfiguration->num_events > 0)
    {
        catchCUPTIError(cuptiEventGroupSetsCreate(TVMCuptiInterface::CurrentDevice->device_context, 
                                                  sizeof(CUpti_EventID)*TVMCuptiInterface::CurrentConfiguration->num_events, 
                                                  TVMCuptiInterface::CurrentConfiguration->eventId, 
                                                  &(TVMCuptiInterface::CurrentConfiguration->event)));
        TVMCuptiInterface::CurrentConfiguration->num_events_passes = &(TVMCuptiInterface::CurrentConfiguration->event->numSets);
    } 
    std::cout<<"initalised event groups"<<std::endl;
}

void TVMCuptiInterface::start_CUPTI_Gathering()
{
    std::cout<<"starting gathering"<<std::endl;
    std::cout<<"event: "<<TVMCuptiInterface::CurrentConfiguration->num_events<<std::endl;
    std::cout<<"metric: "<<TVMCuptiInterface::CurrentConfiguration->num_metrics<<std::endl;
    if (TVMCuptiInterface::CurrentConfiguration->num_events > 0)
    {
        for(int i = 0; i < (int) TVMCuptiInterface::CurrentConfiguration->event->numSets; i++)
        { 
            uint32_t flag = 1;
            cuptiEventGroupSetAttribute(TVMCuptiInterface::CurrentConfiguration->event->sets->eventGroups[i], 
                                        CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES, sizeof(flag), &flag);
            cuptiEventGroupEnable(TVMCuptiInterface::CurrentConfiguration->event->sets->eventGroups[i]);
        }
    }
    std::cout<<"started event gathering"<<std::endl;
    std::cout<<"metric sets: "<<(int) TVMCuptiInterface::CurrentConfiguration->metric->numSets<<std::endl;
    if (TVMCuptiInterface::CurrentConfiguration->num_metrics > 0)
    {
        for(int i = 0; i < (int) TVMCuptiInterface::CurrentConfiguration->metric->numSets; i++)
        {
            uint32_t flag = 1; 
            
            cuptiEventGroupSetAttribute(TVMCuptiInterface::CurrentConfiguration->metric->sets->eventGroups[i], 
                                        CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES, sizeof(flag), &flag);
            cuptiEventGroupEnable(TVMCuptiInterface::CurrentConfiguration->metric->sets->eventGroups[i]);
        }
    }
    
    std::cout<<"started metric gathering"<<std::endl;
    cudaEventRecord(TVMCuptiInterface::CurrentConfiguration->Start, NULL);
}

std::string TVMCuptiInterface::stop_CUPTI_Gathering()
{
    std::cout<<"stopping collection"<<std::endl;
    json report = json::object();
    report["metrics"] = json::array();
    report["events"] = json::array();
    report["kernels"] = json::array();
    cudaEventRecord(TVMCuptiInterface::CurrentConfiguration->Stop, NULL);
    cudaDeviceSynchronize();
    std::vector<uint64_t*> eventValues;
    std::vector<CUpti_EventID*> eventIDs;
    std::vector<size_t> eventSizes;
    std::vector<size_t> eventNumbers;
    int numEventGroups;
    if (TVMCuptiInterface::CurrentConfiguration->num_events > 0)
    {
        numEventGroups = (int) (TVMCuptiInterface::CurrentConfiguration->metric->numSets + TVMCuptiInterface::CurrentConfiguration->event->numSets);
    }
    else
    {
        numEventGroups = (int) TVMCuptiInterface::CurrentConfiguration->metric->numSets;
    }
    for (int j = 0; j < numEventGroups; j++)
    {
        CUpti_EventGroup group;
        if (j < (int) TVMCuptiInterface::CurrentConfiguration->metric->numSets)
        {
            group = TVMCuptiInterface::CurrentConfiguration->metric->sets->eventGroups[j];
        }
        else
        {
            std::cout<<"event groups"<<std::endl;
            int i = j - ((int) TVMCuptiInterface::CurrentConfiguration->metric->numSets);
            group = TVMCuptiInterface::CurrentConfiguration->event->sets->eventGroups[i];
        }
        std::cout<<"gathering events"<<std::endl;
        uint32_t numInstnaces, numTotalInstances, numEvents;
        CUpti_EventDomainID groupDomain;
        size_t numInstanceSize = sizeof(uint32_t);
        size_t numEventsSize = sizeof(uint32_t);
        size_t numTotalInstancesSize = sizeof(uint32_t);
        size_t groupDomainSize = sizeof(groupDomain);

        catchCUPTIError(cuptiEventGroupGetAttribute(group, CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT, &numInstanceSize, &numInstnaces));
        catchCUPTIError(cuptiEventGroupGetAttribute(group, CUPTI_EVENT_GROUP_ATTR_EVENT_DOMAIN_ID, &groupDomainSize, &groupDomain));
        catchCUPTIError(cuptiDeviceGetEventDomainAttribute(TVMCuptiInterface::CurrentDevice->device, groupDomain, CUPTI_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT, 
                                                            &numTotalInstancesSize, &numTotalInstances));
        catchCUPTIError(cuptiEventGroupGetAttribute(group, CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS, &numEventsSize, &numEvents));
        
        size_t eventValueBufferSizeBytes = sizeof(uint64_t) * numInstnaces * numEvents;
        size_t eventIDSize = numEvents * sizeof(CUpti_EventID);
        uint64_t *eventValueBuffer = (uint64_t*) malloc(eventValueBufferSizeBytes);
        CUpti_EventID *eventArray = new CUpti_EventID();
        size_t *eventNumEventIds = new size_t();

        catchCUPTIError(cuptiEventGroupReadAllEvents(group, CUPTI_EVENT_READ_FLAG_NONE, &eventValueBufferSizeBytes, eventValueBuffer, 
                                                    &eventIDSize, eventArray, eventNumEventIds));

        size_t *normalizedBuffer = (size_t*) calloc(sizeof(size_t), numEvents);
        size_t sum;
        for (int i = 0; i < numEvents; i++) //normalize
        {
            for (int j = 0; j < (int) numInstnaces; j++) // itterates for the number of times each event is called
            {
                int eventValueIndex = j + (i*numInstnaces);
                sum += eventValueBuffer[eventValueIndex];
            }
            normalizedBuffer[i] = (sum*numTotalInstances)/numInstnaces;
        }
        eventValues.push_back(normalizedBuffer);
        eventIDs.push_back(eventArray);
        eventSizes.push_back(numInstnaces*numEvents);
        eventNumbers.push_back(numEvents);
    }
    
    float *Duration = new float();
    cudaEventElapsedTime(Duration, TVMCuptiInterface::CurrentConfiguration->Start, TVMCuptiInterface::CurrentConfiguration->Stop);
    CUpti_EventGroup MetricGroup = TVMCuptiInterface::CurrentConfiguration->metric->sets->eventGroups[0];
    
    std::cout<<"time: "<<*Duration<<" ms"<<std::endl;
    for (int i = 0; i < TVMCuptiInterface::CurrentConfiguration->num_metrics; i++)
    {
        CUpti_MetricValue *MetricValue = new CUpti_MetricValue();
        catchCUPTIError(cuptiMetricGetValue(TVMCuptiInterface::CurrentDevice->device, 
                                            TVMCuptiInterface::CurrentConfiguration->metricId[i],
                                            eventSizes[0]*sizeof(CUpti_EventID),
                                            eventIDs[0],
                                            eventSizes[0]*sizeof(uint64_t),
                                            eventValues[0],
                                            (uint64_t) &Duration,
                                            MetricValue));

        json metric = {{"metric ID", (uint32_t) TVMCuptiInterface::CurrentConfiguration->metricId[i]}, 
                        {"Name", TVMCuptiInterface::CurrentConfiguration->metric_names[i]}};
        CUpti_MetricValueKind metricValueType;
        size_t valueTypeSize = sizeof(metricValueType);
        catchCUPTIError(cuptiMetricGetAttribute(TVMCuptiInterface::CurrentConfiguration->metricId[i], 
                                                CUPTI_METRIC_ATTR_VALUE_KIND, &valueTypeSize, &metricValueType));

        if (metricValueType == CUPTI_METRIC_VALUE_KIND_DOUBLE)
        {
            metric["Type"] = std::string("double");
            metric["Value"] = MetricValue->metricValueDouble;
        }
        else if (metricValueType == CUPTI_METRIC_VALUE_KIND_INT64)
        {
            metric["Type"] = std::string("int64_t");
            metric["Value"] = MetricValue->metricValueInt64;
        }
        else if (metricValueType == CUPTI_METRIC_VALUE_KIND_PERCENT)
        {
            metric["Type"] = std::string("percent");
            metric["Value"] = MetricValue->metricValuePercent;
        }
        else if (metricValueType == CUPTI_METRIC_VALUE_KIND_THROUGHPUT)
        {
            metric["Type"] = std::string("throughput");
            metric["Value"] = MetricValue->metricValueThroughput;
            break;
        }
        else if (metricValueType == CUPTI_METRIC_VALUE_KIND_UINT64)
        {
            metric["Type"] = std::string("uint64_t");
            metric["Value"] = MetricValue->metricValueUint64;
        }
        else if (metricValueType == CUPTI_METRIC_VALUE_KIND_UTILIZATION_LEVEL)
        {
            metric["Type"] = std::string("utilization_level");
            metric["Value"] = MetricValue->metricValueUtilizationLevel;
        }
        else
        {
            std::cerr<<"ERROR: Undefined value type\n";
            exit(-1);
        }
        report["metrics"].push_back(metric);
        free(MetricValue);
    }

    for(int i = 0; i < eventIDs.capacity(); i++)
    {
        CUpti_EventID *e_ids = (CUpti_EventID*) calloc(sizeof(CUpti_EventID), eventNumbers[i]);
        uint64_t *e_values = (uint64_t*) calloc(sizeof(uint64_t), eventSizes[i]);
        e_ids = eventIDs[i];
        e_values = eventValues[i];
        for (int j = 0; j < eventNumbers[i]; j++)
        {
            size_t eventNameSize = sizeof(char) *100;   
            char * eventName = (char*) malloc(eventNameSize);
            catchCUPTIError(cuptiEventGetAttribute(e_ids[j], CUPTI_EVENT_ATTR_NAME, &eventNameSize, eventName));
            for (int p = 0; p < eventSizes[i]/eventNumbers[i]; p++)
            {
                json event = {{"event ID", (uint32_t) *eventIDs[i]}, {"Name", std::string(eventName)}, 
                            {"Type", std::string("double")}, {"Value", (uint64_t) e_values[p]}};
                report["events"].push_back(event);
            }
            free(eventName);
        }
        free(e_values);
        free(e_ids);
    }

    for(int i = 0; i < TVMCuptiInterface::Currentkernels.capacity(); i++)
    {
        TVMCuptiInterface::Kernel_t * Kernel = TVMCuptiInterface::Currentkernels[i];
        json kernel = {{"Name", std::string(Kernel->name), {"ContextID", (uint32_t) Kernel->contextID}},
                        {"Start", (uint64_t) Kernel->start},{"End", (uint64_t) Kernel->end},
                        {"Completed", (uint64_t) Kernel->completed},{"StreamID", (uint32_t) Kernel->streamID},
                        {"GridID", (int64_t) Kernel->gridID}, {"GridX", (int32_t) Kernel->gridX}, {"Gridy", (int32_t) Kernel->gridY},
                        {"GridZ", (int32_t) Kernel->gridZ},{"BlockX", (int32_t) Kernel->blockX},{"BlockY", (int32_t) Kernel->blockY},
                        {"BlockZ", (int32_t) Kernel->gridZ},{"StaticSharedMemory", (int32_t) Kernel->staticSharedMemory},
                        {"DynamicSharedMemory", (int32_t) Kernel->dynamicSharedMemory},{"LocalMemoryPerThread", (int32_t) Kernel->localMemoryPerThread},
                        {"PartitionedGlobalCacheRequested", (CUpti_ActivityPartitionedGlobalCacheConfig) Kernel->partitionedGlobalCacheRequested},
                        {"PartitionedGlobalCacheExecuted", (CUpti_ActivityPartitionedGlobalCacheConfig) Kernel->partitionedGlobalCacheExecuted}};
        report["kernels"].push_back(kernel);
        free(Kernel);
    }
    return report.dump();
}

void TVMCuptiInterface::Insert_CUPTI_Config(std::string input)
{
    json config = json::parse(input);
    if (config != NULL)
    {
        std::vector<std::string> metrics;
        std::vector<std::string> events;
        for(int i = 0;i < config["metrics"].size(); i++)
        {
            metrics.push_back(config["metrics"].at(i)["name"]);
        }
        for(int i = 0;i < config["events"].size(); i++)
        {
            events.push_back(config["events"].at(i)["name"]);
        }
        TVMCuptiInterface::parse(events.data(), metrics.data(), metrics.size(), events.size(), 0);
    }
    else
    {
        std::cerr<<"No config";
        exit(-1);
    }
}

void TVMCuptiInterface::helloWorld()
{
    std::cout<<"Hello World"<<std::endl;
}

TVMCuptiInterface::~TVMCuptiInterface(){}
