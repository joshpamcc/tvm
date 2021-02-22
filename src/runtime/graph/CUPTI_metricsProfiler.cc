#include "CUPTI_metricsProfiler.h"
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

using json = nlohmann::json;
#define BUF_SIZE (16 * 1024)
#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)                                            \
  (((uintptr_t) (buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t) (buffer) & ((align)-1))) : (buffer))

TVMCuptiInterface::EventData_t *TVMCuptiInterface::CurrentConfiguration = NULL;
TVMCuptiInterface::DeviceData_t *TVMCuptiInterface::CurrentDevice = NULL;
TVMCuptiInterface::EventValues_t *TVMCuptiInterface::CurrentEventValues = NULL;
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

void catchNVMLError(nvmlReturn_t error)
{
    if (error != NVML_SUCCESS)
    {
        const char *errMsg = (char*) malloc(sizeof(char)*100);
        errMsg = nvmlErrorString(error);
        std::cout<<"Error For NVML API: "<<error<<" msg: "<<errMsg<<std::endl;
        exit(-1);
    }
}

void CUPTIAPI allocate_Buffer(uint8_t **buffer, size_t *size, size_t *maxNumRecords)
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

void CUPTIAPI get_CUPTI_Activity(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize)
{
    CUptiResult status;
    CUpti_Activity *record = NULL;
    CUpti_ActivityKernel4 * kernel;
    if (validSize > 0)
    {
        status = cuptiActivityGetNextRecord(buffer, size, &record);
        kernel = (CUpti_ActivityKernel4 *) record;
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
        std::cout<<"met: "<<metricNames[i]<<std::endl;
    }
    for (int i = 0; i < num_events; i++)
    {
        eventData->eventId[i] = *event_ID[i];
        std::cout<<"ev:"<<eventData->eventId[i]<<std::endl;
    }
    for (int i = 0; i < num_metrics; i++)
    {
        eventData->metricId[i] = *metric_ID[i];
        std::cout<<"met:"<<eventData->metricId[i]<<std::endl;
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
        //std::cout<<events[i]<<std::endl;
    }
}

void TVMCuptiInterface::createDevice(int deviceID) //creates and gets device context for the provided device id
{
    TVMCuptiInterface::DeviceData_t *Devicedata = new TVMCuptiInterface::DeviceData_t();
    Devicedata->device_id = deviceID;
    catchCUDAError(cuDeviceGet(&(Devicedata->device), deviceID));
    catchCUDAError(cuDevicePrimaryCtxRetain(&(Devicedata->device_context), Devicedata->device));
    catchNVMLError(nvmlInit());
    catchNVMLError(nvmlDeviceGetHandleByIndex(deviceID, &Devicedata->NVML_Device));
    cudaSetDevice(0);
   // catchNVMLError(nvmlShutdown());
    char name[64];
    cuDeviceGetName(name, sizeof(name) - 1, Devicedata->device);
    name[sizeof(name) - 1] = '\0';
    Devicedata->device_name = std::string(name);
    TVMCuptiInterface::CurrentDevice = Devicedata;
    //std::cout<<"device created"<<std::endl;
}

void TVMCuptiInterface::parse(std::string events[], std::string metrics[], int num_metrics, int num_events)
{
    //std::cout<<"parsing"<<std::endl;
    CUpti_MetricID *metricID[num_metrics];
    CUpti_EventID *eventID[num_events];
    getMetrics(metrics, num_metrics, metricID);
    getEvents(events, num_events, eventID);
    createEventData(eventID, metricID, num_metrics, num_events, metrics);
    //std::cout<<"created event data"<<std::endl;
}

void TVMCuptiInterface::newConfig()
{
    cudaDeviceSynchronize();
    TVMCuptiInterface::CurrentConfiguration->event = NULL;
    TVMCuptiInterface::CurrentConfiguration->metric = NULL;
    TVMCuptiInterface::CurrentEventValues = NULL;
    TVMCuptiInterface::Currentkernels.clear();   
}

void TVMCuptiInterface::setup_Kernel_Gathering()
{
    //std::cout<<"kernels setup"<<std::endl;
    cudaDeviceSynchronize();
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL);
    cuptiActivityRegisterCallbacks(allocate_Buffer, get_CUPTI_Activity);
    catchCUPTIError(cuptiSetEventCollectionMode(TVMCuptiInterface::CurrentDevice->device_context, CUPTI_EVENT_COLLECTION_MODE_KERNEL)); 
    cudaDeviceSynchronize(); 
}

std::string TVMCuptiInterface::stop_Kernel_Gathering()
{
    cudaDeviceSynchronize();
    cuptiActivityFlushAll(0);
    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_KERNEL);
    cudaDeviceSynchronize();
    //"kernel amount: "<<TVMCuptiInterface::Currentkernels.size()<<std::endl;
    json report = json::object();
    report["kernels"] = json::array();
    for(int i = 0; i < TVMCuptiInterface::Currentkernels.size(); i++)
    {
        TVMCuptiInterface::Kernel_t * Kernel = TVMCuptiInterface::Currentkernels[i];
        json kernel = {{"Name", std::string(Kernel->name)}, {"ContextID", (uint32_t) Kernel->contextID},
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
    //std::cout<<"Gathered kernel data"<<std::endl;
    TVMCuptiInterface::Currentkernels.clear(); 
    return report.dump();
}

void TVMCuptiInterface::setup_CUPTI_Metrics(int index)
{
    cudaDeviceSynchronize();
   // catchNVMLError(nvmlInit());
    catchCUPTIError(cuptiSetEventCollectionMode(TVMCuptiInterface::CurrentDevice->device_context, CUPTI_EVENT_COLLECTION_MODE_KERNEL)); 
    TVMCuptiInterface::CurrentConfiguration->metric = NULL;
    TVMCuptiInterface::CurrentEventValues = NULL;
    if (TVMCuptiInterface::CurrentConfiguration->num_metrics > 0)
    {
        catchCUPTIError(cuptiMetricCreateEventGroupSets(TVMCuptiInterface::CurrentDevice->device_context, 
                                                        sizeof(CUpti_MetricID), 
                                                        &TVMCuptiInterface::CurrentConfiguration->metricId[index], 
                                                        &(TVMCuptiInterface::CurrentConfiguration->metric)));
        TVMCuptiInterface::CurrentConfiguration->metric;
        TVMCuptiInterface::CurrentConfiguration->num_metric_passes = &(TVMCuptiInterface::CurrentConfiguration->metric->numSets);
    }
    TVMCuptiInterface::EventValues_t *data = new EventValues_t();
    TVMCuptiInterface::CurrentEventValues = data;
}

void TVMCuptiInterface::setup_CUPTI_Events()
{
    cudaDeviceSynchronize();
    TVMCuptiInterface::CurrentConfiguration->event = NULL;
    if (TVMCuptiInterface::CurrentConfiguration->num_events > 0)
    {
        catchCUPTIError(cuptiEventGroupSetsCreate(TVMCuptiInterface::CurrentDevice->device_context, 
                                                  sizeof(CUpti_EventID)*TVMCuptiInterface::CurrentConfiguration->num_events, 
                                                  TVMCuptiInterface::CurrentConfiguration->eventId, 
                                                  &(TVMCuptiInterface::CurrentConfiguration->event)));
        TVMCuptiInterface::CurrentConfiguration->num_events_passes = &(TVMCuptiInterface::CurrentConfiguration->event->numSets);
    } 
    //std::cout<<"initalised event groups"<<std::endl;
}

void TVMCuptiInterface::start_CUPTI_Gathering(int metGroupSet, int metGroup, bool eventsSampled)
{
    cudaDeviceSynchronize();
    catchCUPTIError(cuptiSetEventCollectionMode(TVMCuptiInterface::CurrentDevice->device_context, CUPTI_EVENT_COLLECTION_MODE_KERNEL));
    //std::cout<<"starting gathering"<<std::endl;
    if (eventsSampled == 0)
    {
        if (TVMCuptiInterface::CurrentConfiguration->num_events > 0)
        {
            for(int i = 0; i < (int) TVMCuptiInterface::CurrentConfiguration->event->numSets; i++)
            { 
                uint32_t flag = 1;
                catchCUPTIError(cuptiEventGroupSetAttribute(TVMCuptiInterface::CurrentConfiguration->event->sets->eventGroups[i], 
                                            CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES, sizeof(flag), &flag));
                catchCUPTIError(cuptiEventGroupEnable(TVMCuptiInterface::CurrentConfiguration->event->sets->eventGroups[i]));
            }
        }
    }
    if (TVMCuptiInterface::CurrentConfiguration->num_metrics > 0)
    {
        uint32_t flag = 1; 
        catchCUPTIError(cuptiEventGroupSetAttribute(TVMCuptiInterface::CurrentConfiguration->metric->sets[metGroupSet].eventGroups[metGroup], 
                                    CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES, sizeof(flag), &flag));
        
        catchCUPTIError(cuptiEventGroupEnable(TVMCuptiInterface::CurrentConfiguration->metric->sets[metGroupSet].eventGroups[metGroup]));
    }
    cudaDeviceSynchronize();
    //std::cout<<"started metric gathering"<<std::endl;
    if (metGroup == 0)
    {
        cudaEventRecord(TVMCuptiInterface::CurrentConfiguration->Start, NULL);
    }
    
}

std::string TVMCuptiInterface::stop_CUPTI_Gathering(int metGroupSet, int metGroup, bool eventsSampled, int metIndex)
{
    cudaDeviceSynchronize();
    
    int numEventGroups = 1;
    if (eventsSampled == 0)
    {
        if (TVMCuptiInterface::CurrentConfiguration->num_events > 0)
        {
            numEventGroups = 2;
        }
    }

    
    //std::cout<<"stopping collection"<<std::endl;
    json report = json::object();
    report["metrics"] = json::array();
    report["events"] = json::array();
    report["duration"] = json::array();
    report["NVML"] = json::array();
    cudaDeviceSynchronize();
    // std::vector<uint64_t*> eventValues;
    // std::vector<CUpti_EventID*> eventIDs;
    // std::vector<size_t> eventSizes;
    // std::vector<size_t> eventNumbers;

    for (int j = 0; j < numEventGroups; j++)
    {
        CUpti_EventGroup group;
        if (j < 1)
        {
            group = TVMCuptiInterface::CurrentConfiguration->metric->sets[metGroupSet].eventGroups[metGroup];
        }
        else
        {
            group = TVMCuptiInterface::CurrentConfiguration->event->sets->eventGroups[0];
        }
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
        cudaDeviceSynchronize();
        // std::cout<<"num event ids "<<numEvents<<std::endl;
        // for (int i = 0; i < numEvents;i++)
        // {
        //     std::cout<<"event id: "<<eventArray[i]<<std::endl;
        // }
        cudaEventRecord(TVMCuptiInterface::CurrentConfiguration->Stop, NULL);
        CUptiResult res = cuptiEventGroupDisable(group);
        //std::cout<<"disable res for group: "<<j<<" : "<<res<<std::endl;
        //std::cout<<"gathered eventGroup Data"<<std::endl;

        size_t *normalizedBuffer = (size_t*) calloc(sizeof(size_t), numEvents);
        size_t sum;
        for (int i = 0; i < numEvents; i++) //normalize
        {
            for (int j = 0; j < (int) numInstnaces; j++) // itterates for the number of times each event is called
            {
                //int eventValueIndex = j + (i*numInstnaces);
                sum += eventValueBuffer[j];
            }
            normalizedBuffer[i] = (sum*numTotalInstances)/numInstnaces;
        }
        TVMCuptiInterface::CurrentEventValues->eventValues.push_back(normalizedBuffer);
        TVMCuptiInterface::CurrentEventValues->eventIDs.push_back(eventArray);
        TVMCuptiInterface::CurrentEventValues->eventSizes.push_back(numInstnaces*numEvents);
        TVMCuptiInterface::CurrentEventValues->eventNumbers.push_back(numEvents);
    }
    
    // for (int i = 0; i < TVMCuptiInterface::CurrentConfiguration->num_metrics; i++)
    //
       // catchCUPTIError(cuptiEventGroupGetAttribute(evGroup->sets->eventGroups[0], CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS, &numEventsSize, &numEvents));
    //std::cout<<"cur: "<<metGroupSet<<"/"<<(int) *TVMCuptiInterface::CurrentConfiguration->num_metric_passes -1<<std::endl;
    if (metGroupSet >= (int) *TVMCuptiInterface::CurrentConfiguration->num_metric_passes -1)
    {
        if (metGroup >= TVMCuptiInterface::CurrentConfiguration->metric->sets[metGroupSet].numEventGroups-1)
        {
            
           // catchNVMLError(nvmlShutdown());
            float *Duration = new float();
            cudaEventElapsedTime(Duration, TVMCuptiInterface::CurrentConfiguration->Start, TVMCuptiInterface::CurrentConfiguration->Stop);
            int noOfEvents = 0;
            for (int l = 0; l < TVMCuptiInterface::CurrentEventValues->eventNumbers.size(); l++)
            {
                noOfEvents += TVMCuptiInterface::CurrentEventValues->eventNumbers[l];
                //std::cout<<"number of events: "<<l<<" no: "<<TVMCuptiInterface::CurrentEventValues->eventNumbers[l]<<std::endl;
            }
            //std::cout<<"time: "<<*Duration<<" ms"<<std::endl;
            uint64_t *eventValues = (uint64_t*) calloc(sizeof(uint64_t), noOfEvents);
            CUpti_EventID *eventIDs = (CUpti_EventID*) calloc(sizeof(CUpti_EventID), noOfEvents);
            size_t *eventSizes = (size_t*) calloc(sizeof(size_t), noOfEvents);
            
            int index = 0; 
            for(int p = 0; p < TVMCuptiInterface::CurrentEventValues->eventIDs.size(); p++)
            {
                for (int z = 0; z < TVMCuptiInterface::CurrentEventValues->eventNumbers[p]; z++)
                {
                    eventIDs[index + z] = TVMCuptiInterface::CurrentEventValues->eventIDs[p][z];
                }
                index += TVMCuptiInterface::CurrentEventValues->eventNumbers[p];
            }
            index = 0;
            for(int p = 0; p < TVMCuptiInterface::CurrentEventValues->eventValues.size(); p++)
            {
                for (int z = 0; z < TVMCuptiInterface::CurrentEventValues->eventNumbers[p]; z++)
                {
                    eventValues[index + z] = TVMCuptiInterface::CurrentEventValues->eventValues[p][z];
                }
                index += TVMCuptiInterface::CurrentEventValues->eventNumbers[p];
            }
            int eventSize = 0;

            for(int p = 0; p < TVMCuptiInterface::CurrentEventValues->eventSizes.size(); p++)
            {
                eventSize += TVMCuptiInterface::CurrentEventValues->eventSizes[p];
            }

            
            // std::cout<<";;;;;;;;;;"<<std::endl;
            // for (int l = 0; l < noOfEvents; l++)
            // {
            //     std::cout<<eventIDs[l]<<std::endl;
            // }

            CUpti_MetricValue *MetricValue = new CUpti_MetricValue();
    
            CUptiResult res = cuptiMetricGetValue(TVMCuptiInterface::CurrentDevice->device, 
                                                TVMCuptiInterface::CurrentConfiguration->metricId[metIndex],
                                                eventSize*sizeof(CUpti_EventID),
                                                eventIDs,
                                                eventSize*sizeof(uint64_t),
                                                eventValues,
                                                (uint64_t) &Duration,
                                                MetricValue);
            if (res != CUPTI_SUCCESS)
            {
                std::cout<<"could not get values for metric: "<<TVMCuptiInterface::CurrentConfiguration->metric_names[metIndex]<<std::endl;
                const char *errMsg = (char*) malloc(sizeof(char)*100);
                cuptiGetResultString(res,&errMsg);
                std::cout<<"ERR: "<<errMsg<<std::endl;
                exit(-1);
            }

            json metric = {{"metric ID", (uint32_t) TVMCuptiInterface::CurrentConfiguration->metricId[metIndex]}, 
                            {"Name", TVMCuptiInterface::CurrentConfiguration->metric_names[metIndex]},
                            {"Start", (uint64_t) TVMCuptiInterface::CurrentConfiguration->Start},{"Stop", (uint64_t) TVMCuptiInterface::CurrentConfiguration->Stop}};
            CUpti_MetricValueKind metricValueType;
            size_t valueTypeSize = sizeof(metricValueType);
            catchCUPTIError(cuptiMetricGetAttribute(TVMCuptiInterface::CurrentConfiguration->metricId[metIndex], 
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
            json dur = {{"Name",TVMCuptiInterface::CurrentConfiguration->metric_names[metIndex]},{"Duration",*Duration}};
            report["duration"].push_back(dur);
            free (Duration);
            free(MetricValue);

            for(int i = 0; i < TVMCuptiInterface::CurrentEventValues->eventIDs.size(); i++)
            {
                CUpti_EventID *e_ids = (CUpti_EventID*) calloc(sizeof(CUpti_EventID), TVMCuptiInterface::CurrentEventValues->eventNumbers[i]);
                uint64_t *e_values = (uint64_t*) calloc(sizeof(uint64_t), TVMCuptiInterface::CurrentEventValues->eventSizes[i]);
                e_ids = TVMCuptiInterface::CurrentEventValues->eventIDs[i];
                e_values = TVMCuptiInterface::CurrentEventValues->eventValues[i];
                for (int j = 0; j < TVMCuptiInterface::CurrentEventValues->eventNumbers[i]; j++)
                {
                    size_t eventNameSize = sizeof(char) * 1000;   
                    char * eventName = (char*) malloc(eventNameSize);
                    catchCUPTIError(cuptiEventGetAttribute(*TVMCuptiInterface::CurrentEventValues->eventIDs[i], CUPTI_EVENT_ATTR_NAME, &eventNameSize, eventName));
                    for (int p = 0; p < TVMCuptiInterface::CurrentEventValues->eventSizes[i]/TVMCuptiInterface::CurrentEventValues->eventNumbers[i]; p++)
                    {
                        json event = {{"event ID", (uint32_t) *TVMCuptiInterface::CurrentEventValues->eventIDs[i]}, {"Name", std::string(eventName)}, 
                                    {"Type", std::string("uint64_t")}, {"Value", (uint64_t) e_values[p]}};
                        report["events"].push_back(event);
                    }
                    free(eventName);
                }
                free(e_values);
                free(e_ids);
            }
            
            free (eventIDs);
            free (eventValues);
            free (eventSizes);

            TVMCuptiInterface::NVMLData_t *NVMLData = new NVMLData_t();
            NVMLData->powerLimit = (uint*) malloc(100);
            NVMLData->temperature = (uint*) malloc(100);
            NVMLData->powerUsage = (uint*) malloc(100);
            catchNVMLError(nvmlDeviceGetTemperature(TVMCuptiInterface::CurrentDevice->NVML_Device, NVML_TEMPERATURE_GPU, NVMLData->temperature));
            catchNVMLError(nvmlDeviceGetPowerUsage(TVMCuptiInterface::CurrentDevice->NVML_Device, NVMLData->powerUsage));
            catchNVMLError(nvmlDeviceGetEnforcedPowerLimit(TVMCuptiInterface::CurrentDevice->NVML_Device, NVMLData->powerLimit));

            json temp = {{"Name","Temperature"},{"Value",(uint) *NVMLData->temperature}};
            json powerUsage = {{"Name","Power Usage"},{"Value", (uint) *NVMLData->powerUsage}};
            json powerLimit = {{"Name","Power Limit"},{"Value", (uint) *NVMLData->powerLimit}};
            report["NVML"].push_back(temp);
            report["NVML"].push_back(powerUsage);
            report["NVML"].push_back(powerLimit);
            free(NVMLData);
            NVMLData = NULL;
            // eventValues.clear();
            // eventIDs.clear();
            // eventSizes.clear();
            // eventNumbers.clear();
            //std::cout<<"--------------"<<std::endl;
            free(TVMCuptiInterface::CurrentEventValues);
            TVMCuptiInterface::CurrentEventValues = NULL;
            catchCUPTIError(cuptiEventGroupSetsDestroy(TVMCuptiInterface::CurrentConfiguration->metric));
            cudaDeviceSynchronize();
            return report.dump();
        }
    }
    else
    {
        return "wait";
    }
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
        TVMCuptiInterface::parse(events.data(), metrics.data(), metrics.size(), events.size());
    }
    else
    {
        std::cerr<<"No config";
        exit(-1);
    }
}

TVMCuptiInterface::~TVMCuptiInterface(){}
