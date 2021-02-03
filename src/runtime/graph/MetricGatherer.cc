// #include "MetricGatherer.h"
// #include <nlohmann/json.hpp>
// #include <tvm/runtime/container.h>
// #include <tvm/runtime/device_api.h>
// #include <tvm/runtime/ndarray.h>
// #include <tvm/runtime/packed_func.h>
// #include <tvm/runtime/registry.h>
// #include <tvm/runtime/serializer.h>
// #include <chrono>
// #include "Sampling/CUPTI_metricsProfiler.h"
// using json = nlohmann::json;
// namespace tvm {
// namespace runtime {

// void MetricGatherer::run()
// {
//     for (int i = 0; i < op_execs_.size(); i++)
//     {
//         if (op_execs_[i]) op_execs_[i]();
//     }
// }

// std::string MetricGatherer::collect(std::string config) //dev id = 0
// {
//   json Config = json::parse(config); //gets metric/event config
//   json operationData = json::object();
//   operationData["operation"] = json::array();
//   operationData["CUPTI"] = json::array();
//   run();
//   std::chrono::time_point<std::chrono::high_resolution_clock, std::chrono::nanoseconds> tbegin, tend;
//   tbegin = std::chrono::high_resolution_clock::now();
//   for(int i = 0; i < op_execs_.size(); i++)
//   {
//     json value = runOP(i);
//     operationData["operation"].push_back(value["operation"]);
//     operationData["CUPTI"].push_back(value["CUPTI"]);
//   }
//   tend = std::chrono::high_resolution_clock::now();
//   double duration = std::chrono::duration_cast<std::chrono::duration<double>>(tend - tbegin).count(); //duration
//   json op = {{"op","total"},{"Duration",duration}};
//   operationData["operation"].push_back(op);
//   return operationData.dump();
// }

// json MetricGatherer::runOP(int index)
// {
//     auto op_tbegin = std::chrono::high_resolution_clock::now(); //op start time
//     TVMCuptiInterface::setup_CUPTI_Gathering();
//     TVMCuptiInterface::start_CUPTI_Gathering();
//     op_execs_[index](); //run op
//     const TVMContext& ctx = data_entry_[entry_id(index, 0)]->ctx; //get the operation context
//     TVMSynchronize(ctx.device_type, ctx.device_id, nullptr); //ensure op is finished
//     json CUPTI_Data = TVMCuptiInterface::stop_CUPTI_Gathering();
//     auto op_tend = std::chrono::high_resolution_clock::now(); //op end time
//     double duration = std::chrono::duration_cast<std::chrono::duration<double>>(op_tend - op_tbegin).count(); //duration
//     json op = {{"op",index}, {"Duration", duration}};
//     json rv = json::object();
//     rv["operation"] = json::array();
//     rv["CUPTI"] = json::array();
//     rv["operation"].push_back(op);
//     rv["CUPTI"].push_back(CUPTI_Data);
//     return rv;
// }

// }
// }