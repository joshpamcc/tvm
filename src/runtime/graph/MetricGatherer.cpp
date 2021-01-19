#include "graph_runtime.h"
#include <nlohmann/json.hpp>
#include <chrono>
#include "Sampling/CUPTI_metricsProfiler.h"
using json = nlohmann::json;
namespace tvm {
namespace runtime {


class MetricGatherer : public GraphRuntime 
{
 public:
 
    void run()
    {
        for (int i = 0; i < op_execs_.size(); i++)
        {
            if (op_execs_[i]) op_execs_[i]();
        }
    }

    void setup(json &config, TVMContext ctx) //dev id = 0
    {
        json operationData = json::object();
        operationData["operation"] = json::array();
        operationData["CUPTI"] = json::array();
        run();
        std::chrono::time_point<std::chrono::high_resolution_clock, std::chrono::nanoseconds> tbegin, tend;
        tbegin = std::chrono::high_resolution_clock::now();
        for(int i = 0; i < op_execs_.size(); i++)
        {
            runOP(i, operationData);
        }
        tend = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration_cast<std::chrono::duration<double> >(tend - tbegin).count(); //duration
        json op = {{"op","total"},{"Duration",duration}};
        operationData["operation"].push_back(op);
    }

    void runOP(int index, json object)
    {
        auto op_tbegin = std::chrono::high_resolution_clock::now(); //op start time
        TVMCuptiInterface::setup_CUPTI_Gathering();
        TVMCuptiInterface::start_CUPTI_Gathering();
        op_execs_[index](); //run op
        const TVMContext& ctx = data_entry_[entry_id(index, 0)]->ctx; //get the operation context
        TVMSynchronize(ctx.device_type, ctx.device_id, nullptr); //ensure op is finished
        json CUPTI_Data = TVMCuptiInterface::stop_CUPTI_Gathering();
        auto op_tend = std::chrono::high_resolution_clock::now(); //op end time
        double duration = std::chrono::duration_cast<std::chrono::duration<double> >(op_tend - op_tbegin).count(); //duration
        json op = {{"op",index}, {"Duration", duration}};
        object["operation"].push_back(op);
        object["CUPTI"].push_back(CUPTI_Data);
    }

    /*!
    * \brief GetFunction Get the function based on input.
    * \param name The function which needs to be invoked.
    * \param sptr_to_self Packed function pointer.
    */
    PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self);
};

PackedFunc MetricGatherer::GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) {
  // return member functions during query.
  if (name == "setup") 
  {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) 
    {
      *rv = this->MetricGatherer::setup(*args[0], args[1]); //run thing
    });
  } 
  else 
  {
      //err
    return GraphRuntime::GetFunction(name, sptr_to_self);
  }
}


}
}