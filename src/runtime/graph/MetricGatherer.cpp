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

    std::string collect(std::string config) //dev id = 0
    {
      json Config = json::parse(config); //gets metric/event config
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
      return operationData.dump();
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

PackedFunc MetricGatherer::GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) 
{
  // return member functions during query.
  if (name == "collect") 
  {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) 
    {
      *rv = this->MetricGatherer::collect(args[0]); //run thing
    });
  } 
  else 
  {
    std::cerr<<"Not a valid function";
    exit(-1);
  }
}

Module MetricGathererGraphCreate(const std::string& sym_json, const tvm::runtime::Module& m, const std::vector<TVMContext>& ctxs, PackedFunc lookup_linked_param_func, const std::string& cupti_json) 
{
  auto exec = make_object<MetricGatherer>();
  exec->Init(sym_json, m, ctxs, lookup_linked_param_func);
  exec->collect(cupti_json);
  return Module(exec);
}


TVM_REGISTER_GLOBAL("tvm.graph_runtime_sampling.create").set_body([](TVMArgs args, TVMRetValue* rv) 
{
  ICHECK_GE(args.num_args, 5) << "The expected number of arguments for graph_runtime.create is "
                                 "at least 5, but it has "
                              << args.num_args;
  PackedFunc lookup_linked_param_func;
  int ctx_start_arg = 2;
  if (args[2].type_code() == kTVMPackedFuncHandle) 
  {
    lookup_linked_param_func = args[2];
    ctx_start_arg++;
  }

  *rv = MetricGathererGraphCreate(args[0], args[1], GetAllContext(args, ctx_start_arg), lookup_linked_param_func, args[2]);
});

}
}