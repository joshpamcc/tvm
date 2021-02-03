#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <nlohmann/json.hpp>

#ifndef TVM_RUNTIME_GRAPH_METRICGATHERER_H_
#define TVM_RUNTIME_GRAPH_METRICGATHERER_H_

#include <dlpack/dlpack.h>
#include <dmlc/json.h>
#include <dmlc/memory_io.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using json = nlohmann::json;
namespace tvm {
namespace runtime {

class TVM_DLL MetricGatherer : public ModuleNode
{

    struct OpArgs 
    {
        std::vector<DLTensor> args;
        std::vector<TVMValue> arg_values;
        std::vector<int> arg_tcodes;
        std::vector<int64_t> shape_data;
    };

    public:

    const char* type_key() const final{ return "MetricGather"; }

    void run();

    json runOP(int index);

    std::string collect(std::string config);
    
};

}
}
#endif