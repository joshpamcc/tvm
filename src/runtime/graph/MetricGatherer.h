#include "graph_runtime.h"
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
namespace tvm {
namespace runtime {

class TVM_DLL MetricGatherer : public GraphRuntime 
{
    public:

        virtual std::string collect(std::string config);
};
}
}