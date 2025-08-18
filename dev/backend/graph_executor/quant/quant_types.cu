#include "quant/quant_types.cuh"
namespace quant {
static QuantCache g_cache;
static RuntimeFlags g_flags;
QuantCache& cache(){ return g_cache; }
RuntimeFlags& runtime(){ return g_flags; }
} // namespace quant
