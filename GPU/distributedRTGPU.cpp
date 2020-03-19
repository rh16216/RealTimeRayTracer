//compile using: g++ -std=c++0x -I$CPATH -I/sw/lang/cuda_10.1.105/NVIDIA-OptiX-SDK-7.0.0-linux64/include -o distributedRTGPU distributedRTGPU.cpp -L/sw/lang/cuda_10.1.105/lib64 -ldl -lutil -lcublas -lcudart

#include "distributedRTGPU.h"


template <typename T>
struct SbtRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenData> RayGenSbtRecord;
typedef SbtRecord<MissData>   MissSbtRecord;
typedef SbtRecord<int>        HitGroupSbtRecord;

struct Vertex
{
    float x, y, z, pad;
};


const int32_t TRIANGLE_COUNT = 32;
const int32_t MAT_COUNT      = 4;

const static std::array<Vertex, TRIANGLE_COUNT* 3> g_vertices =
{  {
    // Floor  -- white lambert
    {    0.0f,    0.0f,    0.0f, 0.0f },
    {    0.0f,    0.0f,  559.2f, 0.0f },
    {  556.0f,    0.0f,  559.2f, 0.0f },
    {    0.0f,    0.0f,    0.0f, 0.0f },
    {  556.0f,    0.0f,  559.2f, 0.0f },
    {  556.0f,    0.0f,    0.0f, 0.0f },

    // Ceiling -- white lambert
    {    0.0f,  548.8f,    0.0f, 0.0f },
    {  556.0f,  548.8f,    0.0f, 0.0f },
    {  556.0f,  548.8f,  559.2f, 0.0f },

    {    0.0f,  548.8f,    0.0f, 0.0f },
    {  556.0f,  548.8f,  559.2f, 0.0f },
    {    0.0f,  548.8f,  559.2f, 0.0f },

    // Back wall -- white lambert
    {    0.0f,    0.0f,  559.2f, 0.0f },
    {    0.0f,  548.8f,  559.2f, 0.0f },
    {  556.0f,  548.8f,  559.2f, 0.0f },

    {    0.0f,    0.0f,  559.2f, 0.0f },
    {  556.0f,  548.8f,  559.2f, 0.0f },
    {  556.0f,    0.0f,  559.2f, 0.0f },

    // Right wall -- green lambert
    {    0.0f,    0.0f,    0.0f, 0.0f },
    {    0.0f,  548.8f,    0.0f, 0.0f },
    {    0.0f,  548.8f,  559.2f, 0.0f },

    {    0.0f,    0.0f,    0.0f, 0.0f },
    {    0.0f,  548.8f,  559.2f, 0.0f },
    {    0.0f,    0.0f,  559.2f, 0.0f },

    // Left wall -- red lambert
    {  556.0f,    0.0f,    0.0f, 0.0f },
    {  556.0f,    0.0f,  559.2f, 0.0f },
    {  556.0f,  548.8f,  559.2f, 0.0f },

    {  556.0f,    0.0f,    0.0f, 0.0f },
    {  556.0f,  548.8f,  559.2f, 0.0f },
    {  556.0f,  548.8f,    0.0f, 0.0f },

    // Short block -- white lambert
    {  130.0f,  165.0f,   65.0f, 0.0f },
    {   82.0f,  165.0f,  225.0f, 0.0f },
    {  242.0f,  165.0f,  274.0f, 0.0f },

    {  130.0f,  165.0f,   65.0f, 0.0f },
    {  242.0f,  165.0f,  274.0f, 0.0f },
    {  290.0f,  165.0f,  114.0f, 0.0f },

    {  290.0f,    0.0f,  114.0f, 0.0f },
    {  290.0f,  165.0f,  114.0f, 0.0f },
    {  240.0f,  165.0f,  272.0f, 0.0f },

    {  290.0f,    0.0f,  114.0f, 0.0f },
    {  240.0f,  165.0f,  272.0f, 0.0f },
    {  240.0f,    0.0f,  272.0f, 0.0f },

    {  130.0f,    0.0f,   65.0f, 0.0f },
    {  130.0f,  165.0f,   65.0f, 0.0f },
    {  290.0f,  165.0f,  114.0f, 0.0f },

    {  130.0f,    0.0f,   65.0f, 0.0f },
    {  290.0f,  165.0f,  114.0f, 0.0f },
    {  290.0f,    0.0f,  114.0f, 0.0f },

    {   82.0f,    0.0f,  225.0f, 0.0f },
    {   82.0f,  165.0f,  225.0f, 0.0f },
    {  130.0f,  165.0f,   65.0f, 0.0f },

    {   82.0f,    0.0f,  225.0f, 0.0f },
    {  130.0f,  165.0f,   65.0f, 0.0f },
    {  130.0f,    0.0f,   65.0f, 0.0f },

    {  240.0f,    0.0f,  272.0f, 0.0f },
    {  240.0f,  165.0f,  272.0f, 0.0f },
    {   82.0f,  165.0f,  225.0f, 0.0f },

    {  240.0f,    0.0f,  272.0f, 0.0f },
    {   82.0f,  165.0f,  225.0f, 0.0f },
    {   82.0f,    0.0f,  225.0f, 0.0f },

    // Tall block -- white lambert
    {  423.0f,  330.0f,  247.0f, 0.0f },
    {  265.0f,  330.0f,  296.0f, 0.0f },
    {  314.0f,  330.0f,  455.0f, 0.0f },

    {  423.0f,  330.0f,  247.0f, 0.0f },
    {  314.0f,  330.0f,  455.0f, 0.0f },
    {  472.0f,  330.0f,  406.0f, 0.0f },

    {  423.0f,    0.0f,  247.0f, 0.0f },
    {  423.0f,  330.0f,  247.0f, 0.0f },
    {  472.0f,  330.0f,  406.0f, 0.0f },

    {  423.0f,    0.0f,  247.0f, 0.0f },
    {  472.0f,  330.0f,  406.0f, 0.0f },
    {  472.0f,    0.0f,  406.0f, 0.0f },

    {  472.0f,    0.0f,  406.0f, 0.0f },
    {  472.0f,  330.0f,  406.0f, 0.0f },
    {  314.0f,  330.0f,  456.0f, 0.0f },

    {  472.0f,    0.0f,  406.0f, 0.0f },
    {  314.0f,  330.0f,  456.0f, 0.0f },
    {  314.0f,    0.0f,  456.0f, 0.0f },

    {  314.0f,    0.0f,  456.0f, 0.0f },
    {  314.0f,  330.0f,  456.0f, 0.0f },
    {  265.0f,  330.0f,  296.0f, 0.0f },

    {  314.0f,    0.0f,  456.0f, 0.0f },
    {  265.0f,  330.0f,  296.0f, 0.0f },
    {  265.0f,    0.0f,  296.0f, 0.0f },

    {  265.0f,    0.0f,  296.0f, 0.0f },
    {  265.0f,  330.0f,  296.0f, 0.0f },
    {  423.0f,  330.0f,  247.0f, 0.0f },

    {  265.0f,    0.0f,  296.0f, 0.0f },
    {  423.0f,  330.0f,  247.0f, 0.0f },
    {  423.0f,    0.0f,  247.0f, 0.0f },

    // Ceiling light -- emmissive
    {  343.0f,  548.6f,  227.0f, 0.0f },
    {  213.0f,  548.6f,  227.0f, 0.0f },
    {  213.0f,  548.6f,  332.0f, 0.0f },

    {  343.0f,  548.6f,  227.0f, 0.0f },
    {  213.0f,  548.6f,  332.0f, 0.0f },
    {  343.0f,  548.6f,  332.0f, 0.0f }
} };

static std::array<uint32_t, TRIANGLE_COUNT> g_mat_indices = {{
    0, 0,                          // Floor         -- white lambert
    0, 0,                          // Ceiling       -- white lambert
    0, 0,                          // Back wall     -- white lambert
    1, 1,                          // Right wall    -- green lambert
    2, 2,                          // Left wall     -- red lambert
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // Short block   -- white lambert
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // Tall block    -- white lambert
    3, 3                           // Ceiling light -- emmissive
}};



const std::array<float3, MAT_COUNT> g_emission_colors =
{ {
    {  0.0f,  0.0f,  0.0f },
    {  0.0f,  0.0f,  0.0f },
    {  0.0f,  0.0f,  0.0f },
    { 15.0f, 15.0f,  5.0f }

} };


const std::array<float3, MAT_COUNT> g_diffuse_colors =
{ {
    { 0.80f, 0.80f, 0.80f },
    { 0.05f, 0.80f, 0.05f },
    { 0.80f, 0.05f, 0.05f },
    { 0.50f, 0.00f, 0.00f }
} };

int main()
{
  int width  = 512;
  int height = 512;

  char log[2048]; // For error reporting from OptiX creation functions

  CUDA_CHECK(cudaFree(0)); //Initialize CUDA for this device on this thread
  CUcontext cuCtx = 0; //Zero means take the current context
  OptixDeviceContext context;
  OPTIX_CHECK( optixInit() );
  OPTIX_CHECK(optixDeviceContextCreate(cuCtx, 0, &context));

  //
  // copy mesh data to device
  //
  CUdeviceptr d_vertices;
  const size_t vertices_size_in_bytes = g_vertices.size() * sizeof( Vertex );
  CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_vertices ), vertices_size_in_bytes ) );
  CUDA_CHECK( cudaMemcpy(
              reinterpret_cast<void*>( d_vertices ),
              g_vertices.data(), vertices_size_in_bytes,
              cudaMemcpyHostToDevice
              ) );

  CUdeviceptr  d_mat_indices;
  const size_t mat_indices_size_in_bytes = g_mat_indices.size() * sizeof( uint32_t );
  CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_mat_indices ), mat_indices_size_in_bytes ) );
  CUDA_CHECK( cudaMemcpy(
              reinterpret_cast<void*>( d_mat_indices ),
              g_mat_indices.data(),
              mat_indices_size_in_bytes,
              cudaMemcpyHostToDevice
              ) );

  //
  // Build triangle GAS
  //
  uint32_t triangle_input_flags[MAT_COUNT] =  // One per SBT record for this build input
  {
      OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
      OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
      OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
      OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT
  };

  OptixBuildInput triangle_input                           = {};
  triangle_input.type                                      = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
  triangle_input.triangleArray.vertexFormat                = OPTIX_VERTEX_FORMAT_FLOAT3;
  triangle_input.triangleArray.vertexStrideInBytes         = sizeof( Vertex );
  triangle_input.triangleArray.numVertices                 = static_cast<uint32_t>( g_vertices.size() );
  triangle_input.triangleArray.vertexBuffers               = &d_vertices;
  triangle_input.triangleArray.flags                       = triangle_input_flags;
  triangle_input.triangleArray.numSbtRecords               = MAT_COUNT;
  triangle_input.triangleArray.sbtIndexOffsetBuffer        = d_mat_indices;
  triangle_input.triangleArray.sbtIndexOffsetSizeInBytes   = sizeof( uint32_t );
  triangle_input.triangleArray.sbtIndexOffsetStrideInBytes = sizeof( uint32_t );

  OptixAccelBuildOptions accel_options = {};
  accel_options.buildFlags             = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
  accel_options.operation              = OPTIX_BUILD_OPERATION_BUILD;

  OptixAccelBufferSizes gas_buffer_sizes;
  OPTIX_CHECK( optixAccelComputeMemoryUsage(
              context,
              &accel_options,
              &triangle_input,
              1,  // num_build_inputs
              &gas_buffer_sizes
              ) );

  CUdeviceptr d_temp_buffer;
  CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_temp_buffer ), gas_buffer_sizes.tempSizeInBytes ) );

  // non-compacted output
  CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
  size_t      compactedSizeOffset = roundUp<size_t>( gas_buffer_sizes.outputSizeInBytes, 8ull );
  CUDA_CHECK( cudaMalloc(
              reinterpret_cast<void**>( &d_buffer_temp_output_gas_and_compacted_size ),
              compactedSizeOffset + 8
              ) );

  OptixAccelEmitDesc emitProperty = {};
  emitProperty.type               = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
  emitProperty.result             = ( CUdeviceptr )( (char*)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset );

  OptixTraversableHandle gas_handle;

  OPTIX_CHECK( optixAccelBuild(
              context,
              0,                                  // CUDA stream
              &accel_options,
              &triangle_input,
              1,                                  // num build inputs
              d_temp_buffer,
              gas_buffer_sizes.tempSizeInBytes,
              d_buffer_temp_output_gas_and_compacted_size,
              gas_buffer_sizes.outputSizeInBytes,
              &gas_handle,
              &emitProperty,                      // emitted property list
              1                                   // num emitted properties
              ) );


  CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_temp_buffer ) ) );
  CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_mat_indices ) ) );

  size_t compacted_gas_size;
  CUdeviceptr d_gas_output_buffer;
  CUDA_CHECK( cudaMemcpy( &compacted_gas_size, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost ) );

  if( compacted_gas_size < gas_buffer_sizes.outputSizeInBytes )
  {
      CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_gas_output_buffer ), compacted_gas_size ) );

      // use handle as input and output
      OPTIX_CHECK( optixAccelCompact( context, 0, gas_handle, d_gas_output_buffer, compacted_gas_size, &gas_handle ) );

      CUDA_CHECK( cudaFree( (void*)d_buffer_temp_output_gas_and_compacted_size ) );
  }
  else
  {
      d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
  }


  //
  // Create module
  //
  OptixModule module = nullptr;
  OptixPipelineCompileOptions pipeline_compile_options = {};
  {
      OptixModuleCompileOptions module_compile_options = {};
      module_compile_options.maxRegisterCount     = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
      module_compile_options.optLevel             = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
      module_compile_options.debugLevel           = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

      pipeline_compile_options.usesMotionBlur        = false;
      pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
      pipeline_compile_options.numPayloadValues      = 3;
      pipeline_compile_options.numAttributeValues    = 0;
      pipeline_compile_options.exceptionFlags        = OPTIX_EXCEPTION_FLAG_NONE;  // TODO: should be OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
      pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

      //nvcc -ptx -Ipath-to-optix-sdk/include --use_fast_math myprogram.cu -o myprogram.ptx
      const std::string ptx = getPtxString( "PTX", "distributedRTGPU", nullptr );
      size_t sizeof_log = sizeof( log );

      OPTIX_CHECK_LOG( optixModuleCreateFromPTX(
                  context,
                  &module_compile_options,
                  &pipeline_compile_options,
                  ptx.c_str(),
                  ptx.size(),
                  log,
                  &sizeof_log,
                  &module
                  ) );
  }


  //
  // Create program groups, including NULL miss and hitgroups
  //
  OptixProgramGroup raygen_prog_group   = nullptr;
  OptixProgramGroup miss_prog_group     = nullptr;
  OptixProgramGroup hitgroup_prog_group = nullptr;
  {
      OptixProgramGroupOptions program_group_options   = {}; // Initialize to zeros

      OptixProgramGroupDesc raygen_prog_group_desc  = {}; //
      raygen_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
      raygen_prog_group_desc.raygen.module            = module;
      raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
      size_t sizeof_log = sizeof( log );
      OPTIX_CHECK_LOG( optixProgramGroupCreate(
                  context,
                  &raygen_prog_group_desc,
                  1,   // num program groups
                  &program_group_options,
                  log,
                  &sizeof_log,
                  &raygen_prog_group
                  ) );

      // Leave miss group's module and entryfunc name null
      OptixProgramGroupDesc miss_prog_group_desc = {};
      miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
      miss_prog_group_desc.miss.module            = module;
      miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
      sizeof_log = sizeof( log );
      OPTIX_CHECK_LOG( optixProgramGroupCreate(
                  context,
                  &miss_prog_group_desc,
                  1,   // num program groups
                  &program_group_options,
                  log,
                  &sizeof_log,
                  &miss_prog_group
                  ) );

      // Leave hit group's module and entryfunc name null
      OptixProgramGroupDesc hitgroup_prog_group_desc = {};
      hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
      hitgroup_prog_group_desc.hitgroup.moduleCH            = module;
      hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
      sizeof_log = sizeof( log );
      OPTIX_CHECK_LOG( optixProgramGroupCreate(
                  context,
                  &hitgroup_prog_group_desc,
                  1,   // num program groups
                  &program_group_options,
                  log,
                  &sizeof_log,
                  &hitgroup_prog_group
                  ) );
  }



  //
  // Link pipeline
  //
  OptixPipeline pipeline = nullptr;
  {
      OptixProgramGroup program_groups[] = { raygen_prog_group, miss_prog_group, hitgroup_prog_group };

      OptixPipelineLinkOptions pipeline_link_options = {};
      pipeline_link_options.maxTraceDepth          = 5;
      pipeline_link_options.debugLevel             = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
      pipeline_link_options.overrideUsesMotionBlur = false;
      size_t sizeof_log = sizeof( log );
      OPTIX_CHECK_LOG( optixPipelineCreate(
                  context,
                  &pipeline_compile_options,
                  &pipeline_link_options,
                  program_groups,
                  sizeof( program_groups ) / sizeof( program_groups[0] ),
                  log,
                  &sizeof_log,
                  &pipeline
                  ) );
  }


  //
  // Set up shader binding table
  //
  OptixShaderBindingTable sbt = {};
  {
      CUdeviceptr  raygen_record;
      const size_t raygen_record_size = sizeof( RayGenSbtRecord );
      CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &raygen_record ), raygen_record_size ) );
      RayGenSbtRecord rg_sbt;
      OPTIX_CHECK( optixSbtRecordPackHeader( raygen_prog_group, &rg_sbt ) );
      rg_sbt.data = {};
      rg_sbt.data.cameraPos     = make_float3( 278.0f, 273.0f, -900.0f );
      rg_sbt.data.cameraRight   = make_float3( 1.0f, 0.0f, 0.0f );
      rg_sbt.data.cameraUp      = make_float3( 0.0f, 1.0f, 0.0f );
      rg_sbt.data.cameraForward = make_float3( 0.0f, 0.0f, 1.0f );
      CUDA_CHECK( cudaMemcpy(
                  reinterpret_cast<void*>( raygen_record ),
                  &rg_sbt,
                  raygen_record_size,
                  cudaMemcpyHostToDevice
                  ) );

      CUdeviceptr miss_record;
      size_t      miss_record_size = sizeof( MissSbtRecord );
      CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &miss_record ), miss_record_size ) );
      MissSbtRecord ms_sbt;
      OPTIX_CHECK( optixSbtRecordPackHeader( miss_prog_group, &ms_sbt ) );
      ms_sbt.data = {};
      ms_sbt.data.backgroundColour = make_float3( 0.0f, 0.0f, 0.0f );
      CUDA_CHECK( cudaMemcpy(
                  reinterpret_cast<void*>( miss_record ),
                  &ms_sbt,
                  miss_record_size,
                  cudaMemcpyHostToDevice
                  ) );

      CUdeviceptr hitgroup_record;
      size_t      hitgroup_record_size = sizeof( HitGroupSbtRecord );
      CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &hitgroup_record ), hitgroup_record_size ) );
      RayGenSbtRecord hg_sbt;
      OPTIX_CHECK( optixSbtRecordPackHeader( hitgroup_prog_group, &hg_sbt ) );
      CUDA_CHECK( cudaMemcpy(
                  reinterpret_cast<void*>( hitgroup_record ),
                  &hg_sbt,
                  hitgroup_record_size,
                  cudaMemcpyHostToDevice
                  ) );

      sbt.raygenRecord                = raygen_record;
      sbt.missRecordBase              = miss_record;
      sbt.missRecordStrideInBytes     = sizeof( MissSbtRecord );
      sbt.missRecordCount             = 1;
      sbt.hitgroupRecordBase          = hitgroup_record;
      sbt.hitgroupRecordStrideInBytes = sizeof( HitGroupSbtRecord );
      sbt.hitgroupRecordCount         = 1;
  }

  CUDAOutputBuffer<uchar4> output_buffer( CUDAOutputBufferType::CUDA_DEVICE, width, height );

  //
  // launch
  //
  {
      CUstream stream;
      CUDA_CHECK( cudaStreamCreate( &stream ) );

      Params params;
      params.image       = output_buffer.map();
      params.image_width = width;
      params.handle = gas_handle;

      CUdeviceptr d_param;
      CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_param ), sizeof( Params ) ) );
      CUDA_CHECK( cudaMemcpy(
                  reinterpret_cast<void*>( d_param ),
                  &params, sizeof( params ),
                  cudaMemcpyHostToDevice
                  ) );

      OPTIX_CHECK( optixLaunch( pipeline, stream, d_param, sizeof( Params ), &sbt, width, height, 1 ) );
      CUDA_SYNC_CHECK();

      output_buffer.unmap();
  }

  int max = 0;

  uchar4 *data = output_buffer.getHostPointer();
  for (int i = 0; i < height; i++){
    for (int j = 0; j < width; j++){
      uchar4 pixelValue = data[i*width + j];
      if (max < pixelValue.x) max = pixelValue.x;
      if (max < pixelValue.y) max = pixelValue.y;
      if (max < pixelValue.z) max = pixelValue.z;
    }
  }

  std::ofstream outFile;
  outFile.open ("output.ppm");
  outFile << "P3\n";
  outFile << width << " " << height << "\n";
  outFile << max << "\n";

  for (int i = 0; i < height; i++){
    for (int j = 0; j < width; j++){
      uchar4 displayPixelValue = data[i*width + j];
      outFile << (int)displayPixelValue.x << " ";
      outFile << (int)displayPixelValue.y << " ";
      outFile << (int)displayPixelValue.z << " ";
      outFile << " ";
    }
    outFile << "\n";
  }

  outFile.close();


  //
  // Cleanup
  //
  {

      CUDA_CHECK( cudaFree( reinterpret_cast<void*>( sbt.raygenRecord       ) ) );
      CUDA_CHECK( cudaFree( reinterpret_cast<void*>( sbt.missRecordBase     ) ) );
      CUDA_CHECK( cudaFree( reinterpret_cast<void*>( sbt.hitgroupRecordBase ) ) );

      OPTIX_CHECK( optixPipelineDestroy( pipeline ) );
      OPTIX_CHECK( optixProgramGroupDestroy( hitgroup_prog_group ) );
      OPTIX_CHECK( optixProgramGroupDestroy( miss_prog_group ) );
      OPTIX_CHECK( optixProgramGroupDestroy( raygen_prog_group ) );
      OPTIX_CHECK( optixModuleDestroy( module ) );

      OPTIX_CHECK( optixDeviceContextDestroy( context ) );
  }

  return 0;
}
