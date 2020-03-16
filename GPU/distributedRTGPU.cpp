//compile using: g++ -std=c++0x -I$CPATH -I/sw/lang/cuda_10.1.105/NVIDIA-OptiX-SDK-7.0.0-linux64/include -o distributedRTGPU distributedRTGPU.cpp -L/sw/lang/cuda_10.1.105/lib64 -ldl -lutil -lcublas -lcudart

#include "distributedRTGPU.h"


template <typename T>
struct SbtRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenData> RayGenSbtRecord;
typedef SbtRecord<int>        MissSbtRecord;
typedef SbtRecord<int>        HitGroupSbtRecord;

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
      pipeline_compile_options.numPayloadValues      = 2;
      pipeline_compile_options.numAttributeValues    = 2;
      pipeline_compile_options.exceptionFlags        = OPTIX_EXCEPTION_FLAG_NONE;  // TODO: should be OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
      pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

      //nvcc -ptx -Ipath-to-optix-sdk/include --use_fast_math myprogram.cu -o myprogram.ptx
      const std::string ptx = getPtxString( "PTX", "draw_solid_color", nullptr );
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
      raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__draw_solid_color";
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
      OptixProgramGroup program_groups[] = { raygen_prog_group };

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
      rg_sbt.data = {0.462f, 0.725f, 0.f};
      CUDA_CHECK( cudaMemcpy(
                  reinterpret_cast<void*>( raygen_record ),
                  &rg_sbt,
                  raygen_record_size,
                  cudaMemcpyHostToDevice
                  ) );

      CUdeviceptr miss_record;
      size_t      miss_record_size = sizeof( MissSbtRecord );
      CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &miss_record ), miss_record_size ) );
      RayGenSbtRecord ms_sbt;
      OPTIX_CHECK( optixSbtRecordPackHeader( miss_prog_group, &ms_sbt ) );
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
