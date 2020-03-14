// ======================================================================== //
// Copyright 2009-2020 Intel Corporation                                    //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include <cuda_runtime.h>
#include <cuda.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdexcept>
#include <math.h>
#include <iostream>
#include <fstream>
#include <sstream>

//Exception class used by error checking macros
class Exception : public std::runtime_error
{
 public:
     Exception( const char* msg )
         : std::runtime_error( msg )
     { }

     Exception( OptixResult res, const char* msg )
         : std::runtime_error( createMessage( res, msg ).c_str() )
     { }

 private:
     std::string createMessage( OptixResult res, const char* msg )
     {
         std::ostringstream out;
         out << optixGetErrorName( res ) << ": " << msg;
         return out.str();
     }
};

//------------------------------------------------------------------------------
//
// CUDA error-checking
//
//------------------------------------------------------------------------------

#define CUDA_CHECK( call )                                                     \
    do                                                                         \
    {                                                                          \
        cudaError_t error = call;                                              \
        if( error != cudaSuccess )                                             \
        {                                                                      \
            std::stringstream ss;                                              \
            ss << "CUDA call (" << #call << " ) failed with error: '"          \
               << cudaGetErrorString( error )                                  \
               << "' (" __FILE__ << ":" << __LINE__ << ")\n";                  \
            throw Exception( ss.str().c_str() );                        \
        }                                                                      \
    } while( 0 )


//------------------------------------------------------------------------------
//
// OptiX error-checking
//
//------------------------------------------------------------------------------

#define OPTIX_CHECK( call )                                                    \
    do                                                                         \
    {                                                                          \
        OptixResult res = call;                                                \
        if( res != OPTIX_SUCCESS )                                             \
        {                                                                      \
            std::stringstream ss;                                              \
            ss << "Optix call '" << #call << "' failed: " __FILE__ ":"         \
               << __LINE__ << ")\n";                                           \
            throw Exception( res, ss.str().c_str() );                   \
        }                                                                      \
    } while( 0 )


int main()
{

  CUDA_CHECK(cudaFree(0)); //Initialize CUDA for this device on this thread
  CUcontext cuCtx = 0; //Zero means take the current context
  OptixDeviceContext context;
  OPTIX_CHECK( optixInit() );
  OPTIX_CHECK(optixDeviceContextCreate(cuCtx, 0, &context));

  int width  = 512;
  int height = 512;
  int max = 0;

  unsigned int data[height][width][3];
  for (int i = 0; i < height; i++){
    for (int j = 0; j < width; j++){
      for(int k = 0; k < 3; k++){
        float randNum = rand()%256;
        data[i][j][k] = randNum;
        if (max < randNum) max = randNum;
      }
    }
  }

  std::ofstream outFile;
  outFile.open ("output.ppm");
  outFile << "P3\n";
  outFile << width << " " << height << "\n";
  outFile << max << "\n";

  for (int i = 0; i < height; i++){
    for (int j = 0; j < width; j++){
      for(int k = 0; k < 3; k++){
        outFile << data[i][j][k] << " ";
      }
      outFile << " ";
    }
    outFile << "\n";
  }

  outFile.close();

  OPTIX_CHECK(optixDeviceContextDestroy(context));

  return 0;
}
