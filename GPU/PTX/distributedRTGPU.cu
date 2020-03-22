//
// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

// compile using: nvcc --std c++11 -ptx -I/sw/lang/cuda_10.1.105/NVIDIA-OptiX-SDK-7.0.0-linux64/include --use_fast_math distributedRTGPU.cu -o distributedRTGPU.ptx

#include <optix.h>

#include "../distributedRTGPU.h"

//#include <sutil/vec_math.h>

extern "C" {
__constant__ Params params;
}


static __forceinline__ __device__ void trace(
        OptixTraversableHandle handle,
        float3                 ray_origin,
        float3                 ray_direction,
        float                  tmin,
        float                  tmax,
        float3*                prd
        )
{
    uint32_t p0, p1, p2;
    p0 = float_as_int( prd->x );
    p1 = float_as_int( prd->y );
    p2 = float_as_int( prd->z );
    optixTrace(
            handle,
            ray_origin,
            ray_direction,
            tmin,
            tmax,
            0.0f,                // rayTime
            OptixVisibilityMask( 1 ),
            OPTIX_RAY_FLAG_NONE,
            0,                   // SBT offset
            1,                   // SBT stride
            0,                   // missSBTIndex
            p0, p1, p2 );
    prd->x = int_as_float( p0 );
    prd->y = int_as_float( p1 );
    prd->z = int_as_float( p2 );
}

static __forceinline__ __device__ void setPayload( float3 p )
{
    optixSetPayload_0( float_as_int( p.x ) );
    optixSetPayload_1( float_as_int( p.y ) );
    optixSetPayload_2( float_as_int( p.z ) );
}


static __forceinline__ __device__ float3 getPayload()
{
    return make_float3(
            int_as_float( optixGetPayload_0() ),
            int_as_float( optixGetPayload_1() ),
            int_as_float( optixGetPayload_2() )
            );
}


__forceinline__ __device__ uchar4 make_colour( const float3&  c )
{
    return make_uchar4(
            static_cast<uint8_t>( clamp( c.x, 0.0f, 1.0f ) *255.0f ),
            static_cast<uint8_t>( clamp( c.y, 0.0f, 1.0f ) *255.0f ),
            static_cast<uint8_t>( clamp( c.z, 0.0f, 1.0f ) *255.0f ),
            255u
            );
}


extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    const RayGenData* rtData = (RayGenData*)optixGetSbtDataPointer();
    const float3      U      = rtData->cameraUp;
    const float3      V      = rtData->cameraRight;
    const float3      W      = rtData->cameraForward;
    const float2      d = 2.0f * make_float2(
            static_cast<float>( idx.x ) / static_cast<float>( dim.x ),
            static_cast<float>( idx.y ) / static_cast<float>( dim.y )
            ) - 1.0f;

    const float3 origin      = rtData->cameraPos;
    //-ve as index x is left to right and index y is top to bottom
    //whereas coordinate space x is right to left (change?) and y is bottom to top
    const float3 direction   = normalize( -1.0f*d.y * U + -1.0f*d.x * V + W );
    float3       payload_rgb = make_float3( 0.5f, 0.5f, 0.5f );
    trace( params.handle,
            origin,
            direction,
            0.00f,  // tmin
            1e16f,  // tmax
            &payload_rgb );

    params.image[idx.y * params.image_width + idx.x] = make_colour( payload_rgb );
}


extern "C" __global__ void __miss__ms()
{
    MissData* rt_data  = reinterpret_cast<MissData*>( optixGetSbtDataPointer() );
    float3    payload = getPayload();
    setPayload( rt_data->backgroundColour );
}


extern "C" __global__ void __closesthit__ch()
{
    HitGroupData* rt_data = (HitGroupData*)optixGetSbtDataPointer();

    const int    prim_idx        = optixGetPrimitiveIndex();
    const float3 ray_dir         = optixGetWorldRayDirection();
    const int    vert_idx_offset = prim_idx*3;

    const float3 v0   = make_float3( rt_data->vertices[ vert_idx_offset+0 ] );
    const float3 v1   = make_float3( rt_data->vertices[ vert_idx_offset+1 ] );
    const float3 v2   = make_float3( rt_data->vertices[ vert_idx_offset+2 ] );
    const float3 N_0  = normalize( cross( v1-v0, v2-v0 ) );

    const float3 N    = faceforward( N_0, -ray_dir, N_0 );
    const float3 P    = optixGetWorldRayOrigin() + optixGetRayTmax()*ray_dir;

    const float3 diffuseColour = rt_data->diffuse_color;

    const float  Ldist = length(params.lightPos - P );
    const float3 Ldir  = normalize(params.lightPos - P );
    const float  nDl   = dot( N, Ldir );

    float3 colour = params.lightIntensity*nDl*diffuseColour/(4.0f * M_PIf * Ldist * Ldist);

    printf("colour: %f %f %f \n", colour.x, colour.y, colour.z);

    setPayload(colour);
}
