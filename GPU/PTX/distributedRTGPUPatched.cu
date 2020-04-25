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

// compile using: nvcc --std c++11 -ptx -I/sw/lang/cuda_10.1.105/NVIDIA-OptiX-SDK-7.0.0-linux64/include --use_fast_math distributedRTGPUPatched.cu -o distributedRTGPUPatched.ptx

#include <optix.h>

#include "../distributedRTGPU.h"

//#include <sutil/vec_math.h>

extern "C" {
__constant__ Params params;
}

struct RadiancePRD
{
    // TODO: move some state directly into payload registers?
    float3   emitted;
    float3   radiance;
    float3   attenuation;
    float3   origin;
    float3   direction;
    uint32_t seed;
    int32_t  countEmitted;
    int32_t  done;
    //int32_t  pad;

};

struct Onb
{
  __forceinline__ __device__ Onb(const float3& normal)
  {
    m_normal = normal;

    if( fabs(m_normal.x) > fabs(m_normal.z) )
    {
      m_binormal.x = -m_normal.y;
      m_binormal.y =  m_normal.x;
      m_binormal.z =  0;
    }
    else
    {
      m_binormal.x =  0;
      m_binormal.y = -m_normal.z;
      m_binormal.z =  m_normal.y;
    }

    m_binormal = normalize(m_binormal);
    m_tangent = cross( m_binormal, m_normal );
  }

  __forceinline__ __device__ void inverse_transform(float3& p) const
  {
    p = p.x*m_tangent + p.y*m_binormal + p.z*m_normal;
  }

  float3 m_tangent;
  float3 m_binormal;
  float3 m_normal;
};


static __forceinline__ __device__ void* unpackPointer( uint32_t i0, uint32_t i1 )
{
    const uint64_t uptr = static_cast<uint64_t>( i0 ) << 32 | i1;
    void*           ptr = reinterpret_cast<void*>( uptr );
    return ptr;
}


static __forceinline__ __device__ void  packPointer( void* ptr, uint32_t& i0, uint32_t& i1 )
{
    const uint64_t uptr = reinterpret_cast<uint64_t>( ptr );
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}


static __forceinline__ __device__ RadiancePRD* getPRD()
{
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return reinterpret_cast<RadiancePRD*>( unpackPointer( u0, u1 ) );
}

static __forceinline__ __device__ void trace(
        OptixTraversableHandle handle,
        float3                 ray_origin,
        float3                 ray_direction,
        float                  tmin,
        float                  tmax,
        RadiancePRD*           prd
        )
{
    uint32_t u0, u1;
    packPointer( prd, u0, u1 );
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
            2,                   // SBT stride
            0,                   // missSBTIndex
            u0, u1 );
}

static __forceinline__ __device__ bool traceShadow(
        OptixTraversableHandle handle,
        float3                 ray_origin,
        float3                 ray_direction,
        float                  tmin,
        float                  tmax
        )
{
    uint32_t occluded = 0u;
    optixTrace(
            handle,
            ray_origin,
            ray_direction,
            tmin,
            tmax,
            0.0f,                    // rayTime
            OptixVisibilityMask( 1 ),
            OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
            1,                        // SBT offset
            2,                        // SBT stride
            1,                        // missSBTIndex
            occluded );
    return occluded;
}


static __forceinline__ __device__ void setPayloadOcclusion( bool occluded )
{
    optixSetPayload_0( static_cast<uint32_t>( occluded ) );
}

static __forceinline__ __device__ void cosine_sample_hemisphere(const float u1, const float u2, float3& p)
{
  // Uniformly sample disk.
  const float r   = sqrtf( u1 );
  const float phi = 2.0f*M_PIf * u2;
  p.x = r * cosf( phi );
  p.y = r * sinf( phi );

  // Project up to hemisphere.
  p.z = sqrtf( fmaxf( 0.0f, 1.0f - p.x*p.x - p.y*p.y ) );
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

    uint32_t seed = tea<4>( idx.y*params.image_width + idx.x, 0u );

    const int depth = 2;
    const int numSamples = 32;
    float3 colour = make_float3(0.0f, 0.0f, 0.0f);

    RadiancePRD prd;
    prd.seed         = seed;

    for (int sample = 0; sample < numSamples; sample++){
      int iters = 0;
      prd.emitted      = make_float3(0.f);
      prd.radiance     = make_float3(0.f);
      prd.attenuation  = make_float3(1.f);
      prd.countEmitted = true;
      prd.done = false;

      float3 origin      = rtData->cameraPos;
      //-ve as index x is left to right and index y is top to bottom
      //whereas coordinate space x is right to left (change?) and y is bottom to top
      float3 direction   = normalize( -1.0f*d.y * U + -1.0f*d.x * V + W );

      while (iters < depth && !prd.done){

        trace( params.handle,
                origin,
                direction,
                0.00f,  // tmin
                1e16f,  // tmax
                &prd );

        colour = colour + prd.radiance*prd.attenuation + prd.emitted;
        origin = prd.origin;
        direction = prd.direction;

        iters = iters+1;
      }
    }
    colour = colour/(float)numSamples;
    params.image[idx.y * params.image_width + idx.x] = make_colour(colour);
}


extern "C" __global__ void __miss__ms()
{
  MissData* rt_data  = reinterpret_cast<MissData*>( optixGetSbtDataPointer() );
  RadiancePRD* prd = getPRD();

  prd->radiance = rt_data->backgroundColour;
  prd->done      = true;
}

extern "C" __global__ void __closesthit__shadow()
{
    setPayloadOcclusion( true );
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
    const float3 P    = optixGetWorldRayOrigin() + optixGetRayTmax()*ray_dir + 0.1f*N;

    RadiancePRD* prd = getPRD();

    if (prd->countEmitted) {
      prd->emitted = rt_data->emission_color;
    }
    else {
      prd->emitted = make_float3(0.0f, 0.0f, 0.0f);
    }

    const float3 diffuseColour = rt_data->diffuse_color;

    uint32_t seed = prd->seed;
    const float z1 = rnd(seed);
    const float z2 = rnd(seed);

    //float3 w_in = 2.0f*dot(ray_dir, N) - ray_dir;
    float3 w_in;
    cosine_sample_hemisphere( z1, z2, w_in );
    Onb onb( N );
    onb.inverse_transform( w_in );
    prd->direction = w_in;
    prd->origin    = P;
    prd->attenuation = prd->attenuation*diffuseColour;
    prd->countEmitted = false;

    const float zz1 = rnd(seed);
    const float zz2 = rnd(seed);
    prd->seed = seed;

    const float3 lightPosSample = params.lightPos + params.lightV1 * zz1 + params.lightV2 * zz2;

    const float  Ldist = length(lightPosSample - P );
    const float3 Ldir  = normalize(lightPosSample - P );
    const float  nDl   = dot( N, Ldir );
    const float  LnDl  = -dot( params.lightNorm, Ldir );
    const float A = length(cross(params.lightV1, params.lightV2));

    //prd->radiance = make_float3(0.0f, 0.0f, 0.0f);
    if( nDl > 0.0f && LnDl > 0.0f )
    {
        const bool occluded = traceShadow(
            params.handle,
            P,
            Ldir,
            0.01f,         // tmin
            Ldist - 0.01f  // tmax
            );

        if( !occluded )
        {
          prd->radiance = prd->radiance + params.lightIntensity*LnDl*nDl*A/(M_PIf * Ldist * Ldist);
        }
    }
}
