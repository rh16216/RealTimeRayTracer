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

#include <optix.h>
//#include <optix_world.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <fstream>

/* -------------------------------------------------------------------------- */
void reportErrorMessage( const char* message )
{
    std::cerr << "OptiX Error: '" << message << "'\n";
#if defined(_WIN32) && defined(RELEASE_PUBLIC)
    {
        char s[2048];
        sprintf( s, "OptiX Error: %s", message );
        MessageBoxA( 0, s, "OptiX Error", MB_OK|MB_ICONWARNING|MB_SYSTEMMODAL );
    }
#endif
}


void handleError( RTcontext context, RTresult code, const char* file,
        int line)
{
    const char* message;
    char s[2048];
    rtContextGetErrorString(context, code, &message);
    sprintf(s, "%s\n(%s:%d)", message, file, line);
    reportErrorMessage( s );
}

// Default catch block
#define SUTIL_CATCH( ctx ) catch( const APIError& e ) {     \
    handleError( ctx, e.code, e.file.c_str(), e.line );     \
  }                                                                \
  catch( const std::exception& e ) {                               \
    reportErrorMessage( e.what() );                         \
    exit(1);                                                       \
  }

// Exeption to be thrown by RT_CHECK_ERROR macro
struct APIError
{
    APIError( RTresult c, const std::string& f, int l )
        : code( c ), file( f ), line( l ) {}
    RTresult     code;
    std::string  file;
    int          line;
};

// Error check/report helper for users of the C API
#define RT_CHECK_ERROR( func )                                     \
  do {                                                             \
    RTresult code = func;                                          \
    if( code != RT_SUCCESS )                                       \
      throw APIError( code, __FILE__, __LINE__ );           \
  } while(0)

int main()
{

  unsigned int version;
  RTresult result = rtGetVersion(&version);

  printf("result: %d\n", result);
  printf("version: %d\n", version);

  RTcontext context = 0;
  try{
    /* Create our objects and set state */
    RT_CHECK_ERROR(rtContextCreate( &context ));
  }SUTIL_CATCH( context )
  rtContextSetRayTypeCount( context, 0 );
  //RT_CHECK_ERROR( rtContextSetEntryPointCount( context, 1 ) );

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

  rtContextDestroy( context );

  return 0;
}
